import os
import random
import numpy as np
import csv
import time
from absl import app
from pysc2.env import sc2_env
from pysc2.lib import actions, features

# 定義人族單位 ID
COMMAND_CENTER_ID = 18      # 指揮中心
SUPPLY_DEPOT_ID = 19        # 補給站
REFINERY_ID = 20           # 瓦斯煉油廠
BARRACKS_ID = 21           # 兵營
ENGINEERING_BAY_ID = 22     # 電機工程所
BARRACKS_TECHLAB_ID = 37    # 兵營科技實驗室
SCV_ID = 45                # 工兵 (SCV)
MARAUDER_ID = 51           # 掠奪者
MINERAL_FIELD_ID = 341      # 晶體礦
GEYSER_ID = 342            # 瓦斯湧泉
FACTORY_ID = 27            # 工廠
STARPORT_ID = 28           # 星際港
ARMORY_ID = 29             # 兵工廠
FUSION_CORE_ID = 30        # 核融合核心
GHOST_ACADEMY_ID = 26       # 幽靈特務學院
ORBITAL_COMMAND_ID = 132    # 軌道指揮部

BASE_LOCATION_CODE = 0
PLANETARY_FORTRESS_ID = 130

# =========================================================
# 📊 數據收集器: 紀錄資源與訓練狀態
# =========================================================
class DataCollector:
    def __init__(self):
        # 確保 logs 資料夾存在
        if not os.path.exists("logs"):
            os.makedirs("logs")
        # 以目前時間戳記建立檔案名稱
        self.filename = f"logs/terran_log_{int(time.time())}.csv"
        # 寫入 CSV 標題列
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "Minerals", "Vespene", "Workers", "Ideal", "Barracks", "Action_ID"])

    def log_step(self, time_val, minerals, vespene, workers, ideal, barracks, action_id):
        """ 將當前的遊戲數據寫入 CSV """
        display_time = float(time_val[0]) if hasattr(time_val, "__len__") else float(time_val)
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([round(display_time, 2), minerals, vespene, workers, ideal, barracks, action_id])

# =========================================================
# 🧠 生產大腦: 整合所有功能與修正
# =========================================================
class ProductionAI:
    def __init__(self):
        self.collector = DataCollector() #各項數據（如礦物、瓦斯、人口、執行的動作
        self.refinery_target = None   # 用來儲存瓦斯湧泉（Geyser）的座標點
        self.cc_x_screen = 42         # 預設螢幕中心點 X
        self.cc_y_screen = 42         # 預設螢幕中心點 Y
        self.gas_workers_assigned = 0 # 紀錄目前已經指派了多少工兵去採集瓦斯
        self.active_parameter = 1     # 目前選定的建築位置編號
        self.locked_action = None     # 用於處理需要多步驟完成的動作（如：選SCV -> 蓋建築）
        self.lock_timer = 0           # 鎖定計時器，避免 AI 卡死
        self.base_location_code = 0
        self.locked_target = None
        self.cc_is_bound = False  # 追蹤主堡是否已編隊

    #檢查當前選取狀態中是否包含工兵。
    def _is_scv_selected(self, obs):
        # 檢查單一選取：使用 len() 判斷是否有選中單位
        if len(obs.observation.single_select) > 0:
            return obs.observation.single_select[0].unit_type == SCV_ID
        
        # 檢查複數選取：同樣使用 len()
        if len(obs.observation.multi_select) > 0:
            # 只要選取的清單中包含任何一個工兵，就視為已選中
            return any(u.unit_type == SCV_ID for u in obs.observation.multi_select)
            
        return False
    
    #尋找畫面上所有指定 ID 建築的中心點。
    def _find_units_centers(self, unit_type, unit_id):
        y, x = (unit_type == unit_id).nonzero()
        if not x.any(): return []
        
        centers = []
        temp_x, temp_y = list(x), list(y)
        
        # 針對掠奪者 (MARAUDER_ID) 稍微縮小判定半徑，避免將相鄰的單位合併
        # 一般步兵單位在 84x84 解析度下大約佔 3-5 個像素寬度
        radius = 5 if unit_id == MARAUDER_ID else 12 

        while temp_x:
            bx, by = temp_x[0], temp_y[0]
            # 找到與當前點距離在 radius 內的所有點
            mask = (np.abs(np.array(temp_x) - bx) < radius) & (np.abs(np.array(temp_y) - by) < radius)
            
            # 計算這個群體的中心點
            cluster_x = np.array(temp_x)[mask]
            cluster_y = np.array(temp_y)[mask]
            center = (int(np.mean(cluster_x)), int(np.mean(cluster_y)))
            
            # 估算這坨像素包含了幾隻單位
            # 掠奪者在 84x84 下大約佔 10-15 個像素點
            pixels_count = len(cluster_x)
            estimated_units = max(1, int(round(pixels_count / 12.0))) if unit_id == MARAUDER_ID else 1
            
            # 根據估算數量，將同一個中心點加入多次 (為了後續點擊時能對著同一坨擠在一起的兵多點幾次)
            # 或者稍微加入一點點偏移量
            if unit_id == MARAUDER_ID and estimated_units > 1:
                for i in range(estimated_units):
                     # 稍微偏移，模擬點擊人群中不同位置
                    offset_x = random.randint(-2, 2)
                    offset_y = random.randint(-2, 2)
                    centers.append((np.clip(center[0] + offset_x, 0, 83), np.clip(center[1] + offset_y, 0, 83)))
            else:
                centers.append(center)

            # 將已處理過的點移除
            temp_x = [px for i, px in enumerate(temp_x) if not mask[i]]
            temp_y = [py for i, py in enumerate(temp_y) if not mask[i]]
            
        return centers

    def get_action(self, obs, action_id, parameter=None):
        # --- 處理鎖定與超時 ---
        if self.locked_action is not None:
            self.lock_timer += 1
            if self.lock_timer > 6:
                self.locked_target = None 
                self.locked_action = None
                self.lock_timer = 0
            else:
                action_id = self.locked_action
                # 【關鍵修正】鎖定期間停止更新 active_parameter，維持最初的奇偶數意圖
        else:
            if parameter is not None: 
                self.active_parameter = parameter
        
        # 定義必要變數，避免 NameError
        unit_type = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
        player = obs.observation.player
        available = obs.observation.available_actions
        # --- 基地位置環境變數定義 ---
        # 1. 獲取小地圖上的玩家相對位置 (1 代表自己)
        player_relative_minimap = obs.observation.feature_minimap[features.MINIMAP_FEATURES.player_relative.index]
        y_mini, x_mini = (player_relative_minimap == 1).nonzero()
        
        # 2. 判斷基地在全圖的象限 (0: 左上, 1: 右下)
        # 透過小地圖平均座標是否大於中心點 (32) 來判斷
        if x_mini.any() and y_mini.any():
            base_x_mini, base_y_mini = x_mini.mean(), y_mini.mean()
            self.base_location_code = 1 if (base_x_mini > 32 and base_y_mini > 32) else 0
        else:
            self.base_location_code = 0 # 預設左上

        # 3. 獲取指揮中心在「當前螢幕」的精確中心座標
        cc_centers = []
        for cid in [COMMAND_CENTER_ID, ORBITAL_COMMAND_ID, PLANETARY_FORTRESS_ID]:
            centers = self._find_units_centers(unit_type, cid)
            if centers:
                cc_centers = centers
                break

        if cc_centers:
            self.cc_x_screen, self.cc_y_screen = cc_centers[0]

        # --- 🎯 自由位置坐標系 ---
        grid_size = 8
        p_idx = max(0, self.active_parameter - 1) % 64
        row = p_idx // 8
        col = p_idx % 8
        
        # 將 0-7 的網格轉化為相對於主堡的偏移 (-35 到 +35)
        offset_x = (col - 3.5) * 10 
        offset_y = (row - 3.5) * 10
        
        # 最終座標 = 主堡中心 + 偏移量
        tx = self.cc_x_screen + offset_x
        ty = self.cc_y_screen + offset_y
        
        grid_pos = (np.clip(tx, 5, 78), np.clip(ty, 5, 78)) # 留一點邊界避免點出螢幕外

       
        # [Action 1] 建造補給站 (100 M)
        if action_id == 1:
            if player.minerals < 100:
                self.locked_action = None
                return actions.FUNCTIONS.no_op()
            
            # A. 按鈕出現，直接蓋下去
            if actions.FUNCTIONS.Build_SupplyDepot_screen.id in available:
                self.locked_action = None 
                self.lock_timer = 0
                return actions.FUNCTIONS.Build_SupplyDepot_screen("now", grid_pos)

            # 這樣可以讓遊戲引擎有時間把建築選單顯示出來
            if self._is_scv_selected(obs):
                self.locked_action = 1
                return actions.FUNCTIONS.no_op() # 靜止等待按鈕加載

            # C. 還沒選到人
            self.locked_action = 1 
            return self._select_scv_prioritized(obs, unit_type, available)
         
        # [Action 2] 建造兵營 (150 M)
        elif action_id == 2:
            if player.minerals < 150:
                self.locked_action = None
                return actions.FUNCTIONS.no_op()
            
            if actions.FUNCTIONS.Build_Barracks_screen.id in available:
                self.locked_action = None 
                self.lock_timer = 0
                return actions.FUNCTIONS.Build_Barracks_screen("now", grid_pos)
            if self._is_scv_selected(obs):
                self.locked_action = 2
                return actions.FUNCTIONS.no_op()

            self.locked_action = 2
            return self._select_scv_prioritized(obs, unit_type, available)
        
        # [Action 3] 建造工廠 (150 M, 100 V)
        elif action_id == 3:
            if player.minerals < 150 or player.vespene < 100:
                self.locked_action = None
                return actions.FUNCTIONS.no_op()
            
            # 1. 檢查建築按鈕是否已出現
            if actions.FUNCTIONS.Build_Factory_screen.id in available:
                self.locked_action = None 
                self.lock_timer = 0
                return actions.FUNCTIONS.Build_Factory_screen("now", grid_pos)
            
            # 2. 選中工兵後，嘗試打開高級選單 (安全名稱查找)
            if self._is_scv_selected(obs):
                adv_menu = None
                for name in ["Build_Advanced_quick", "Build_Menu_Advanced_quick", "Build_Advanced_Terran_quick"]:
                    try:
                        adv_menu = getattr(actions.FUNCTIONS, name)
                        break
                    except (KeyError, AttributeError): continue
                
                if adv_menu and adv_menu.id in available:
                    self.locked_action = 3
                    return adv_menu("now")
                return actions.FUNCTIONS.no_op()

            self.locked_action = 3
            return self._select_scv_prioritized(obs, unit_type, available)

        # [Action 4] 建造星際港 (150 M, 100 V)
        elif action_id == 4:
            if player.minerals < 150 or player.vespene < 100:
                self.locked_action = None
                return actions.FUNCTIONS.no_op()
            
            if actions.FUNCTIONS.Build_Starport_screen.id in available:
                self.locked_action = None 
                self.lock_timer = 0
                return actions.FUNCTIONS.Build_Starport_screen("now", grid_pos)
            
            if self._is_scv_selected(obs):
                adv_menu = None
                for name in ["Build_Advanced_quick", "Build_Menu_Advanced_quick"]:
                    try:
                        adv_menu = getattr(actions.FUNCTIONS, name)
                        break
                    except (KeyError, AttributeError): continue
                
                if adv_menu and adv_menu.id in available:
                    self.locked_action = 4
                    return adv_menu("now")
                return actions.FUNCTIONS.no_op()

            self.locked_action = 4
            return self._select_scv_prioritized(obs, unit_type, available)

        # [Action 5] 建造核融合核心 (150 M, 150 V)
        elif action_id == 5:
            if player.minerals < 150 or player.vespene < 150:
                self.locked_action = None
                return actions.FUNCTIONS.no_op()
            
            # 1. 檢查建築按鈕是否已出現
            if actions.FUNCTIONS.Build_FusionCore_screen.id in available:
                self.locked_action = None 
                self.lock_timer = 0
                return actions.FUNCTIONS.Build_FusionCore_screen("now", grid_pos)
            
            # 2. 選中工兵後，安全嘗試打開高級選單
            if self._is_scv_selected(obs):
                adv_menu = None
                for name in ["Build_Advanced_quick", "Build_Menu_Advanced_quick"]:
                    try:
                        adv_menu = getattr(actions.FUNCTIONS, name)
                        break
                    except (KeyError, AttributeError): continue
                
                if adv_menu and adv_menu.id in available:
                    self.locked_action = 5
                    return adv_menu("now")
                return actions.FUNCTIONS.no_op()

            self.locked_action = 5
            return self._select_scv_prioritized(obs, unit_type, available)
        
        # [Action 6] 建造指揮中心 (400 M)
        elif action_id == 6:
            if player.minerals < 400 :
                self.locked_action = None
                return actions.FUNCTIONS.no_op()
            if actions.FUNCTIONS.Build_CommandCenter_screen.id in available:
                self.locked_action = None 
                self.lock_timer = 0
                return actions.FUNCTIONS.Build_CommandCenter_screen("now", grid_pos)
            if self._is_scv_selected(obs):
                self.locked_action = 6
                return actions.FUNCTIONS.no_op()
            self.locked_action = 6
            return self._select_scv_prioritized(obs, unit_type, available)

        # [Action 7] 建造電機工程所 (125 M)
        elif action_id == 7:
            if player.minerals < 125 :
                self.locked_action = None
                return actions.FUNCTIONS.no_op()
            if actions.FUNCTIONS.Build_EngineeringBay_screen.id in available:
                self.locked_action = None 
                self.lock_timer = 0
                return actions.FUNCTIONS.Build_EngineeringBay_screen("now", grid_pos)
            if self._is_scv_selected(obs):
                self.locked_action = 7
                return actions.FUNCTIONS.no_op()
            self.locked_action = 7
            return self._select_scv_prioritized(obs, unit_type, available)

        # [Action 8] 建造感應塔 (125 M, 50 V)
        elif action_id == 8:
            if player.minerals < 125 or player.vespene < 50 :
                self.locked_action = None
                return actions.FUNCTIONS.no_op()
            if actions.FUNCTIONS.Build_SensorTower_screen.id in available:
                self.locked_action = None 
                self.lock_timer = 0
                return actions.FUNCTIONS.Build_SensorTower_screen("now", grid_pos)
            if self._is_scv_selected(obs):
                self.locked_action = 8
                return actions.FUNCTIONS.no_op()
            self.locked_action = 8
            return self._select_scv_prioritized(obs, unit_type, available)

        # [Action 9] 建造幽靈特務學院 (150 M, 50 V)
        elif action_id == 9:
            if player.minerals < 150 or player.vespene < 50:
                self.locked_action = None
                return actions.FUNCTIONS.no_op()
            
            # 1. 檢查建築按鈕是否已出現
            if actions.FUNCTIONS.Build_GhostAcademy_screen.id in available:
                self.locked_action = None 
                self.lock_timer = 0
                return actions.FUNCTIONS.Build_GhostAcademy_screen("now", grid_pos)
            
            # 2. 選中工兵後，安全嘗試打開高級選單
            if self._is_scv_selected(obs):
                adv_menu = None
                # 嘗試所有可能的 API 名稱以避開 KeyError
                for name in ["Build_Advanced_quick", "Build_Menu_Advanced_quick", "Build_Advanced_Terran_quick"]:
                    try:
                        adv_menu = getattr(actions.FUNCTIONS, name)
                        break
                    except (KeyError, AttributeError): continue
                
                if adv_menu and adv_menu.id in available:
                    self.locked_action = 9
                    return adv_menu("now") # 打開選單不帶座標
                return actions.FUNCTIONS.no_op()

            self.locked_action = 9
            return self._select_scv_prioritized(obs, unit_type, available)

        # [Action 10] 建造兵工廠 (150 M, 100 V)
        elif action_id == 10:
            if player.minerals < 150 or player.vespene < 100:
                self.locked_action = None
                return actions.FUNCTIONS.no_op()
            
            # 1. 檢查建築按鈕是否已出現
            if actions.FUNCTIONS.Build_Armory_screen.id in available:
                self.locked_action = None 
                self.lock_timer = 0
                return actions.FUNCTIONS.Build_Armory_screen("now", grid_pos)
            
            # 2. 開啟高級選單
            if self._is_scv_selected(obs):
                adv_menu = None
                for name in ["Build_Advanced_quick", "Build_Menu_Advanced_quick"]:
                    try:
                        adv_menu = getattr(actions.FUNCTIONS, name)
                        break
                    except (KeyError, AttributeError): continue
                
                if adv_menu and adv_menu.id in available:
                    self.locked_action = 10
                    return adv_menu("now")
                return actions.FUNCTIONS.no_op()

            self.locked_action = 10
            return self._select_scv_prioritized(obs, unit_type, available)
        
        # [Action 11] 建造瓦斯廠
        elif action_id == 11:
            if player.minerals < 75:
                self.locked_action = None
                return actions.FUNCTIONS.no_op()
            if actions.FUNCTIONS.Build_Refinery_screen.id in available:
                self.locked_action = None 
                self.lock_timer = 0
                self.refinery_target = self._find_geyser(unit_type)
                if self.refinery_target:
                    # 這裡會回傳湧泉的中心座標 (x, y)
                    return actions.FUNCTIONS.Build_Refinery_screen("now", self.refinery_target)
            self.locked_action = 11
            return self._select_scv_prioritized(obs, unit_type, available)
        
        # [Action 12] 建造飛彈砲台 (100 M)
        elif action_id == 12:
            if player.minerals < 100:
                self.locked_action = None
                return actions.FUNCTIONS.no_op()
            if actions.FUNCTIONS.Build_MissileTurret_screen.id in available:
                self.locked_action = None 
                self.lock_timer = 0
                return actions.FUNCTIONS.Build_MissileTurret_screen("now", grid_pos)
            if self._is_scv_selected(obs):
                self.locked_action = 12
                return actions.FUNCTIONS.no_op()
            self.locked_action = 12
            return self._select_scv_prioritized(obs, unit_type, available)

        # [Action 13] 建造碉堡 (100 M)
        elif action_id == 13:
            if player.minerals < 100:
                self.locked_action = None
                return actions.FUNCTIONS.no_op()
            if actions.FUNCTIONS.Build_Bunker_screen.id in available:
                self.locked_action = None 
                self.lock_timer = 0
                return actions.FUNCTIONS.Build_Bunker_screen("now", grid_pos)
            if self._is_scv_selected(obs):
                self.locked_action = 13
                return actions.FUNCTIONS.no_op()
            self.locked_action = 13
            return self._select_scv_prioritized(obs, unit_type, available)
        
        # --- [Action 14-32] 單位生產指令集 ---

        # [Action 14] 製造 SCV 
        elif action_id == 14:
            if player.minerals >= 200 and actions.FUNCTIONS.Train_SCV_quick.id in available:
                return actions.FUNCTIONS.Train_SCV_quick("now")
            
            # 【關鍵】如果正在鎖定蓋房子，不要去點主堡，否則工兵選取會消失
            if self.locked_action is not None:
                return actions.FUNCTIONS.no_op()
                
            return self._select_unit(unit_type, COMMAND_CENTER_ID)

        # [Action 15] 製造礦騾 (MULE) - 修正後的魯棒寫法
        elif action_id == 15:
            mule_action = None
            # 嘗試兩種常見的 pysc2 動作名稱
            for act_name in ["Call_OrbitalCommand_Mule_screen", "Effect_OrbitalCommand_Mule_screen"]:
                try:
                    mule_action = getattr(actions.FUNCTIONS, act_name)
                    break # 找到就跳出
                except KeyError:
                    continue

            if mule_action and mule_action.id in available:
                y_m, x_m = (unit_type == MINERAL_FIELD_ID).nonzero()
                if x_m.any():
                    target = (int(x_m.mean()), int(y_m.mean()))
                    return mule_action("now", target)
            return self._select_unit(unit_type, ORBITAL_COMMAND_ID)

        # [Action 16] 製造陸戰隊 (Marine) - 50 M
        elif action_id == 16:
            # 1. 如果資源不足，直接放棄並解除鎖定，避免卡死
            if player.minerals < 50:
                self.locked_action = None
                return actions.FUNCTIONS.no_op()

            # 2. 如果生產按鈕已經出現，直接點擊生產，並「解除鎖定」
            if actions.FUNCTIONS.Train_Marine_quick.id in available:
                self.locked_action = None 
                self.lock_timer = 0
                return actions.FUNCTIONS.Train_Marine_quick("now")
            
            # 3. 檢查目前是否已經選中兵營
            is_barracks_selected = any(u.unit_type == BARRACKS_ID for u in obs.observation.single_select) or \
                                   any(u.unit_type == BARRACKS_ID for u in obs.observation.multi_select)
            
            if is_barracks_selected:
                # 已經選中了兵營，但按鈕還沒出來 (可能 UI 還在加載)
                self.locked_action = 16 # 繼續維持鎖定
                return actions.FUNCTIONS.no_op() # 原地等待按鈕加載
                
            # 4. 還沒選中兵營：去畫面上隨機點擊一個兵營，並「啟動鎖定」
            self.locked_action = 16
            return self._select_unit(unit_type, BARRACKS_ID)

        # [Action 17] 製造死神 (Reaper) - 50 M, 50 V
        elif action_id == 17:
            if player.minerals >= 50 and player.vespene >= 50 and actions.FUNCTIONS.Train_Reaper_quick.id in available:
                return actions.FUNCTIONS.Train_Reaper_quick("now")
            return self._select_unit(unit_type, BARRACKS_ID)

        
        # [Action 18] 製造掠奪者 (加入鎖定機制)
        elif action_id == 18:
            # 1. 如果生產按鈕已經出現，直接點擊生產，並「解除鎖定」
            if actions.FUNCTIONS.Train_Marauder_quick.id in available:
                self.locked_action = None 
                self.lock_timer = 0
                return actions.FUNCTIONS.Train_Marauder_quick("now")
            
            # 2. 檢查目前是否已經選中兵營
            is_barracks_selected = any(u.unit_type == BARRACKS_ID for u in obs.observation.single_select) or \
                                   any(u.unit_type == BARRACKS_ID for u in obs.observation.multi_select)
            
            if is_barracks_selected:
                # 已經選中了兵營，但按鈕還沒出來
                # (可能是因為資源不足，或是點到了沒有科技實驗室的兵營)
                self.locked_action = 18 # 繼續維持鎖定
                return actions.FUNCTIONS.no_op() # 原地等待按鈕加載或超時放棄
                
            # 3. 還沒選中兵營：去畫面上隨機點擊一個兵營，並「啟動鎖定」
            self.locked_action = 18
            return self._select_unit(unit_type, BARRACKS_ID)
        
        # [Action 19] 製造幽靈特務 (Ghost) - 150 M, 125 V
        elif action_id == 19:
            if player.minerals >= 150 and player.vespene >= 125 and actions.FUNCTIONS.Train_Ghost_quick.id in available:
                return actions.FUNCTIONS.Train_Ghost_quick("now")
            return self._select_unit(unit_type, BARRACKS_ID)

        # [Action 20] 製造惡狼 (Hellion) - 100 M
        elif action_id == 20:
            if player.minerals >= 100 and actions.FUNCTIONS.Train_Hellion_quick.id in available:
                return actions.FUNCTIONS.Train_Hellion_quick("now")
            return self._select_unit(unit_type, FACTORY_ID)

        # [Action 21] 製造戰狼 (Hellbat) - 100 M (需兵工廠)
        elif action_id == 21:
            if player.minerals >= 100 and actions.FUNCTIONS.Train_Hellbat_quick.id in available:
                return actions.FUNCTIONS.Train_Hellbat_quick("now")
            return self._select_unit(unit_type, FACTORY_ID)

        # [Action 22] 製造寡婦詭雷 (Widow Mine) - 75 M, 25 V
        elif action_id == 22:
            if player.minerals >= 75 and player.vespene >= 25 and actions.FUNCTIONS.Train_WidowMine_quick.id in available:
                return actions.FUNCTIONS.Train_WidowMine_quick("now")
            return self._select_unit(unit_type, FACTORY_ID)

        # [Action 23] 製造工程坦克 (Siege Tank) - 修正選取邏輯
        elif action_id == 23:
            # 1. 資源與按鈕檢查
            if player.minerals >= 150 and player.vespene >= 125 and actions.FUNCTIONS.Train_SiegeTank_quick.id in available:
                return actions.FUNCTIONS.Train_SiegeTank_quick("now")
            
            # 2. 如果沒按鈕，我們需要選取「有科技實驗室」的軍工廠
            # 這裡簡單化處理：先獲取實驗室的位置，並點擊它左邊一點點（通常是軍工廠主體）
            techlab_y, techlab_x = (unit_type == 38).nonzero() # 38 是軍工廠科技實驗室的 ID
            if techlab_x.any():
                # 點擊實驗室中心偏左 10 像素的位置（嘗試選中本體）
                target = (max(0, int(techlab_x.mean()) - 10), int(techlab_y.mean()))
                return actions.FUNCTIONS.select_point("select", target)
            
            # 3. 如果連實驗室都沒看到，才隨機選工廠
            return self._select_unit(unit_type, FACTORY_ID)
        
        # [Action 24] 製造颶風飛彈車 (Cyclone) - 150 M, 100 V
        elif action_id == 24:
            # 1. 資源與按鈕檢查
            if player.minerals >= 150 and player.vespene >= 100 and actions.FUNCTIONS.Train_Cyclone_quick.id in available:
                return actions.FUNCTIONS.Train_Cyclone_quick("now")
            
            # 2. 核心修正：使用 Action 23 的成功邏輯，尋找科技實驗室 (ID 38)
            techlab_y, techlab_x = (unit_type == 38).nonzero() 
            if techlab_x.any():
                # 點擊實驗室中心偏左 12 像素的位置，確保選中本體
                target = (max(0, int(techlab_x.mean()) - 12), int(techlab_y.mean()))
                return actions.FUNCTIONS.select_point("select", target)
            
            # 3. 若沒看到實驗室，則退而求其次點擊軍工廠中心
            factory_centers = self._find_units_centers(unit_type, FACTORY_ID)
            if factory_centers:
                return actions.FUNCTIONS.select_point("select", random.choice(factory_centers))
                
            return actions.FUNCTIONS.no_op()

        # [Action 25] 製造雷神 (Thor) - 300 M, 200 V
        elif action_id == 25:
            if player.minerals >= 300 and player.vespene >= 200 and actions.FUNCTIONS.Train_Thor_quick.id in available:
                return actions.FUNCTIONS.Train_Thor_quick("now")
            return self._select_unit(unit_type, FACTORY_ID)

        # [Action 26] 製造維京戰機 (Viking) - 150 M, 75 V
        elif action_id == 26:
            if player.minerals >= 150 and player.vespene >= 75 and actions.FUNCTIONS.Train_VikingFighter_quick.id in available:
                return actions.FUNCTIONS.Train_VikingFighter_quick("now")
            return self._select_unit(unit_type, STARPORT_ID)

        # [Action 27] 製造醫療艇 (Medivac) - 100 M, 100 V
        elif action_id == 27:
            if player.minerals >= 100 and player.vespene >= 100 and actions.FUNCTIONS.Train_Medivac_quick.id in available:
                return actions.FUNCTIONS.Train_Medivac_quick("now")
            return self._select_unit(unit_type, STARPORT_ID)

        # [Action 28] 製造解放者 (Liberator) - 150 M, 150 V
        elif action_id == 28:
            if player.minerals >= 150 and player.vespene >= 150 and actions.FUNCTIONS.Train_Liberator_quick.id in available:
                return actions.FUNCTIONS.Train_Liberator_quick("now")
            return self._select_unit(unit_type, STARPORT_ID)

        # [Action 29] 製造渡鴉 (強化指令版)
        elif action_id == 29:
            raven_act = self._get_safe_func("Train_Raven_quick")
            if raven_act == actions.FUNCTIONS.no_op:
                raven_act = self._get_safe_func("Train_Raven_Starport_quick")

            # A. 如果按鈕可用
            if raven_act.id in available:
                if player.minerals >= 100 and player.vespene >= 150:
                    self.locked_action = None; self.locked_target = None; self.lock_timer = 0
                    return raven_act("now")
                else:
                    # 【關鍵修正】資源不足時「不要解鎖」，繼續保持選取直到超時或資源足夠
                    return actions.FUNCTIONS.no_op()

            # B. 鎖定等待期間
            if self.locked_action == 29 and self.locked_target is not None:
                return actions.FUNCTIONS.no_op()

            # C. 執行選取
            techlab_centers = self._find_units_centers(unit_type, 39)
            if techlab_centers:
                self.locked_target = (np.clip(techlab_centers[0][0] - 12, 0, 83), techlab_centers[0][1])
                self.locked_action = 29; self.lock_timer = 1
                return actions.FUNCTIONS.select_point("select", self.locked_target)
            
            self.locked_action = None
            return actions.FUNCTIONS.no_op()
        
        # [Action 30] 製造戰巡艦 (Battlecruiser) - 400 M, 300 V
        elif action_id == 30:
            # 1. 資源檢查 (如果資源不足，直接放棄並解鎖)
            if player.minerals < 400 or player.vespene < 300:
                self.locked_action = None; self.locked_target = None
                return actions.FUNCTIONS.no_op()

            # 2. 獲取按鈕
            train_act = self._get_safe_func("Train_Battlecruiser_quick")
            
            # 3. 如果按鈕已經在畫面上，直接生產
            if train_act.id in available:
                self.locked_action = None; self.locked_target = None; self.lock_timer = 0
                return train_act("now")
            
            # 4. 如果正在鎖定等待按鈕加載
            if self.locked_action == 30 and self.locked_target is not None:
                return actions.FUNCTIONS.no_op()

            # 5. 核心邏輯：尋找星際港科技實驗室 (ID 39)
            techlab_centers = self._find_units_centers(unit_type, 39)
            if techlab_centers:
                # 點擊第一個實驗室左側 12 像素處 (選中星際港主體)
                self.locked_target = (np.clip(techlab_centers[0][0] - 12, 0, 83), techlab_centers[0][1])
                self.locked_action = 30; self.lock_timer = 1
                return actions.FUNCTIONS.select_point("select", self.locked_target)
            
            # 6. 若沒看到實驗室，說明前提未達成，解鎖讓 AI 執行 Action 36
            self.locked_action = None
            return actions.FUNCTIONS.no_op()

        # [Action 31] 製造女妖轟炸機 (Banshee) - 150 M, 100 V
        elif action_id == 31:
            if player.minerals >= 150 and player.vespene >= 100 and actions.FUNCTIONS.Train_Banshee_quick.id in available:
                return actions.FUNCTIONS.Train_Banshee_quick("now")
            return self._select_unit(unit_type, STARPORT_ID)

        # [Action 32] 升級為行星要塞 (Planetary Fortress) - 150 M, 150 V
        elif action_id == 32:
            if player.minerals >= 150 and player.vespene >= 150 and actions.FUNCTIONS.Morph_PlanetaryFortress_quick.id in available:
                return actions.FUNCTIONS.Morph_PlanetaryFortress_quick("now")
            return self._select_unit(unit_type, COMMAND_CENTER_ID)
        
        # [Action 33] 補給站上升或下降 (自動切換)
        elif action_id == 33:
            if actions.FUNCTIONS.Morph_SupplyDepot_Lower_quick.id in available:
                return actions.FUNCTIONS.Morph_SupplyDepot_Lower_quick("now")
            if actions.FUNCTIONS.Morph_SupplyDepot_Raise_quick.id in available:
                return actions.FUNCTIONS.Morph_SupplyDepot_Raise_quick("now")
            return self._select_unit(unit_type, SUPPLY_DEPOT_ID)

        # [Action 34] 兵營升級 (優化穩定版 - 補上鎖定機制)
        elif action_id == 34:
            is_tech_lab = 1 #(目前強制蓋科技實驗室)
            
            # 獲取動
            if is_tech_lab:
                action = getattr(actions.FUNCTIONS, "Build_TechLab_quick", None)
                if not action: action = getattr(actions.FUNCTIONS, "Build_TechLab_Barracks_quick", None)
                req_m, req_v = 50, 25
            else:
                action = getattr(actions.FUNCTIONS, "Build_Reactor_quick", None)
                if not action: action = getattr(actions.FUNCTIONS, "Build_Reactor_Barracks_quick", None)
                req_m, req_v = 50, 50

            # 1. 執行升級 (若按鈕已出現且資源足夠)
            if action and action.id in available and player.minerals >= req_m and player.vespene >= req_v:
                self.locked_action = None # 動作成功，解除鎖定
                self.lock_timer = 0
                return action("now")
            
            # 2. 如果正在鎖定等待按鈕出現 (這一步是原本漏掉的)
            if self.locked_action == 34:
                return actions.FUNCTIONS.no_op()
            
            # 3. 若還沒選到兵營，精確選取兵營並「啟動鎖定」
            barracks_centers = self._find_units_centers(unit_type, BARRACKS_ID)
            if barracks_centers:
                self.locked_action = 34 # 👉 關鍵修正：鎖定動作，防止 AI 分心
                self.lock_timer = 1
                return actions.FUNCTIONS.select_point("select", random.choice(barracks_centers))
            
            return actions.FUNCTIONS.no_op()
        
        # [Action 35] 軍工廠升級 (修正版：增加掛件偵測)
        elif action_id == 35:
            # 1. 判定要蓋哪種掛件
            is_tech_lab = (self.active_parameter % 2 == 1)
            action_name = "Build_TechLab_quick" if is_tech_lab else "Build_Reactor_quick"
            # 備用名稱 (部分版本使用特定名稱)
            alt_name = "Build_TechLab_Factory_quick" if is_tech_lab else "Build_Reactor_Factory_quick"
            
            action = self._get_safe_func(action_name)
            if action == actions.FUNCTIONS.no_op:
                action = self._get_safe_func(alt_name)

            # 2. 資源檢查
            req_m, req_v = (50, 25) if is_tech_lab else (50, 50)
            if player.minerals < req_m or player.vespene < req_v:
                return actions.FUNCTIONS.no_op()

            # 3. 如果按鈕可用，直接升級
            if action.id in available:
                return action("now")

            # 4. 如果沒看到按鈕，檢查畫面上是否已經有實驗室 (ID: 38) 或反應爐 (ID: 40)
            # 如果已經有掛件了，就不要再點工廠，直接結束動作
            addon_y, addon_x = ((unit_type == 38) | (unit_type == 40)).nonzero()
            if addon_x.any():
                # 簡單判定：如果掛件數量等於工廠數量，代表都滿了
                factory_centers = self._find_units_centers(unit_type, FACTORY_ID)
                if len(addon_x) >= len(factory_centers) * 10: # 像素點粗估
                    return actions.FUNCTIONS.no_op()

            # 5. 精確選取軍工廠
            return self._select_unit(unit_type, FACTORY_ID)
        
        # [Action 36] 星際港升級 (意圖固定版)
        elif action_id == 36:
            # 受 get_action 開頭的保護，這裏的 is_tech_lab 在鎖定期間不會變動
            is_tech_lab = (self.active_parameter % 2 == 1)
            req_m, req_v = (50, 25) if is_tech_lab else (50, 50)
            
            # 資源檢查
            if player.minerals < req_m or player.vespene < req_v:
                self.locked_action = None; self.locked_target = None
                return actions.FUNCTIONS.no_op()

            # 搜尋正確的升級按鈕
            target_action = None
            action_names = ["Build_TechLab_Starport_quick", "Build_TechLab_quick"] if is_tech_lab else \
                           ["Build_Reactor_Starport_quick", "Build_Reactor_quick"]
            for name in action_names:
                func = self._get_safe_func(name)
                if func.id in available: target_action = func; break

            if target_action:
                self.locked_action = None; self.locked_target = None; self.lock_timer = 0
                return target_action("now")

            # 鎖定中等待按鈕出現
            if self.locked_action == 36 and self.locked_target is not None:
                return actions.FUNCTIONS.no_op()

            # 選取動作
            starport_centers = self._find_units_centers(unit_type, STARPORT_ID)
            if starport_centers:
                # 這裡建議過濾掉已有掛件的建築，邏輯同你目前的代碼
                self.locked_target = starport_centers[0]
                self.locked_action = 36; self.lock_timer = 1
                return actions.FUNCTIONS.select_point("select", self.locked_target)
            
            self.locked_action = None; return actions.FUNCTIONS.no_op()
        
        # [Action 37] 核融合核心升級 (奇數: 大和砲 / 偶數: 戰巡艦加速)
        elif action_id == 37:
            act_name = "Research_BattlecruiserWeaponRefit_quick" if self.active_parameter % 2 == 1 else "Research_BattlecruiserTacticalJump_quick"
            res_act = self._get_safe_func(act_name)
            if res_act and res_act.id in available and player.minerals >= 150 and player.vespene >= 150:
                return res_act("now")
            return self._select_unit(unit_type, FUSION_CORE_ID)

        # [Action 38] 電機工程所升級 (奇數: 步兵攻擊 / 偶數: 步兵防禦)
        elif action_id == 38:
            act_name = "Research_TerranInfantryWeapons_quick" if self.active_parameter % 2 == 1 else "Research_TerranInfantryArmor_quick"
            res_act = self._get_safe_func(act_name)
            # 注意：若以上名稱失敗，嘗試 Level1 版本
            if not res_act:
                act_name = "Research_TerranInfantryWeaponsLevel1_quick" if self.active_parameter % 2 == 1 else "Research_TerranInfantryArmorLevel1_quick"
                res_act = self._get_safe_func(act_name)
            
            if res_act and res_act.id in available and player.minerals >= 100 and player.vespene >= 100:
                return res_act("now")
            return self._select_unit(unit_type, ENGINEERING_BAY_ID)

        # [Action 39] 幽靈特務學院升級 (修正 KeyError)
        elif action_id == 39:
            # 修正名稱：隱形通常為 PersonalCloaking
            act_name = "Research_PersonalCloaking_quick" if self.active_parameter % 2 == 1 else "Research_GhostMoebiusReactor_quick"
            res_act = self._get_safe_func(act_name)
            if res_act and res_act.id in available:
                return res_act("now")
            return self._select_unit(unit_type, GHOST_ACADEMY_ID)
        
        # [Action 40]移動視角
        elif action_id == 40:
            p_idx = max(0, self.active_parameter - 1) % 64
            # D. 標準網格移動
            r, c = p_idx // 8, p_idx % 8
            target_pos = (np.clip(int((c + 0.5) * 8), 0, 63), 
                          np.clip(int((r + 0.5) * 8), 0, 63))
            return actions.FUNCTIONS.move_camera(target_pos)
        
       # [Action 41] 經濟重啟修正 (智慧打散工兵)
        elif action_id == 41:
            # 1. 沒空閒工兵，且沒選中任何工兵 -> 沒事做
            if player.idle_worker_count == 0 and not self._is_scv_selected(obs):
                self.locked_action = None
                return actions.FUNCTIONS.no_op()

            # 2. 如果已經成功選中工兵
            if self._is_scv_selected(obs):
                # ✨ 核心修正：利用現有函式找出「每一塊」礦脈的中心點，而不是全部平均
                mineral_centers = self._find_units_centers(unit_type, MINERAL_FIELD_ID)
                
                # A. 如果畫面上看得到礦
                if mineral_centers and actions.FUNCTIONS.Smart_screen.id in available:
                    # ✨ 隨機指派到其中一塊礦脈，避免所有工兵擠在同一顆礦上排隊
                    target = random.choice(mineral_centers)
                    self.locked_action = None 
                    self.lock_timer = 0
                    #print(f"⛏️ [DEBUG] 指派閒置工兵去採礦！目標: {target}")
                    return actions.FUNCTIONS.Smart_screen("now", target)
                
                # B. 畫面上沒礦 (可能剛蓋完房子人在外面) -> 移回基地
                else:
                    self.locked_action = 41 
                    if actions.FUNCTIONS.move_camera.id in available:
                        return actions.FUNCTIONS.move_camera(self._get_home_pos())
                    return actions.FUNCTIONS.no_op()
            
            # 3. 如果還沒選中工兵 -> 點擊左下角選取
            if actions.FUNCTIONS.select_idle_worker.id in available:
                self.locked_action = 41
                self.lock_timer = 1
                return actions.FUNCTIONS.select_idle_worker("select") 
                
            return actions.FUNCTIONS.no_op()
        
        # [Action 42] 派遣工兵採集瓦斯 (加入可用性檢查)
        elif action_id == 42:
            refinery_centers = self._find_units_centers(unit_type, REFINERY_ID)
            if not refinery_centers or self.gas_workers_assigned >= len(refinery_centers) * 3:
                self.locked_action = None; return actions.FUNCTIONS.no_op()

            if self.locked_action == 42 and self.lock_timer > 0 and self._is_scv_selected(obs):
                # ✨ 核心修正：加入 Smart_screen.id in available 判定
                if actions.FUNCTIONS.Smart_screen.id in available:
                    target = random.choice(refinery_centers)
                    self.gas_workers_assigned += 1 
                    self.locked_action = None; self.lock_timer = 0
                    return actions.FUNCTIONS.Smart_screen("now", target)
            
            self.locked_action = 42; self.lock_timer = 1
            return self._select_mineral_worker(obs, unit_type, available)
        
        # [Action 43] 全選並出動
        elif action_id == 43:
            # 1. 判斷敵方基地坐標 (基於你目前的 base_location_code)
            # 如果我方在左上 (0)，敵方就在右下；反之亦然
            enemy_base_minimap = (48, 48) if self.base_location_code == 0 else (16, 16)
            
            # 2. 執行階段：如果已經全選部隊且攻擊指令可用
            if self.locked_action == 43 and actions.FUNCTIONS.Attack_minimap.id in available:
                self.locked_action = None
                self.lock_timer = 0
                return actions.FUNCTIONS.Attack_minimap("now", enemy_base_minimap)

            # 3. 選取階段：執行全選軍事單位 (select_army)
            if actions.FUNCTIONS.select_army.id in available:
                self.locked_action = 43
                self.lock_timer = 1
                return actions.FUNCTIONS.select_army("select")
            
            return actions.FUNCTIONS.no_op()
        
        # [Action 44] 發呆
        elif action_id ==44:
            return actions.FUNCTIONS.no_op()
        
        # [Action 45] 主堡編隊 若以編隊移動視角
        elif action_id == 45:
            # 1. 檢查相關動作是否可用
            if actions.FUNCTIONS.select_control_group.id not in available:
                return actions.FUNCTIONS.no_op()

            # 2. 檢查熱鍵 1 是否已經有東西
            try:
                group_1_count = obs.observation.control_groups[1][1]
            except:
                group_1_count = 0

            # 3. 定義主堡 ID 並檢查目前選中狀態
            cc_ids = [COMMAND_CENTER_ID, ORBITAL_COMMAND_ID, PLANETARY_FORTRESS_ID]
            is_cc_selected = any(u.unit_type in cc_ids for u in obs.observation.multi_select) or \
                             (len(obs.observation.single_select) > 0 and obs.observation.single_select[0].unit_type in cc_ids)

            # --- 核心邏輯 ---
            if group_1_count == 0:
                # 【狀態 A: 還沒編隊】
                if is_cc_selected:
                    #print("🔘 [DEBUG] 偵測到選取，執行 Ctrl + 1 (Set Group 1)")
                    self.cc_is_bound = True
                    self.locked_action = None 
                    return actions.FUNCTIONS.select_control_group("set", 1) 
                else:
                    target_pos = None
                    for cid in cc_ids:
                        centers = self._find_units_centers(unit_type, cid)
                        if centers:
                            target_pos = centers[0]
                            break
                    if target_pos:
                        #print(f"📍 [DEBUG] 點擊主堡座標: {target_pos}")
                        self.locked_action = 45 
                        return actions.FUNCTIONS.select_point("select", target_pos) 
            else:
                # 【狀態 B: 已編隊成功 -> 模擬雙擊跳轉視角】
                self.cc_is_bound = True
                
                if not is_cc_selected:
                    # 第一次「點擊」: 叫出編組 (選取主堡)
                    #print("🔘 [DEBUG] 叫出主堡編隊 (Recall)")
                    self.locked_action = 45 # 保持鎖定，讓下一幀繼續執行視角跳轉
                    return actions.FUNCTIONS.select_control_group("recall", 1)
                else:
                    # 第二次「點擊」: 精確移動小地圖視角
                    #print("🎥 [DEBUG] 視角精確跳轉至主堡中心")
                    self.locked_action = None # 動作完成，解鎖
                    
                    if actions.FUNCTIONS.move_camera.id in available:
                        # ✨ 核心修正：讀取小地圖上「當前被選中單位」的發光像素點
                        selected_minimap = obs.observation.feature_minimap[features.MINIMAP_FEATURES.selected.index]
                        y_sel, x_sel = (selected_minimap == 1).nonzero()
                        
                        if x_sel.any():
                            # 計算被選取單位（主堡）在小地圖上的精準中心座標
                            target_minimap = (int(x_sel.mean()), int(y_sel.mean()))
                        else:
                            # 萬一抓不到（通常不會發生），才退回使用預設粗略座標
                            target_minimap = self._get_home_pos()
                            
                        return actions.FUNCTIONS.move_camera(target_minimap)

        # [Action 46] 時間軸戰術編隊 (Hotkey 2) - F2 真實版
        elif action_id == 46:
            step = self.lock_timer
            current_loop = int(obs.observation.game_loop[0])
            is_late_game = current_loop >= 10752 # 8分鐘分水嶺
            
            try:
                # 👉 修正：讀取編隊 2 的數量
                group_2_count = obs.observation.control_groups[2][1]
            except:
                group_2_count = 0

            # --- 第 0 步：掃描畫面並決定策略 ---
            if step == 0:
                # 【8 分鐘後：直接進入 F2 總攻模式】
                if is_late_game:
                    if actions.FUNCTIONS.select_army.id not in available:
                        self.locked_action = None
                        return actions.FUNCTIONS.no_op()
                    
                    self.locked_target = "F2_ALL"
                    self.locked_action = 46
                    
                # 【8 分鐘前：維持特種部隊模式】
                else:
                    marauder_centers = self._find_units_centers(unit_type, MARAUDER_ID)
                    total_marauders = len(marauder_centers)
                    
                    if total_marauders == 0:
                        self.locked_action = None
                        if group_2_count > 0 and actions.FUNCTIONS.select_control_group.id in available:
                            return actions.FUNCTIONS.select_control_group("recall", 2) # 👉 修正為 2
                        return actions.FUNCTIONS.no_op()
                        
                    if group_2_count >= 5:
                        self.locked_action = None
                        if actions.FUNCTIONS.select_control_group.id in available:
                            return actions.FUNCTIONS.select_control_group("recall", 2) # 👉 修正為 2
                        return actions.FUNCTIONS.no_op()
                        
                    target_count = min(5 - group_2_count, total_marauders)
                    self.locked_target = marauder_centers[:target_count]
                    self.locked_action = 46

            # --- 防呆機制 ---
            if getattr(self, 'locked_target', None) is None:
                self.locked_action = None
                self.lock_timer = 0
                return actions.FUNCTIONS.no_op()

            # ==========================================
            # 模式 A：總攻模式 (8分鐘後：使用 F2 全選)
            # ==========================================
            if self.locked_target == "F2_ALL":
                if step == 0:
                    return actions.FUNCTIONS.select_army("select")
                elif step == 1:
                    self.locked_action = None
                    self.lock_timer = 0
                    self.locked_target = None
                    if actions.FUNCTIONS.select_control_group.id in available:
                        print("🔥 [8分鐘後] F2 大軍集結！全圖戰鬥單位已覆蓋編入 Group 2！")
                        return actions.FUNCTIONS.select_control_group("set", 2) # 👉 修正為 2
                
                return actions.FUNCTIONS.no_op()

            # ==========================================
            # 模式 B：特種部隊模式 (8分鐘前：點擊缺額補齊)
            # ==========================================
            else:
                target_count = len(self.locked_target)
                
                if step < target_count:
                    target_pos = self.locked_target[step]
                    # 👉 修正：使用 group_2_count 判斷
                    if step == 0 and group_2_count == 0:
                        return actions.FUNCTIONS.select_point("select", target_pos)
                    else:
                        return actions.FUNCTIONS.select_point("toggle", target_pos)

                elif step == target_count:
                    self.locked_action = None
                    self.lock_timer = 0
                    self.locked_target = None
                    if actions.FUNCTIONS.select_control_group.id in available:
                        print(f"🎯 [8分鐘前] 乾淨補齊小隊！(已過濾工兵/建築) 目前 Group 2 總數: {group_2_count + target_count}")
                        return actions.FUNCTIONS.select_control_group("append", 2) # 👉 修正為 2

            return actions.FUNCTIONS.no_op()
        
        # [Action 47] 操控編隊2 A-move 到指定區域（穩定版）
        elif action_id == 47:
            step = self.lock_timer

            try:
                group_2_count = obs.observation.control_groups[2][1]
            except:
                group_2_count = 0

            if group_2_count == 0:
                self.locked_action = None
                self.lock_timer = 0
                self.locked_target = None
                return actions.FUNCTIONS.no_op()

            # 用 parameter 決定目標區域
            p_idx = max(0, self.active_parameter - 1) % 64
            r, c = p_idx // 8, p_idx % 8

            target_minimap = (
                np.clip(int((c + 0.5) * 8), 0, 63),
                np.clip(int((r + 0.5) * 8), 0, 63)
            )

            target_screen = (
                np.clip(int((c + 0.5) * 10.5), 0, 83),
                np.clip(int((r + 0.5) * 10.5), 0, 83)
            )

            if step == 0:
                self.locked_action = 47
                self.locked_target = {
                    "minimap": target_minimap,
                    "screen": target_screen
                }
                self.lock_timer = 0

                if actions.FUNCTIONS.select_control_group.id in available:
                    return actions.FUNCTIONS.select_control_group("recall", 2)
                return actions.FUNCTIONS.no_op()

            elif step == 1:
                if actions.FUNCTIONS.move_camera.id in available:
                    return actions.FUNCTIONS.move_camera(self.locked_target["minimap"])
                return actions.FUNCTIONS.no_op()

            elif step == 2:
                # 優先 A-move
                if actions.FUNCTIONS.Attack_screen.id in available:
                    self.locked_action = None
                    self.lock_timer = 0
                    self.locked_target = None
                    return actions.FUNCTIONS.Attack_screen("now", self.locked_target["screen"])

                # 備案：普通移動
                if actions.FUNCTIONS.Move_screen.id in available:
                    self.locked_action = None
                    self.lock_timer = 0
                    self.locked_target = None
                    return actions.FUNCTIONS.Move_screen("now", self.locked_target["screen"])

                self.locked_action = None
                self.lock_timer = 0
                self.locked_target = None
                return actions.FUNCTIONS.no_op()

            self.locked_action = None
            self.lock_timer = 0
            self.locked_target = None
            return actions.FUNCTIONS.no_op()
        
        # [Action 48] 編隊2 鎖定畫面中的敵方單位（集火穩定版）
        elif action_id == 48:
            step = self.lock_timer

            try:
                group_2_count = obs.observation.control_groups[2][1]
            except:
                group_2_count = 0

            if group_2_count == 0:
                self.locked_action = None
                self.lock_timer = 0
                self.locked_target = None
                return actions.FUNCTIONS.no_op()

            # 用 parameter 決定要看的區域
            p_idx = max(0, self.active_parameter - 1) % 64
            r, c = p_idx // 8, p_idx % 8

            target_minimap = (
                np.clip(int((c + 0.5) * 8), 0, 63),
                np.clip(int((r + 0.5) * 8), 0, 63)
            )

            fallback_screen = (
                np.clip(int((c + 0.5) * 10.5), 0, 83),
                np.clip(int((r + 0.5) * 10.5), 0, 83)
            )

            if step == 0:
                self.locked_action = 48
                self.locked_target = {
                    "minimap": target_minimap,
                    "fallback": fallback_screen
                }
                self.lock_timer = 0

                if actions.FUNCTIONS.select_control_group.id in available:
                    return actions.FUNCTIONS.select_control_group("recall", 2)
                return actions.FUNCTIONS.no_op()

            elif step == 1:
                if actions.FUNCTIONS.move_camera.id in available:
                    return actions.FUNCTIONS.move_camera(self.locked_target["minimap"])
                return actions.FUNCTIONS.no_op()

            elif step == 2:
                player_relative = obs.observation.feature_screen[
                    features.SCREEN_FEATURES.player_relative.index
                ]

                # 先找敵方戰鬥單位，再找工兵
                enemy_mask = (player_relative == 4)
                ey, ex = enemy_mask.nonzero()

                # 優先抓敵方工兵
                worker_mask = (player_relative == 4) & (
                    (unit_type == 45) | (unit_type == 84) | (unit_type == 104)
                )
                wy, wx = worker_mask.nonzero()

                # 若有敵方工兵，優先集火工兵
                if wx.any():
                    target = (int(wx[0]), int(wy[0]))
                    if actions.FUNCTIONS.Attack_screen.id in available:
                        self.locked_action = None
                        self.lock_timer = 0
                        self.locked_target = None
                        return actions.FUNCTIONS.Attack_screen("now", target)

                # 沒工兵就打最近敵軍
                if ex.any():
                    center = np.array([42, 42])
                    pts = np.column_stack((ex, ey))
                    dists = np.sum((pts - center) ** 2, axis=1)
                    idx = int(np.argmin(dists))
                    target = (int(pts[idx][0]), int(pts[idx][1]))

                    if actions.FUNCTIONS.Attack_screen.id in available:
                        self.locked_action = None
                        self.lock_timer = 0
                        self.locked_target = None
                        return actions.FUNCTIONS.Attack_screen("now", target)

                # 找不到敵人就 A 過去那格
                if actions.FUNCTIONS.Attack_screen.id in available:
                    self.locked_action = None
                    self.lock_timer = 0
                    self.locked_target = None
                    return actions.FUNCTIONS.Attack_screen("now", self.locked_target["fallback"])

                if actions.FUNCTIONS.Move_screen.id in available:
                    self.locked_action = None
                    self.lock_timer = 0
                    self.locked_target = None
                    return actions.FUNCTIONS.Move_screen("now", self.locked_target["fallback"])

                self.locked_action = None
                self.lock_timer = 0
                self.locked_target = None
                return actions.FUNCTIONS.no_op()

            self.locked_action = None
            self.lock_timer = 0
            self.locked_target = None
            return actions.FUNCTIONS.no_op()
        
        # [Action 49] 蓋2礦（工兵先走過去，再蓋；工兵編隊3）
        elif action_id == 49:
            step = self.lock_timer

            # 礦不夠
            if player.minerals < 400:
                self.locked_action = None
                self.lock_timer = 0
                self.locked_target = None
                return actions.FUNCTIONS.no_op()

            # 已有二礦就不做
            cc_count = 0
            for cid in [COMMAND_CENTER_ID, ORBITAL_COMMAND_ID, PLANETARY_FORTRESS_ID]:
                cc_count += len(self._find_units_centers(unit_type, cid))
            if cc_count >= 2:
                self.locked_action = None
                self.lock_timer = 0
                self.locked_target = None
                return actions.FUNCTIONS.no_op()

            # 寫死二礦位置
            if self.base_location_code == 1:   # 右下出生
                expand_screen = (32, 66)
                expand_minimap = (25, 50)
            else:                              # 左上出生
                expand_screen = (64, 37)
                expand_minimap = (48, 28)

            scv_selected = self._is_scv_selected(obs)

            # 找目前被選中的SCV位置（若有）
            scv_pos = None
            selected = obs.observation.feature_screen[features.SCREEN_FEATURES.selected.index]
            mask = (unit_type == SCV_ID) & (selected == 1)
            y_sel, x_sel = mask.nonzero()
            if x_sel.any():
                scv_pos = (int(x_sel.mean()), int(y_sel.mean()))

            # Step 0：選工兵
            if step == 0:
                self.locked_action = 49
                self.locked_target = {
                    "screen": expand_screen,
                    "minimap": expand_minimap
                }
                self.lock_timer = 0
                return self._select_scv_prioritized(obs, unit_type, available)

            # Step 1：工兵編隊3
            elif step == 1:
                if scv_selected and actions.FUNCTIONS.select_control_group.id in available:
                    return actions.FUNCTIONS.select_control_group("set", 3)

                return self._select_scv_prioritized(obs, unit_type, available)

            # Step 2：叫出編隊3
            elif step == 2:
                if actions.FUNCTIONS.select_control_group.id in available:
                    return actions.FUNCTIONS.select_control_group("recall", 3)
                return actions.FUNCTIONS.no_op()

            # Step 3：移動視角到二礦
            elif step == 3:
                if actions.FUNCTIONS.move_camera.id in available:
                    return actions.FUNCTIONS.move_camera(self.locked_target["minimap"])
                return actions.FUNCTIONS.no_op()

            # Step 4：工兵先走過去
            elif step == 4:
                if not scv_selected and actions.FUNCTIONS.select_control_group.id in available:
                    return actions.FUNCTIONS.select_control_group("recall", 3)

                if scv_selected and actions.FUNCTIONS.Move_screen.id in available:
                    return actions.FUNCTIONS.Move_screen("now", self.locked_target["screen"])

                return actions.FUNCTIONS.no_op()

            # Step 5：等靠近，靠近後就停一拍
            elif step == 5:
                target = self.locked_target["screen"]

                if not scv_selected and actions.FUNCTIONS.select_control_group.id in available:
                    self.lock_timer = 3
                    return actions.FUNCTIONS.select_control_group("recall", 3)

                if scv_pos is not None:
                    dist = np.sqrt((scv_pos[0] - target[0]) ** 2 + (scv_pos[1] - target[1]) ** 2)

                    if dist <= 8:
                        return actions.FUNCTIONS.no_op()

                    if actions.FUNCTIONS.Move_screen.id in available:
                        self.lock_timer = 4
                        return actions.FUNCTIONS.Move_screen("now", target)

                return actions.FUNCTIONS.no_op()

            # Step 6：開始蓋二礦
            elif step == 6:
                target = self.locked_target["screen"]

                if not scv_selected and actions.FUNCTIONS.select_control_group.id in available:
                    self.lock_timer = 2
                    return actions.FUNCTIONS.select_control_group("recall", 3)

                if actions.FUNCTIONS.Build_CommandCenter_screen.id in available:
                    self.locked_action = None
                    self.lock_timer = 0
                    self.locked_target = None
                    return actions.FUNCTIONS.Build_CommandCenter_screen("now", target)

                # 還沒出現建造按鈕，等一下
                return actions.FUNCTIONS.no_op()

            # 超過步驟還沒成功就解鎖，避免卡死
            self.locked_action = None
            self.lock_timer = 0
            self.locked_target = None
            return actions.FUNCTIONS.no_op()

        return actions.FUNCTIONS.no_op()
        
    # --- 內部輔助函式 ---
    def _get_home_pos(self):
        return (16, 16) if self.base_location_code == 0 else (48, 48)

    def _select_unit(self, unit_type, unit_id):
        y, x = (unit_type == unit_id).nonzero()
        if x.any():
            i = random.randint(0, len(x) - 1) # 增加安全性檢查
            target = (x[i], y[i])
            return actions.FUNCTIONS.select_point("select", target)
        
        # 找不到單位時，如果是鎖定狀態則強制解鎖
        self.locked_action = None
        return actions.FUNCTIONS.no_op()
    
    def _get_safe_func(self, name):
        try:
            if hasattr(actions.FUNCTIONS, name):
                return getattr(actions.FUNCTIONS, name)
        except (KeyError, AttributeError):
            pass
        return actions.FUNCTIONS.no_op

    # --- 修改後的選取工兵邏輯 ---

    def _select_scv_prioritized(self, obs, unit_type, available):
        """ 基礎選人：空閒 > 採礦 > 採瓦斯 """
        if actions.FUNCTIONS.select_idle_worker.id in available:
            return actions.FUNCTIONS.select_idle_worker("select")
            
        y, x = (unit_type == SCV_ID).nonzero()
        
        # --- 核心修正：防止畫面上沒有 SCV 時導致報錯 ---
        if not x.any():
            return actions.FUNCTIONS.move_camera(self._get_home_pos())
        
        # 優先抓礦工 (距離礦脈 12 像素內)
        m_y, m_x = (unit_type == MINERAL_FIELD_ID).nonzero()
        if m_x.any():
            for i in range(len(x)):
                dist = np.min(np.sqrt((m_x - x[i])**2 + (m_y - y[i])**2))
                if dist < 12: return actions.FUNCTIONS.select_point("select", (x[i], y[i]))
        
        # 沒礦工才隨機抓
        idx = random.randint(0, len(x) - 1)
        return actions.FUNCTIONS.select_point("select", (x[idx], y[idx]))
    
    def _select_mineral_worker(self, obs, unit_type, available):
        """ 專門尋找「遠離瓦斯」且「靠近礦脈」的工兵 """
        if actions.FUNCTIONS.select_idle_worker.id in available:
            return actions.FUNCTIONS.select_idle_worker("select")
        
        scv_y, scv_x = (unit_type == SCV_ID).nonzero()
        m_y, m_x = (unit_type == MINERAL_FIELD_ID).nonzero()
        r_y, r_x = (unit_type == REFINERY_ID).nonzero()
        
        

        candidates = []
        for i in range(len(scv_x)):
            # 條件 1: 靠近礦脈 (距離 < 10)
            is_near_min = False
            if m_x.any():
                dist_m = np.min(np.sqrt((m_x - scv_x[i])**2 + (m_y - scv_y[i])**2))
                if dist_m < 10: is_near_min = True
            
            # 條件 2: 必須「遠離」任何瓦斯廠 (距離 > 12)
            # 這是防止選到已經在瓦斯廠工作的工兵的關鍵
            is_not_gas_worker = True
            if r_x.any():
                dist_r = np.min(np.sqrt((r_x - scv_x[i])**2 + (r_y - scv_y[i])**2))
                if dist_r < 12: is_not_gas_worker = False
            
            if is_near_min and is_not_gas_worker:
                candidates.append((scv_x[i], scv_y[i]))

        if candidates:
            # 從礦工中隨機選一個
            target = random.choice(candidates)
            return actions.FUNCTIONS.select_point("select", target)
        
        # 如果真的沒找到純礦工，回傳 no_op 等待下一幀，不要亂抓
        return actions.FUNCTIONS.no_op()
    
    def _calc_barracks_pos(self, obs):
        """ 修正版：根據指揮中心位置動態計算兵營座標，確保右側空間 """
        global BASE_LOCATION_CODE  # 宣告使用全域變數
        
        player_relative = obs.observation.feature_minimap[features.MINIMAP_FEATURES.player_relative.index]
        y_mini, x_mini = (player_relative == 1).nonzero()
        
        # 計算平均座標
        bx = x_mini.mean() if x_mini.any() else 0
        by = y_mini.mean() if y_mini.any() else 0
        
        # 判斷位置
        is_on_right_side = bx > 32
        is_on_bottom_side = by > 32
        
        # --- 核心邏輯：如果是右下就變成 1 ---
        if is_on_right_side and is_on_bottom_side:
            BASE_LOCATION_CODE = 1
        else:
            BASE_LOCATION_CODE = 0
            
        # 原有的兵營座標計算邏輯
        if is_on_right_side:
            # 如果基地在右側，兵營要往左偏，留出右邊空間給科技實驗室
            target_x = self.cc_x_screen - 20
            target_y = self.cc_y_screen - 15
        else:
            # 如果基地在左側，兵營往右偏
            target_x = self.cc_x_screen + 20
            target_y = self.cc_y_screen - 15

        # 確保座標在安全範圍內 (0-83)
        return (np.clip(target_x, 10, 70), np.clip(target_y, 10, 70))
    
    def _find_geyser(self, unit_type):
        """ 局部像素遮罩：精確鎖定單一湧泉中心 """
        y, x = (unit_type == GEYSER_ID).nonzero()
        if x.any():
            ax, ay = x[0], y[0]
            mask = (np.abs(x - ax) < 10) & (np.abs(y - ay) < 10)
            return (int(x[mask].mean()), int(y[mask].mean()))
        return None

# =========================================================
# 🎮 主程式啟動器 (無限對局循環)
# =========================================================
# --- 修改 production_ai.py 的最後測試部分 ---
def main(argv):
    del argv
    agent = ProductionAI()
    with sc2_env.SC2Env(
        map_name="Simple96",
        players=[
    sc2_env.Agent(sc2_env.Race.terran), 
    sc2_env.Agent(sc2_env.Race.terran)
    #sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy) # 改為電腦 AI，難度最簡單
],
        agent_interface_format=sc2_env.AgentInterfaceFormat(
            feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64),
            use_raw_units=False),
        step_mul=4,
        realtime=False,
    ) as env:
        while True:
            print("--- 啟動新對局 ---")
            obs_list = env.reset()
            while True:
                action_id =  49#random.randint(1,2,11,18,34,41,42)#45 #1
                param = 1#1# # 網格限制 1-64
                
                sc2_action = agent.get_action(obs_list[0], action_id, parameter=param)
                
                # 同時傳入兩位玩家的指令
                obs_list = env.step([sc2_action, actions.FUNCTIONS.no_op()])
                
                if obs_list[0].last():
                    break

if __name__ == "__main__":
    app.run(main)