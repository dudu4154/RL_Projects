import os
import random
import numpy as np
import csv
import time
from absl import app
from pysc2.env import sc2_env
from pysc2.lib import actions, features  # 刪掉最後面的 , units

# 定義人族單位 ID
COMMAND_CENTER_ID = 18
SUPPLY_DEPOT_ID = 19
REFINERY_ID = 20
BARRACKS_ID = 21
ENGINEERING_BAY_ID = 22  
BARRACKS_TECHLAB_ID = 37
SCV_ID = 45
MARAUDER_ID = 51
MINERAL_FIELD_ID = 341
GEYSER_ID = 342
BASE_LOCATION_CODE = 0
FACTORY_ID = 27
STARPORT_ID = 28
ARMORY_ID = 29
FUSION_CORE_ID = 30
GHOST_ACADEMY_ID = 26
ORBITAL_COMMAND_ID = 132
PLANETARY_FORTRESS_ID = 130

# =========================================================
# 📊 數據收集器: 紀錄資源與訓練狀態
# =========================================================
class DataCollector:
    def __init__(self):
        if not os.path.exists("logs"):
            os.makedirs("logs")
        self.filename = f"logs/terran_log_{int(time.time())}.csv"
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # 【新增】加入 Barracks 欄位
            writer.writerow(["Time", "Minerals", "Vespene", "Workers", "Ideal", "Barracks", "Action_ID"])

    def log_step(self, time_val, minerals, vespene, workers, ideal, barracks, action_id):
        # 【更新】接收並寫入 barracks 參數
        display_time = float(time_val[0]) if hasattr(time_val, "__len__") else float(time_val)
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([round(display_time, 2), minerals, vespene, workers, ideal, barracks, action_id])

# =========================================================
# 🧠 生產大腦: 整合所有功能與修正
# =========================================================
class ProductionAI:
    def _is_scv_selected(self, obs):
        if len(obs.observation.single_select) > 0:
            return obs.observation.single_select[0].unit_type == SCV_ID
        if len(obs.observation.multi_select) > 0:
            return any(u.unit_type == SCV_ID for u in obs.observation.multi_select)
        return False
    # --- 新增安全獲取函式 ---
    def _get_safe_func(self, name):
        try:
            return getattr(actions.FUNCTIONS, name)
        except KeyError:
            return None
        
    def __init__(self):
        self.collector = DataCollector()
        self.depots_built = 0
        self.refinery_target = None
        self.cc_x_screen = 42
        self.cc_y_screen = 42
        self.gas_workers_assigned = 0
        self.active_parameter = 1 
        self.base_minimap_coords = None
        self.locked_action = None
        self.lock_timer = 0  # 【新增】鎖定計時器

    def _find_units_centers(self, unit_type, unit_id):
        """ 尋找畫面上所有指定 ID 的建築中心點，避免點擊到空地 """
        y, x = (unit_type == unit_id).nonzero()
        if not x.any(): return []
        
        centers = []
        # 簡單的聚類技巧：找第一個點及其周圍像素
        temp_x, temp_y = list(x), list(y)
        while temp_x:
            bx, by = temp_x[0], temp_y[0]
            mask = (np.abs(np.array(temp_x) - bx) < 12) & (np.abs(np.array(temp_y) - by) < 12)
            centers.append((int(np.mean(np.array(temp_x)[mask])), int(np.mean(np.array(temp_y)[mask]))))
            temp_x = [px for i, px in enumerate(temp_x) if not mask[i]]
            temp_y = [py for i, py in enumerate(temp_y) if not mask[i]]
        return centers

    def get_action(self, obs, action_id, parameter=None):
        # --- A. 處理鎖定與超時 (確保連鎖不被中斷) ---
        if self.locked_action is not None:
            self.lock_timer += 1
            if self.lock_timer > 60:
                self.locked_action = None
                self.lock_timer = 0
            else:
                action_id = self.locked_action
        if parameter is not None: self.active_parameter = parameter
        
        unit_type = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
        player = obs.observation.player
        available = obs.observation.available_actions
        
        # 定義建築網格
        b_id = self.active_parameter
        # 將網格間距從 21 縮小或位移，避免點擊中心 (42, 42)
        row, col = (b_id - 1) // 4, (b_id - 1) % 4
        # 修正：讓 x 和 y 避開中央區域 (25~60 像素區間)
        tx = (col * 20) + 12
        ty = (row * 20) + 12
        if 25 < tx < 60: tx = 15 if tx < 42 else 70 # 強制推離中心
        if 25 < ty < 60: ty = 15 if ty < 42 else 70
        grid_pos = (np.clip(tx, 0, 83), np.clip(ty, 0, 83))

       
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

            # B. 【修正點】如果已經選中工兵，就不要再重複執行 _select_scv
            # 這樣可以讓遊戲引擎有時間把建築選單顯示出來
            if self._is_scv_selected(obs):
                self.locked_action = 1
                return actions.FUNCTIONS.no_op() # 靜止等待按鈕加載

            # C. 還沒選到人
            self.locked_action = 1 
            return self._select_scv(unit_type, available)
         
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
            return self._select_scv(unit_type, available)
        
        elif action_id == 3:
            if player.minerals < 150 or player.vespene < 100:
                self.locked_action = None
                return actions.FUNCTIONS.no_op()
            if actions.FUNCTIONS.Build_Factory_screen.id in available:
                self.locked_action = None 
                self.lock_timer = 0
                return actions.FUNCTIONS.Build_Factory_screen("now", grid_pos)
            if self._is_scv_selected(obs):
                self.locked_action = 3
                return actions.FUNCTIONS.no_op()
            self.locked_action = 3 # 【關鍵】
            return self._select_scv(unit_type, available)

        # [Action 4] 建造星際港 (150 M, 100 V)
        elif action_id == 4:
            if player.minerals < 150 or player.vespene < 100 :
                self.locked_action = None
                return actions.FUNCTIONS.no_op()
            if actions.FUNCTIONS.Build_Starport_screen.id in available:
                self.locked_action = None 
                self.lock_timer = 0
                return actions.FUNCTIONS.Build_Starport_screen("now", grid_pos)
            if self._is_scv_selected(obs):
                self.locked_action = 4
                return actions.FUNCTIONS.no_op()
            self.locked_action = 4 # 【關鍵】
            return self._select_scv(unit_type, available)

        # [Action 5] 建造核融合核心 (150 M, 150 V)
        elif action_id == 5:
            if player.minerals < 150 or player.vespene < 150 :
                self.locked_action = None
                return actions.FUNCTIONS.no_op()
            if actions.FUNCTIONS.Build_FusionCore_screen.id in available:
                self.locked_action = None 
                self.lock_timer = 0
                return actions.FUNCTIONS.Build_FusionCore_screen("now", grid_pos)
            if self._is_scv_selected(obs):
                self.locked_action = 5
                return actions.FUNCTIONS.no_op()
            self.locked_action = 5
            return self._select_scv(unit_type, available)

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
            return self._select_scv(unit_type, available)

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
            return self._select_scv(unit_type, available)

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
            return self._select_scv(unit_type, available)

        # [Action 9] 建造幽靈特務學院 (150 M, 50 V)
        elif action_id == 9:
            if player.minerals < 150 or player.vespene < 50:
                self.locked_action = None
                return actions.FUNCTIONS.no_op()
            if actions.FUNCTIONS.Build_GhostAcademy_screen.id in available:
                self.locked_action = None 
                self.lock_timer = 0
                return actions.FUNCTIONS.Build_GhostAcademy_screen("now", grid_pos)
            if self._is_scv_selected(obs):
                self.locked_action = 9
                return actions.FUNCTIONS.no_op()
            self.locked_action = 9
            return self._select_scv(unit_type, available)

        # [Action 10] 建造兵工廠 (150 M, 100 V)
        elif action_id == 10:
            if player.minerals < 150 or player.vespene < 100:
                self.locked_action = None
                return actions.FUNCTIONS.no_op()
            if actions.FUNCTIONS.Build_Armory_screen.id in available:
                self.locked_action = None 
                self.lock_timer = 0
                return actions.FUNCTIONS.Build_Armory_screen("now", grid_pos)
            if self._is_scv_selected(obs):
                self.locked_action = 10
                return actions.FUNCTIONS.no_op()
            self.locked_action = 10
            return self._select_scv(unit_type, available)
        
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
            return self._select_scv(unit_type, available)
        
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
            return self._select_scv(unit_type, available)

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
            return self._select_scv(unit_type, available)
        
        # --- [Action 14-32] 單位生產指令集 ---

        # [Action 14] 製造 SCV (注意：此行現在應該在刪除死碼後的第 150 行左右)
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
            if player.minerals >= 50 and actions.FUNCTIONS.Train_Marine_quick.id in available:
                return actions.FUNCTIONS.Train_Marine_quick("now")
            return self._select_unit(unit_type, BARRACKS_ID)

        # [Action 17] 製造死神 (Reaper) - 50 M, 50 V
        elif action_id == 17:
            if player.minerals >= 50 and player.vespene >= 50 and actions.FUNCTIONS.Train_Reaper_quick.id in available:
                return actions.FUNCTIONS.Train_Reaper_quick("now")
            return self._select_unit(unit_type, BARRACKS_ID)

        
        # [Action 18] 製造掠奪者 (優化版)
        elif action_id == 18:
            if actions.FUNCTIONS.Train_Marauder_quick.id in available:
                return actions.FUNCTIONS.Train_Marauder_quick("now")
            
            # 如果已經選中兵營但沒按鈕，代表科技不足，回傳 no_op 靜止等待
            if any(u.unit_type == BARRACKS_ID for u in obs.observation.single_select):
                return actions.FUNCTIONS.no_op()
                
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

        # [Action 23] 製造工程坦克 (Siege Tank) - 150 M, 125 V
        elif action_id == 23:
            if player.minerals >= 150 and player.vespene >= 125 and actions.FUNCTIONS.Train_SiegeTank_quick.id in available:
                return actions.FUNCTIONS.Train_SiegeTank_quick("now")
            return self._select_unit(unit_type, FACTORY_ID)

        # [Action 24] 製造颶風飛彈車 (Cyclone) - 150 M, 100 V
        elif action_id == 24:
            if player.minerals >= 150 and player.vespene >= 100 and actions.FUNCTIONS.Train_Cyclone_quick.id in available:
                return actions.FUNCTIONS.Train_Cyclone_quick("now")
            return self._select_unit(unit_type, FACTORY_ID)

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

        # [Action 29] 製造渡鴉 (Raven) - 100 M, 200 V
        elif action_id == 29:
            if player.minerals >= 100 and player.vespene >= 200 and actions.FUNCTIONS.Train_Raven_quick.id in available:
                return actions.FUNCTIONS.Train_Raven_quick("now")
            return self._select_unit(unit_type, STARPORT_ID)

        # [Action 30] 製造戰巡艦 (Battlecruiser) - 400 M, 300 V
        elif action_id == 30:
            if player.minerals >= 400 and player.vespene >= 300 and actions.FUNCTIONS.Train_Battlecruiser_quick.id in available:
                return actions.FUNCTIONS.Train_Battlecruiser_quick("now")
            return self._select_unit(unit_type, STARPORT_ID)

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

        # [Action 34] 兵營升級 (奇數: 科技實驗室 / 偶數: 反應爐)
        
# [Action 34] 兵營升級 (優化穩定版)
        elif action_id == 34:
            # 1. 決定要升級哪一種
            is_tech_lab = (self.active_parameter % 2 == 1)
            
            # 2. 獲取動作 (優先嘗試通用型，再嘗試專用型)
            if is_tech_lab:
                action = getattr(actions.FUNCTIONS, "Build_TechLab_quick", None)
                if not action: action = getattr(actions.FUNCTIONS, "Build_TechLab_Barracks_quick", None)
                req_m, req_v = 50, 25
            else:
                action = getattr(actions.FUNCTIONS, "Build_Reactor_quick", None)
                if not action: action = getattr(actions.FUNCTIONS, "Build_Reactor_Barracks_quick", None)
                req_m, req_v = 50, 50

            # 3. 執行升級 (若動作可用且資源足夠)
            if action and action.id in available and player.minerals >= req_m and player.vespene >= req_v:
                return action("now")
            
            # 4. 若無法執行，則精確選取兵營 (避免點擊到多個兵營的平均中心空地)
            barracks_centers = self._find_units_centers(unit_type, BARRACKS_ID)
            if barracks_centers:
                # 隨機選一個兵營，增加 AI 嘗試不同建築的機會
                return actions.FUNCTIONS.select_point("select", random.choice(barracks_centers))
            return actions.FUNCTIONS.no_op()

        # [Action 35] 軍工廠升級 (奇數: 科技實驗室 / 偶數: 反應爐)
        elif action_id == 35:
            if self.active_parameter % 2 == 1:
                if player.minerals >= 50 and player.vespene >= 25 and actions.FUNCTIONS.Build_TechLab_Factory_quick.id in available:
                    return actions.FUNCTIONS.Build_TechLab_Factory_quick("now")
            else:
                if player.minerals >= 50 and player.vespene >= 50 and actions.FUNCTIONS.Build_Reactor_Factory_quick.id in available:
                    return actions.FUNCTIONS.Build_Reactor_Factory_quick("now")
            return self._select_unit(unit_type, FACTORY_ID)

        # [Action 36] 星際港升級 (奇數: 科技實驗室 / 偶數: 反應爐)
        elif action_id == 36:
            if self.active_parameter % 2 == 1:
                if player.minerals >= 50 and player.vespene >= 25 and actions.FUNCTIONS.Build_TechLab_Starport_quick.id in available:
                    return actions.FUNCTIONS.Build_TechLab_Starport_quick("now")
            else:
                if player.minerals >= 50 and player.vespene >= 50 and actions.FUNCTIONS.Build_Reactor_Starport_quick.id in available:
                    return actions.FUNCTIONS.Build_Reactor_Starport_quick("now")
            return self._select_unit(unit_type, STARPORT_ID)

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
        # [Action 40] 智慧移動視角 (整合編隊跳轉邏輯)
        elif action_id == 40:
            block_id = self.active_parameter
            
            # A. 判斷目標網格是否為基地位置 (0=左上 1, 1=右下 16)
            is_base_grid = (block_id == 1 and BASE_LOCATION_CODE == 0) or \
                           (block_id == 16 and BASE_LOCATION_CODE == 1)
            
            # B. 檢查編隊 1 是否已經設定為主堡
            control_groups = obs.observation.control_groups
            has_cc_in_group1 = (control_groups[1][0] == COMMAND_CENTER_ID)

            # C. 智慧判定：能用編隊跳轉就用，不能就用相機移動
            if is_base_grid and has_cc_in_group1 and (actions.FUNCTIONS.select_control_group.id in available):
                # print(f"🚀 透過編隊 1 快捷鍵跳轉回基地 (網格 {block_id})")
                return actions.FUNCTIONS.select_control_group("recall", 1)

            # D. 標準網格移動
            r, c = (block_id - 1) // 4, (block_id - 1) % 4
            target_pos = (np.clip(int((c + 0.5) * 16), 0, 63), 
                          np.clip(int((r + 0.5) * 16), 0, 63))
            return actions.FUNCTIONS.move_camera(target_pos)
        
        # [Action 41] 經濟重啟 (優化版：增加空閒檢查與自動回家)
        # [Action 41] 經濟重啟修正
        elif action_id == 41:
            # 只有在真的有閒置工兵時才啟動連鎖動作
            if player.idle_worker_count == 0 and not self._is_scv_selected(obs):
                self.locked_action = None
                return actions.FUNCTIONS.no_op()

            # 剩下的採礦邏輯...

            if self._is_scv_selected(obs):
                y_m, x_m = (unit_type == MINERAL_FIELD_ID).nonzero()
                if x_m.any():
                    target = (int(x_m.mean()), int(y_m.mean()))
                    self.locked_action = None # 成功下令採礦，解鎖
                    return actions.FUNCTIONS.Smart_screen("now", target)
                else:
                    self.locked_action = None # 選了人但畫面沒礦，強制解鎖讓它執行 Action 40 回家
                    return actions.FUNCTIONS.move_camera((16, 16)) 
            
            if actions.FUNCTIONS.select_idle_worker.id in available:
                self.locked_action = 41 # 鎖定，直到下一幀執行採礦
                return actions.FUNCTIONS.select_idle_worker("select_all")
            
            return actions.FUNCTIONS.no_op()
        

    # --- 內部輔助函式 ---
    def _select_unit(self, unit_type, unit_id):
        # 使用歸屬過濾 (Relative == 1) 避免點到敵人的東西
        y, x = (unit_type == unit_id).nonzero()
        if x.any():
            # 取中心點點擊
            return actions.FUNCTIONS.select_point("select", (int(x.mean()), int(y.mean())))
        return actions.FUNCTIONS.no_op()

    # --- 修改後的選取工兵邏輯 ---

    def _select_scv(self, unit_type, available):
        # 1. 優先選取閒置工兵 (最穩定)
        if actions.FUNCTIONS.select_idle_worker.id in available:
            return actions.FUNCTIONS.select_idle_worker("select")
        
        # 2. 如果螢幕上有看到工兵，隨機點一個
        # 將 select_all_type 改為單點選取，確保建築面板會出現
        y, x = (unit_type == SCV_ID).nonzero()
        if x.any():
            idx = random.randint(0, len(x) - 1)
            # 使用 "select" 參數選取單一工兵
            return actions.FUNCTIONS.select_point("select", (x[idx], y[idx]))
            
        # 3. 【修正】如果都沒看到，去點擊礦堆 (Mineral Field) 周邊
        # 礦堆一定在基地旁邊，那裡一定有工兵在採礦
        y_m, x_m = (unit_type == MINERAL_FIELD_ID).nonzero()
        if x_m.any():
            # 隨機點擊礦區的一個點，極高機率選中正在採礦的工兵
            idx = random.randint(0, len(x_m) - 1)
            return actions.FUNCTIONS.select_point("select", (x_m[idx], y_m[idx]))

        return actions.FUNCTIONS.no_op()
    
    def _calc_depot_pos(self):
        """ 三角形排列座標計算 """
        if self.depots_built == 0:
            target = (self.cc_x_screen + 15, self.cc_y_screen + 15)
        elif self.depots_built == 1:
            target = (self.cc_x_screen + 27, self.cc_y_screen + 15)
        else:
            target = (self.cc_x_screen + 21, self.cc_y_screen + 27)
        self.depots_built = (self.depots_built + 1) % 3
        return (np.clip(target[0], 0, 83), np.clip(target[1], 0, 83))

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
        players=[sc2_env.Agent(sc2_env.Race.terran), sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=sc2_env.AgentInterfaceFormat(
            feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64),
            use_raw_units=False),
        step_mul=16,
        realtime=False,
    ) as env:
        while True:
            print("--- 啟動新對局 ---")
            obs_list = env.reset()
            while True:
                action_id = 1#random.randint(1, 41)##random.choice([41,42])#
                param = random.randint(1, 16)#1# # 網格限制 1-16
                
                sc2_action = agent.get_action(obs_list[0], action_id, parameter=param)
                
                # 同時傳入兩位玩家的指令
                obs_list = env.step([sc2_action, actions.FUNCTIONS.no_op()])
                
                if obs_list[0].last():
                    break

if __name__ == "__main__":
    app.run(main)