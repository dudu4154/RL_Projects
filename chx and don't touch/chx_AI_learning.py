import os
import random
import numpy as np
import csv
import time
from absl import app
from pysc2.env import sc2_env
from pysc2.lib import actions, features

# =========================================================
# 🏗️ 定義人族單位 ID
# =========================================================
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
            writer.writerow(["Time", "Minerals", "Vespene", "Workers", "Ideal", "Action_ID"])

    def log_step(self, time_val, minerals, vespene, workers, ideal, action_id):
        # 轉為 float 以避免 NumPy 類型在 round 時報錯
        display_time = float(time_val[0]) if hasattr(time_val, "__len__") else float(time_val)
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([round(display_time, 2), minerals, vespene, workers, ideal, action_id])

# =========================================================
# 🧠 生產大腦: 整合所有功能與修正
# =========================================================
class ProductionAI:
    def __init__(self):
        self.collector = DataCollector()
        self.depots_built = 0
        self.refinery_target = None
        self.cc_x_screen = 42
        self.cc_y_screen = 42
        self.gas_workers_assigned = 0
        
        # 初始化參數
        self.active_parameter = 1 
        self.base_location = 0 # 0: 左上, 1: 右下 (取代原本的全域變數)
        
        # 鏡頭管理座標
        self.base_minimap_coords = None 
        self.scan_points = []
        self.current_scan_idx = 0

    def _get_safe_func(self, name):
        try:
            return getattr(actions.FUNCTIONS, name)
        except KeyError:
            return None

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
        # 1. 優先處理參數更新
        if parameter is not None:
            self.active_parameter = parameter
        elif not hasattr(self, 'active_parameter'):
            self.active_parameter = 1 # 初始預設值
        
        # 2. 計算 4x4 建築網格座標 (用於畫面 84x84)
        b_id = self.active_parameter
        row, col = (b_id - 1) // 4, (b_id - 1) % 4
        jitter_range = 8  # 隨機偏移範圍
        offset_x = random.randint(-jitter_range, jitter_range)
        offset_y = random.randint(-jitter_range, jitter_range)

        grid_pos = (
            np.clip(int((col + 0.5) * 21) + offset_x, 0, 83),
            np.clip(int((row + 0.5) * 21) + offset_y, 0, 83)
        )

        unit_type = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
        player = obs.observation.player
        available = obs.observation.available_actions

        # --- 1. 座標與防禦型掃描點初始化 ---
        if self.base_minimap_coords is None:
            player_relative_mini = obs.observation.feature_minimap[features.MINIMAP_FEATURES.player_relative.index]
            y_mini, x_mini = (player_relative_mini == features.PlayerRelative.SELF).nonzero()
            
            if x_mini.any():
                bx, by = int(x_mini.mean()), int(y_mini.mean())
                self.base_minimap_coords = (bx, by)
                
                # 判斷基地位置
                if bx > 32 and by > 32:
                    self.base_location = 1 # 右下
                else:
                    self.base_location = 0 # 左上
                
                # 以基地為中心擴散的掃描點
                offsets = [(0, 0), (20, 0), (-20, 0), (0, 20), (0, -20), (15, 15), (-15, -15)]
                self.scan_points = [(np.clip(bx + dx, 0, 63), np.clip(by + dy, 0, 63)) for dx, dy in offsets]
        
        # --- 2. 視角更新與基地位置確認 ---
        cc_y, cc_x = (unit_type == COMMAND_CENTER_ID).nonzero()
        
        # 【修正】移除原本這裡衝突的 Action 9 視角切換邏輯
        
        # Action 0-7 (基礎營運): 若畫面沒基地，拉回主基地 (保護機制)
        if action_id <= 7 and not cc_x.any() and self.base_minimap_coords:
             # 只有當真的找不到任何指揮中心時才切換，避免頻繁跳動
            return actions.FUNCTIONS.move_camera(self.base_minimap_coords)

        # 更新基地在螢幕中的座標
        if cc_x.any():
            self.cc_x_screen, self.cc_y_screen = int(cc_x.mean()), int(cc_y.mean())

        # 動態工兵飽和計算
        current_workers = player.food_workers
        refinery_pixels = np.sum(unit_type == REFINERY_ID)
        refinery_count = int(refinery_pixels / 80)
        ideal_workers = 16 + (refinery_count * 3)
        self.collector.log_step(obs.observation.game_loop, player.minerals, 
                                player.vespene, current_workers, ideal_workers, action_id)

        # --- 3. 完整動作邏輯分支 ---

        # [Action 1] 建造補給站
        if action_id == 1:
            if player.minerals >= 100 and actions.FUNCTIONS.Build_SupplyDepot_screen.id in available:
                return actions.FUNCTIONS.Build_SupplyDepot_screen("now", grid_pos)
            return self._select_scv(unit_type, available)
        
        # [Action 2] 建造兵營 (使用智慧座標)
        elif action_id == 2:
            if player.minerals >= 150 and actions.FUNCTIONS.Build_Barracks_screen.id in available:
                # 【修正】改用 _calc_barracks_pos 計算比較好的位置
                smart_pos = self._calc_barracks_pos(obs)
                return actions.FUNCTIONS.Build_Barracks_screen("now", smart_pos)
            return self._select_scv(unit_type, available)
        
        # [Action 3] 建造軍工廠
        elif action_id == 3:
            if player.minerals >= 150 and player.vespene >= 100 and actions.FUNCTIONS.Build_Factory_screen.id in available:
                return actions.FUNCTIONS.Build_Factory_screen("now", grid_pos)
            return self._select_scv(unit_type, available)

        # [Action 4] 建造星際港
        elif action_id == 4:
            if player.minerals >= 150 and player.vespene >= 100 and actions.FUNCTIONS.Build_Starport_screen.id in available:
                return actions.FUNCTIONS.Build_Starport_screen("now", grid_pos)
            return self._select_scv(unit_type, available)

        # [Action 5] 建造核融合核心
        elif action_id == 5:
            if player.minerals >= 150 and player.vespene >= 150 and actions.FUNCTIONS.Build_FusionCore_screen.id in available:
                return actions.FUNCTIONS.Build_FusionCore_screen("now", grid_pos)
            return self._select_scv(unit_type, available)

        # [Action 6] 建造指揮中心 (二礦)
        elif action_id == 6:
            if player.minerals >= 400 and actions.FUNCTIONS.Build_CommandCenter_screen.id in available:
                return actions.FUNCTIONS.Build_CommandCenter_screen("now", grid_pos)
            return self._select_scv(unit_type, available)

        # [Action 7] 建造電機工程所
        elif action_id == 7:
            if player.minerals >= 125 and actions.FUNCTIONS.Build_EngineeringBay_screen.id in available:
                return actions.FUNCTIONS.Build_EngineeringBay_screen("now", grid_pos)
            return self._select_scv(unit_type, available)

        # [Action 8] 建造感應塔
        elif action_id == 8:
            if player.minerals >= 125 and player.vespene >= 50 and actions.FUNCTIONS.Build_SensorTower_screen.id in available:
                return actions.FUNCTIONS.Build_SensorTower_screen("now", grid_pos)
            return self._select_scv(unit_type, available)

        # [Action 9] 建造幽靈特務學院 (修正後的正確邏輯)
        elif action_id == 9:
            if player.minerals >= 150 and player.vespene >= 50 and actions.FUNCTIONS.Build_GhostAcademy_screen.id in available:
                return actions.FUNCTIONS.Build_GhostAcademy_screen("now", grid_pos)
            return self._select_scv(unit_type, available)

        # [Action 10] 建造兵工廠
        elif action_id == 10:
            if player.minerals >= 150 and player.vespene >= 100 and actions.FUNCTIONS.Build_Armory_screen.id in available:
                return actions.FUNCTIONS.Build_Armory_screen("now", grid_pos)
            return self._select_scv(unit_type, available)
        
        # [Action 11] 建造瓦斯廠
        elif action_id == 11:
            if player.minerals >= 75 and actions.FUNCTIONS.Build_Refinery_screen.id in available:
                self.refinery_target = self._find_geyser(unit_type)
                if self.refinery_target:
                    return actions.FUNCTIONS.Build_Refinery_screen("now", self.refinery_target)
            return self._select_scv(unit_type, available)
        
        # [Action 12] 建造飛彈砲台
        elif action_id == 12:
            if player.minerals >= 100 and actions.FUNCTIONS.Build_MissileTurret_screen.id in available:
                return actions.FUNCTIONS.Build_MissileTurret_screen("now", grid_pos)
            return self._select_scv(unit_type, available)

        # [Action 13] 建造碉堡
        elif action_id == 13:
            if player.minerals >= 100 and actions.FUNCTIONS.Build_Bunker_screen.id in available:
                return actions.FUNCTIONS.Build_Bunker_screen("now", grid_pos)
            return self._select_scv(unit_type, available)
        
        # --- [Action 14-32] 單位生產指令集 ---

        # [Action 14] 製造 SCV
        elif action_id == 14:
            if player.minerals >= 50 and actions.FUNCTIONS.Train_SCV_quick.id in available:
                return actions.FUNCTIONS.Train_SCV_quick("now")
            return self._select_unit(unit_type, COMMAND_CENTER_ID)

        # [Action 15] 製造礦騾 (修正版：同時捕捉 AttributeError 與 KeyError)
        elif action_id == 15:
            mule_action = None
            # 嘗試所有可能的 MULE 指令名稱
            for act_name in ["Effect_Call_Down_MULE_screen", "Call_OrbitalCommand_Mule_screen", "Effect_OrbitalCommand_Mule_screen"]:
                try:
                    mule_action = getattr(actions.FUNCTIONS, act_name)
                    if mule_action: break 
                except (AttributeError, KeyError): # ⬅️ 這裡多加一個 KeyError 捕捉
                    continue

            if mule_action and mule_action.id in available:
                y_m, x_m = (unit_type == MINERAL_FIELD_ID).nonzero()
                if x_m.any():
                    target = (int(x_m.mean()), int(y_m.mean()))
                    return mule_action("now", target)
            return self._select_unit(unit_type, ORBITAL_COMMAND_ID)

        # [Action 16] 製造陸戰隊
        elif action_id == 16:
            if player.minerals >= 50 and actions.FUNCTIONS.Train_Marine_quick.id in available:
                return actions.FUNCTIONS.Train_Marine_quick("now")
            return self._select_unit(unit_type, BARRACKS_ID)

        # [Action 17] 製造死神
        elif action_id == 17:
            if player.minerals >= 50 and player.vespene >= 50 and actions.FUNCTIONS.Train_Reaper_quick.id in available:
                return actions.FUNCTIONS.Train_Reaper_quick("now")
            return self._select_unit(unit_type, BARRACKS_ID)

        # [Action 18] 製造掠奪者
        elif action_id == 18:
            if actions.FUNCTIONS.Train_Marauder_quick.id in available:
                return actions.FUNCTIONS.Train_Marauder_quick("now")
            
            centers = self._find_units_centers(unit_type, BARRACKS_ID)
            if centers:
                return actions.FUNCTIONS.select_point("select", random.choice(centers))
            return actions.FUNCTIONS.no_op()
        
        # [Action 19] 製造幽靈特務
        elif action_id == 19:
            if player.minerals >= 150 and player.vespene >= 125 and actions.FUNCTIONS.Train_Ghost_quick.id in available:
                return actions.FUNCTIONS.Train_Ghost_quick("now")
            return self._select_unit(unit_type, BARRACKS_ID)

        # [Action 20] 製造惡狼
        elif action_id == 20:
            if player.minerals >= 100 and actions.FUNCTIONS.Train_Hellion_quick.id in available:
                return actions.FUNCTIONS.Train_Hellion_quick("now")
            return self._select_unit(unit_type, FACTORY_ID)

        # [Action 21] 製造戰狼
        elif action_id == 21:
            if player.minerals >= 100 and actions.FUNCTIONS.Train_Hellbat_quick.id in available:
                return actions.FUNCTIONS.Train_Hellbat_quick("now")
            return self._select_unit(unit_type, FACTORY_ID)

        # [Action 22] 製造寡婦詭雷
        elif action_id == 22:
            if player.minerals >= 75 and player.vespene >= 25 and actions.FUNCTIONS.Train_WidowMine_quick.id in available:
                return actions.FUNCTIONS.Train_WidowMine_quick("now")
            return self._select_unit(unit_type, FACTORY_ID)

        # [Action 23] 製造工程坦克
        elif action_id == 23:
            if player.minerals >= 150 and player.vespene >= 125 and actions.FUNCTIONS.Train_SiegeTank_quick.id in available:
                return actions.FUNCTIONS.Train_SiegeTank_quick("now")
            return self._select_unit(unit_type, FACTORY_ID)

        # [Action 24] 製造颶風飛彈車
        elif action_id == 24:
            if player.minerals >= 150 and player.vespene >= 100 and actions.FUNCTIONS.Train_Cyclone_quick.id in available:
                return actions.FUNCTIONS.Train_Cyclone_quick("now")
            return self._select_unit(unit_type, FACTORY_ID)

        # [Action 25] 製造雷神
        elif action_id == 25:
            if player.minerals >= 300 and player.vespene >= 200 and actions.FUNCTIONS.Train_Thor_quick.id in available:
                return actions.FUNCTIONS.Train_Thor_quick("now")
            return self._select_unit(unit_type, FACTORY_ID)

        # [Action 26] 製造維京戰機
        elif action_id == 26:
            if player.minerals >= 150 and player.vespene >= 75 and actions.FUNCTIONS.Train_VikingFighter_quick.id in available:
                return actions.FUNCTIONS.Train_VikingFighter_quick("now")
            return self._select_unit(unit_type, STARPORT_ID)

        # [Action 27] 製造醫療艇
        elif action_id == 27:
            if player.minerals >= 100 and player.vespene >= 100 and actions.FUNCTIONS.Train_Medivac_quick.id in available:
                return actions.FUNCTIONS.Train_Medivac_quick("now")
            return self._select_unit(unit_type, STARPORT_ID)

        # [Action 28] 製造解放者
        elif action_id == 28:
            if player.minerals >= 150 and player.vespene >= 150 and actions.FUNCTIONS.Train_Liberator_quick.id in available:
                return actions.FUNCTIONS.Train_Liberator_quick("now")
            return self._select_unit(unit_type, STARPORT_ID)

        # [Action 29] 製造渡鴉
        elif action_id == 29:
            if player.minerals >= 100 and player.vespene >= 200 and actions.FUNCTIONS.Train_Raven_quick.id in available:
                return actions.FUNCTIONS.Train_Raven_quick("now")
            return self._select_unit(unit_type, STARPORT_ID)

        # [Action 30] 製造戰巡艦
        elif action_id == 30:
            if player.minerals >= 400 and player.vespene >= 300 and actions.FUNCTIONS.Train_Battlecruiser_quick.id in available:
                return actions.FUNCTIONS.Train_Battlecruiser_quick("now")
            return self._select_unit(unit_type, STARPORT_ID)

        # [Action 31] 製造女妖轟炸機
        elif action_id == 31:
            if player.minerals >= 150 and player.vespene >= 100 and actions.FUNCTIONS.Train_Banshee_quick.id in available:
                return actions.FUNCTIONS.Train_Banshee_quick("now")
            return self._select_unit(unit_type, STARPORT_ID)

        # [Action 32] 升級為行星要塞
        elif action_id == 32:
            if player.minerals >= 150 and player.vespene >= 150 and actions.FUNCTIONS.Morph_PlanetaryFortress_quick.id in available:
                return actions.FUNCTIONS.Morph_PlanetaryFortress_quick("now")
            return self._select_unit(unit_type, COMMAND_CENTER_ID)
        
        # [Action 33] 補給站上升或下降
        elif action_id == 33:
            if actions.FUNCTIONS.Morph_SupplyDepot_Lower_quick.id in available:
                return actions.FUNCTIONS.Morph_SupplyDepot_Lower_quick("now")
            if actions.FUNCTIONS.Morph_SupplyDepot_Raise_quick.id in available:
                return actions.FUNCTIONS.Morph_SupplyDepot_Raise_quick("now")
            return self._select_unit(unit_type, SUPPLY_DEPOT_ID)

        # [Action 34] 兵營升級
        elif action_id == 34:
            is_tech_lab = (self.active_parameter % 2 == 1)
            
            if is_tech_lab:
                action = getattr(actions.FUNCTIONS, "Build_TechLab_quick", None)
                if not action: action = getattr(actions.FUNCTIONS, "Build_TechLab_Barracks_quick", None)
                req_m, req_v = 50, 25
            else:
                action = getattr(actions.FUNCTIONS, "Build_Reactor_quick", None)
                if not action: action = getattr(actions.FUNCTIONS, "Build_Reactor_Barracks_quick", None)
                req_m, req_v = 50, 50

            if action and action.id in available and player.minerals >= req_m and player.vespene >= req_v:
                return action("now")
            
            barracks_centers = self._find_units_centers(unit_type, BARRACKS_ID)
            if barracks_centers:
                return actions.FUNCTIONS.select_point("select", random.choice(barracks_centers))
            return actions.FUNCTIONS.no_op()

        # [Action 35] 軍工廠升級
        elif action_id == 35:
            if self.active_parameter % 2 == 1:
                if player.minerals >= 50 and player.vespene >= 25 and actions.FUNCTIONS.Build_TechLab_Factory_quick.id in available:
                    return actions.FUNCTIONS.Build_TechLab_Factory_quick("now")
            else:
                if player.minerals >= 50 and player.vespene >= 50 and actions.FUNCTIONS.Build_Reactor_Factory_quick.id in available:
                    return actions.FUNCTIONS.Build_Reactor_Factory_quick("now")
            return self._select_unit(unit_type, FACTORY_ID)

        # [Action 36] 星際港升級
        elif action_id == 36:
            if self.active_parameter % 2 == 1:
                if player.minerals >= 50 and player.vespene >= 25 and actions.FUNCTIONS.Build_TechLab_Starport_quick.id in available:
                    return actions.FUNCTIONS.Build_TechLab_Starport_quick("now")
            else:
                if player.minerals >= 50 and player.vespene >= 50 and actions.FUNCTIONS.Build_Reactor_Starport_quick.id in available:
                    return actions.FUNCTIONS.Build_Reactor_Starport_quick("now")
            return self._select_unit(unit_type, STARPORT_ID)

        # [Action 37] 核融合核心升級
        elif action_id == 37:
            act_name = "Research_BattlecruiserWeaponRefit_quick" if self.active_parameter % 2 == 1 else "Research_BattlecruiserTacticalJump_quick"
            res_act = self._get_safe_func(act_name)
            if res_act and res_act.id in available and player.minerals >= 150 and player.vespene >= 150:
                return res_act("now")
            return self._select_unit(unit_type, FUSION_CORE_ID)

        # [Action 38] 電機工程所升級
        elif action_id == 38:
            act_name = "Research_TerranInfantryWeapons_quick" if self.active_parameter % 2 == 1 else "Research_TerranInfantryArmor_quick"
            res_act = self._get_safe_func(act_name)
            if not res_act:
                act_name = "Research_TerranInfantryWeaponsLevel1_quick" if self.active_parameter % 2 == 1 else "Research_TerranInfantryArmorLevel1_quick"
                res_act = self._get_safe_func(act_name)
            
            if res_act and res_act.id in available and player.minerals >= 100 and player.vespene >= 100:
                return res_act("now")
            return self._select_unit(unit_type, ENGINEERING_BAY_ID)

        # [Action 39] 幽靈特務學院升級
        elif action_id == 39:
            act_name = "Research_PersonalCloaking_quick" if self.active_parameter % 2 == 1 else "Research_GhostMoebiusReactor_quick"
            res_act = self._get_safe_func(act_name)
            if res_act and res_act.id in available:
                return res_act("now")
            return self._select_unit(unit_type, GHOST_ACADEMY_ID)
        
        # [Action 40] 移動視角 (正確縮排)
        elif action_id == 40:
            block_id = self.active_parameter
            row = (block_id - 1) // 4
            col = (block_id - 1) % 4
            target_x = int((col + 0.5) * 21) # 修正網格寬度計算
            target_y = int((row + 0.5) * 21)
            final_pos = (np.clip(target_x, 0, 83), np.clip(target_y, 0, 83))
            return actions.FUNCTIONS.move_camera(final_pos)

        return actions.FUNCTIONS.no_op()

    # --- 內部輔助函式 ---
    def _select_unit(self, unit_type, unit_id):
        y, x = (unit_type == unit_id).nonzero()
        if x.any():
            return actions.FUNCTIONS.select_point("select", (int(x.mean()), int(y.mean())))
        return actions.FUNCTIONS.no_op()

    def _select_scv(self, unit_type, available):
        """ 優先選取空閒工兵，若無空閒則從畫面隨機選取 """
        if actions.FUNCTIONS.select_idle_worker.id in available:
            return actions.FUNCTIONS.select_idle_worker("select")
            
        y, x = (unit_type == SCV_ID).nonzero()
        if x.any():
            idx = random.randint(0, len(x) - 1)
            return actions.FUNCTIONS.select_point("select", (x[idx], y[idx]))
            
        return actions.FUNCTIONS.no_op()

    def _select_scv_filtered(self, unit_type, target, available):
        """ 選取遠離目標資源點的工兵，避免拉走正在採氣的人 """
        y, x = (unit_type == SCV_ID).nonzero()
        if x.any() and target:
            dist = np.sqrt((x - target[0])**2 + (y - target[1])**2)
            mask = dist > 15 
            if mask.any():
                idx = random.choice(np.where(mask)[0])
                return actions.FUNCTIONS.select_point("select", (x[idx], y[idx]))
        return self._select_scv(unit_type, available)

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
        # 使用 self.base_location 來判斷
        is_on_right_side = (self.base_location == 1)
        
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
        # 增加容錯：如果 GEYSER_ID 找不到，可以加入其他常見 ID
        y, x = (unit_type == GEYSER_ID).nonzero()
        if x.any():
            ax, ay = x[0], y[0]
            mask = (np.abs(x - ax) < 10) & (np.abs(y - ay) < 10)
            return (int(x[mask].mean()), int(y[mask].mean()))
        return None

# =========================================================
# 🎮 主程式啟動器 (無限對局循環)
# =========================================================
def main(argv):
    del argv
    agent = ProductionAI()
    try:
        with sc2_env.SC2Env(
            map_name="Simple96",
            players=[sc2_env.Agent(sc2_env.Race.terran), sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)],
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
                    # 隨機產生 1-40 的動作，全面測試所有功能
                    action_id = random.choice([1, 2, 11, 18, 34])#random.randint(1, 40)
                    param = random.randint(1, 16)
                    
                    sc2_action = agent.get_action(obs_list[0], action_id, parameter=param)
                    
                    obs_list = env.step([sc2_action])
                    
                    if obs_list[0].last():
                        break
    except KeyboardInterrupt:
        print("停止運行")

if __name__ == "__main__":
    app.run(main)