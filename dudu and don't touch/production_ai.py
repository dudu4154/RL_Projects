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
        
        # --- 【修正】在這裡初始化參數，避免 AttributeError ---
        self.active_parameter = 1 
        
        # 鏡頭管理座標
        self.base_minimap_coords = None 
        self.scan_points = []
        self.current_scan_idx = 0

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
        # 1. 優先處理參數更新，確保後面計算 grid_pos 不會出錯
        if parameter is not None:
            self.active_parameter = parameter
        
        # 2. 計算 4x4 建築網格座標 (用於畫面 84x84)
        b_id = self.active_parameter
        row, col = (b_id - 1) // 4, (b_id - 1) % 4
        jitter_range = 8  # 在 21 像素的範圍內，上下左右偏移 8 像素
        offset_x = random.randint(-jitter_range, jitter_range)
        offset_y = random.randint(-jitter_range, jitter_range)

        grid_pos = (
            np.clip(int((col + 0.5) * 21) + offset_x, 0, 83),
            np.clip(int((row + 0.5) * 21) + offset_y, 0, 83)
        )

        unit_type = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
        player = obs.observation.player
        available = obs.observation.available_actions

        # --- 更新通用參數暫存區 ---
        if parameter is not None:
            self.active_parameter = parameter
        elif not hasattr(self, 'active_parameter'):
            self.active_parameter = 1 # 初始預設值

        # --- 1. 座標與防禦型掃描點初始化 (在這裡加入判斷) ---
        if self.base_minimap_coords is None:
            global BASE_LOCATION_CODE  # 宣告使用全域變數
            
            player_relative_mini = obs.observation.feature_minimap[features.MINIMAP_FEATURES.player_relative.index]
            y_mini, x_mini = (player_relative_mini == features.PlayerRelative.SELF).nonzero()
            
            if x_mini.any():
                bx, by = int(x_mini.mean()), int(y_mini.mean())
                self.base_minimap_coords = (bx, by)
                
                # 【新增】在這裡直接判斷並寫入全域變數
                # bx > 32 (右側) 且 by > 32 (下側) = 右下角
                if bx > 32 and by > 32:
                    BASE_LOCATION_CODE = 1
                else:
                    BASE_LOCATION_CODE = 0
                
                # 以基地為中心擴散的掃描點
                offsets = [(0, 0), (20, 0), (-20, 0), (0, 20), (0, -20), (15, 15), (-15, -15)]
                self.scan_points = [(np.clip(bx + dx, 0, 63), np.clip(by + dy, 0, 63)) for dx, dy in offsets]
        # --- 2. 視角跳轉邏輯 (修正關鍵) ---
        cc_y, cc_x = (unit_type == COMMAND_CENTER_ID).nonzero()
        
        # Action 9 (開礦): 若畫面看得到主基，說明還沒跳轉到礦區
        if action_id == 9 and cc_x.any():
            return actions.FUNCTIONS.move_camera(self.scan_points[1]) # 跳轉到第一個擴散點

        # Action 0-7 (基礎營運): 若畫面沒基地，強制拉回主基地
        if action_id <= 7 and not cc_x.any() and self.base_minimap_coords:
            return actions.FUNCTIONS.move_camera(self.base_minimap_coords)

        # 更新基地在螢幕中的座標
        if cc_x.any():
            self.cc_x_screen, self.cc_y_screen = int(cc_x.mean()), int(cc_y.mean())

        # 動態工兵飽和計算
        current_workers = player.food_workers
        refinery_pixels = np.sum(unit_type == REFINERY_ID)
        refinery_count = int(refinery_pixels / 80) # 改用 80 像素作為門檻，解決識別錯誤
        ideal_workers = 16 + (refinery_count * 3)
        self.collector.log_step(obs.observation.game_loop, player.minerals, 
                                player.vespene, current_workers, ideal_workers, action_id)

        # --- 3. 完整動作邏輯分支 ---

        '''# [Action 1] 訓練 SCV (飽和度檢查)
        if action_id == 1:
            if current_workers < ideal_workers and player.minerals >= 50:
                if actions.FUNCTIONS.Train_SCV_quick.id in available:
                    return actions.FUNCTIONS.Train_SCV_quick("now")
            return self._select_unit(unit_type, COMMAND_CENTER_ID)

        # [Action 2] 建造補給站
        elif action_id == 2:
            if player.minerals >= 100 and actions.FUNCTIONS.Build_SupplyDepot_screen.id in available:
                return actions.FUNCTIONS.Build_SupplyDepot_screen("now", grid_pos)
            return self._select_scv(unit_type, available)

        # [Action 3] 建造瓦斯廠 (精確中心鎖定)
        elif action_id == 3:
            if player.minerals >= 75 and actions.FUNCTIONS.Build_Refinery_screen.id in available:
                self.refinery_target = self._find_geyser(unit_type)
                if self.refinery_target:
                    return actions.FUNCTIONS.Build_Refinery_screen("now", self.refinery_target)
            return self._select_scv(unit_type, available)

        # [Action 4] 指派採瓦斯 (上限 3 人/廠)
        elif action_id == 4:
            max_gas_allowed = refinery_count * 3
            if self.gas_workers_assigned < max_gas_allowed and self.refinery_target:
                if actions.FUNCTIONS.Harvest_Gather_screen.id in available:
                    self.gas_workers_assigned += 1
                    return actions.FUNCTIONS.Harvest_Gather_screen("now", self.refinery_target)
                return self._select_scv_filtered(unit_type, self.refinery_target)
            return actions.FUNCTIONS.no_op()

        # [Action 5] 建造兵營 (自動位移邏輯)
        elif action_id == 5:
            if player.minerals >= 150 and actions.FUNCTIONS.Build_Barracks_screen.id in available:
                return actions.FUNCTIONS.Build_Barracks_screen("now", grid_pos)
            return self._select_scv(unit_type, available)
        
        # [Action 6] 研發科技實驗室 (造掠奪者必備)
        elif action_id == 6:
            if player.minerals >= 50 and player.vespene >= 25:
                if actions.FUNCTIONS.Build_TechLab_quick.id in available:
                    return actions.FUNCTIONS.Build_TechLab_quick("now")
            return self._select_unit(unit_type, BARRACKS_ID)

        # [Action 7] 訓練掠奪者
        elif action_id == 7:
            barracks_list = self._find_units_centers(unit_type, BARRACKS_ID)
            if barracks_list:
                if actions.FUNCTIONS.Train_Marauder_quick.id in available:
                    return actions.FUNCTIONS.Train_Marauder_quick("now")
                # 點擊畫面上的第一個兵營
                return actions.FUNCTIONS.select_point("select", barracks_list[0])
            return actions.FUNCTIONS.no_op()

        # [Action 8] 中心擴散掃描 (偵察周邊)
        elif action_id == 8:
            target = self.scan_points[self.current_scan_idx]
            self.current_scan_idx = (self.current_scan_idx + 1) % len(self.scan_points)
            return actions.FUNCTIONS.move_camera(target)

        # [Action 9] 在視角選定網格建造二礦 (取代原先寫死的 42, 42)
        elif action_id == 9:
            if player.minerals >= 400 and actions.FUNCTIONS.Build_CommandCenter_screen.id in available:
                return actions.FUNCTIONS.Build_CommandCenter_screen("now", grid_pos)
            return self._select_scv(unit_type, available)'''
        # [Action 1.]建造補給站
        if action_id == 1:
            if player.minerals >= 100 and actions.FUNCTIONS.Build_SupplyDepot_screen.id in available:
                return actions.FUNCTIONS.Build_SupplyDepot_screen("now", grid_pos)
            return self._select_scv(unit_type, available)
        
        # [Action 2] 建造兵營 (自動位移邏輯)
        elif action_id == 2:
            if player.minerals >= 150 and actions.FUNCTIONS.Build_Barracks_screen.id in available:
                return actions.FUNCTIONS.Build_Barracks_screen("now", grid_pos)
            return self._select_scv(unit_type, available)
        
        elif action_id == 3:
            if player.minerals >= 150 and player.vespene >= 100 and actions.FUNCTIONS.Build_Factory_screen.id in available:
                return actions.FUNCTIONS.Build_Factory_screen("now", grid_pos)
            return self._select_scv(unit_type, available)

        # [Action 4] 建造星際港 (150 M, 100 V)
        elif action_id == 4:
            if player.minerals >= 150 and player.vespene >= 100 and actions.FUNCTIONS.Build_Starport_screen.id in available:
                return actions.FUNCTIONS.Build_Starport_screen("now", grid_pos)
            return self._select_scv(unit_type, available)

        # [Action 5] 建造核融合核心 (150 M, 150 V)
        elif action_id == 5:
            if player.minerals >= 150 and player.vespene >= 150 and actions.FUNCTIONS.Build_FusionCore_screen.id in available:
                return actions.FUNCTIONS.Build_FusionCore_screen("now", grid_pos)
            return self._select_scv(unit_type, available)

        # [Action 6] 建造指揮中心 (400 M)
        elif action_id == 6:
            if player.minerals >= 400 and actions.FUNCTIONS.Build_CommandCenter_screen.id in available:
                return actions.FUNCTIONS.Build_CommandCenter_screen("now", grid_pos)
            return self._select_scv(unit_type, available)

        # [Action 7] 建造電機工程所 (125 M)
        elif action_id == 7:
            if player.minerals >= 125 and actions.FUNCTIONS.Build_EngineeringBay_screen.id in available:
                return actions.FUNCTIONS.Build_EngineeringBay_screen("now", grid_pos)
            return self._select_scv(unit_type, available)

        # [Action 8] 建造感應塔 (125 M, 50 V)
        elif action_id == 8:
            if player.minerals >= 125 and player.vespene >= 50 and actions.FUNCTIONS.Build_SensorTower_screen.id in available:
                return actions.FUNCTIONS.Build_SensorTower_screen("now", grid_pos)
            return self._select_scv(unit_type, available)

        # [Action 9] 建造幽靈特務學院 (150 M, 50 V)
        elif action_id == 9:
            if player.minerals >= 150 and player.vespene >= 50 and actions.FUNCTIONS.Build_GhostAcademy_screen.id in available:
                return actions.FUNCTIONS.Build_GhostAcademy_screen("now", grid_pos)
            return self._select_scv(unit_type, available)

        # [Action 10] 建造兵工廠 (150 M, 100 V)
        elif action_id == 10:
            if player.minerals >= 150 and player.vespene >= 100 and actions.FUNCTIONS.Build_Armory_screen.id in available:
                return actions.FUNCTIONS.Build_Armory_screen("now", grid_pos)
            return self._select_scv(unit_type, available)
        
        # [Action 11] 建造瓦斯廠 (精確中心鎖定)
        elif action_id == 11:
            if player.minerals >= 75 and actions.FUNCTIONS.Build_Refinery_screen.id in available:
                self.refinery_target = self._find_geyser(unit_type)
                if self.refinery_target:
                    return actions.FUNCTIONS.Build_Refinery_screen("now", self.refinery_target)
            return self._select_scv(unit_type, available)
        
        # [Action 12] 建造飛彈砲台 (100 M)
        elif action_id == 12:
            if player.minerals >= 100 and actions.FUNCTIONS.Build_MissileTurret_screen.id in available:
                return actions.FUNCTIONS.Build_MissileTurret_screen("now", grid_pos)
            return self._select_scv(unit_type, available)

        # [Action 13] 建造碉堡 (100 M)
        elif action_id == 13:
            if player.minerals >= 100 and actions.FUNCTIONS.Build_Bunker_screen.id in available:
                return actions.FUNCTIONS.Build_Bunker_screen("now", grid_pos)
            return self._select_scv(unit_type, available)
        
        # --- [Action 14-32] 單位生產指令集 ---

        # [Action 14] 製造太空工程車 (SCV) - 50 M
        elif action_id == 14:
            if player.minerals >= 50 and actions.FUNCTIONS.Train_SCV_quick.id in available:
                return actions.FUNCTIONS.Train_SCV_quick("now")
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

        # [Action 18] 製造掠奪者 (Marauder) - 100 M, 25 V
        elif action_id == 18:
            if player.minerals >= 100 and player.vespene >= 25 and actions.FUNCTIONS.Train_Marauder_quick.id in available:
                return actions.FUNCTIONS.Train_Marauder_quick("now")
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
        elif action_id == 34:
            if self.active_parameter % 2 == 1: # 奇數分支
                if player.minerals >= 50 and player.vespene >= 25 and actions.FUNCTIONS.Build_TechLab_Barracks_quick.id in available:
                    return actions.FUNCTIONS.Build_TechLab_Barracks_quick("now")
            else: # 偶數分支
                if player.minerals >= 50 and player.vespene >= 50 and actions.FUNCTIONS.Build_Reactor_Barracks_quick.id in available:
                    return actions.FUNCTIONS.Build_Reactor_Barracks_quick("now")
            return self._select_unit(unit_type, BARRACKS_ID)

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
        elif action_id == 40:
        # 使用剛剛存入的 active_parameter (1-16)
            block_id = self.active_parameter
            
            # 4x4 網格計算邏輯
            row = (block_id - 1) // 4
            col = (block_id - 1) % 4
            target_x = int((col + 0.5) * 16)
            target_y = int((row + 0.5) * 16)
            
            final_pos = (np.clip(target_x, 0, 63), np.clip(target_y, 0, 63))
            # print(f"[Action 40] 視角切換至網格 {block_id}: {final_pos}")
            return actions.FUNCTIONS.move_camera(final_pos)

        return actions.FUNCTIONS.no_op()

    # --- 內部輔助函式 ---
    def _select_unit(self, unit_type, unit_id):
        y, x = (unit_type == unit_id).nonzero()
        if x.any():
            return actions.FUNCTIONS.select_point("select", (int(x.mean()), int(y.mean())))
        return actions.FUNCTIONS.no_op()

    # --- 修改後的選取工兵邏輯 ---
    def _select_scv(self, unit_type, available):
        """ 優先選取空閒工兵，若無空閒則從畫面隨機選取 """
        
        # 1. 優先判斷是否有空閒工兵 (select_idle_worker)
        if actions.FUNCTIONS.select_idle_worker.id in available:
            return actions.FUNCTIONS.select_idle_worker("select")
            
        # 2. 如果沒有空閒工兵，才執行原本的畫面隨機點擊邏輯
        y, x = (unit_type == SCV_ID).nonzero()
        if x.any():
            idx = random.randint(0, len(x) - 1)
            return actions.FUNCTIONS.select_point("select", (x[idx], y[idx]))
            
        return actions.FUNCTIONS.no_op()

    def _select_scv_filtered(self, unit_type, target, available): # 這裡要加 available
        """ 選取遠離目標資源點的工兵，避免拉走正在採氣的人 """
        y, x = (unit_type == SCV_ID).nonzero()
        if x.any() and target:
            dist = np.sqrt((x - target[0])**2 + (y - target[1])**2)
            mask = dist > 15 
            if mask.any():
                idx = random.choice(np.where(mask)[0])
                return actions.FUNCTIONS.select_point("select", (x[idx], y[idx]))
        return self._select_scv(unit_type, available) # 這裡原本沒傳參數會報錯

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
        map_name="Simple64",
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
                action_id = random.randint(1, 40)
                param = random.randint(1, 16) # 網格限制 1-16
                
                sc2_action = agent.get_action(obs_list[0], action_id, parameter=param)
                
                # 同時傳入兩位玩家的指令
                obs_list = env.step([sc2_action, actions.FUNCTIONS.no_op()])
                
                if obs_list[0].last():
                    break

if __name__ == "__main__":
    app.run(main)