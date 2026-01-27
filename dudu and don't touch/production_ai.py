import os
import random
import numpy as np
import csv
import time
from absl import app
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

# 定義人族單位 ID
COMMAND_CENTER_ID = 18
SUPPLY_DEPOT_ID = 19
REFINERY_ID = 20
BARRACKS_ID = 21
BARRACKS_TECHLAB_ID = 37
SCV_ID = 45
MARAUDER_ID = 51
MINERAL_FIELD_ID = 341
GEYSER_ID = 342
BASE_LOCATION_CODE = 0

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
        display_time = float(time_val) 
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
        
        # 鏡頭管理座標
        self.base_minimap_coords = None 
        self.scan_points = []
        self.current_scan_idx = 0

    def get_action(self, obs, action_id):
        """ 
        0:無動作, 1:造SCV, 2:蓋補給站, 3:蓋瓦斯廠, 4:採瓦斯, 
        5:蓋兵營, 6:研發科技, 7:造掠奪者, 8:擴散掃描, 9:擴張開礦
        """
        unit_type = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
        player = obs.observation.player
        available = obs.observation.available_actions

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

        # [Action 1] 訓練 SCV (飽和度檢查)
        if action_id == 1:
            if current_workers < ideal_workers and player.minerals >= 50:
                if actions.FUNCTIONS.Train_SCV_quick.id in available:
                    return actions.FUNCTIONS.Train_SCV_quick("now")
            return self._select_unit(unit_type, COMMAND_CENTER_ID)

        # [Action 2] 建造補給站 (三角形排列邏輯)
        elif action_id == 2:
            if player.minerals >= 100 and actions.FUNCTIONS.Build_SupplyDepot_screen.id in available:
                target = self._calc_depot_pos()
                return actions.FUNCTIONS.Build_SupplyDepot_screen("now", target)
            return self._select_scv(unit_type)

        # [Action 3] 建造瓦斯廠 (精確中心鎖定)
        elif action_id == 3:
            if player.minerals >= 75 and actions.FUNCTIONS.Build_Refinery_screen.id in available:
                self.refinery_target = self._find_geyser(unit_type)
                if self.refinery_target:
                    return actions.FUNCTIONS.Build_Refinery_screen("now", self.refinery_target)
            return self._select_scv(unit_type)

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
                target = self._calc_barracks_pos(obs)
                return actions.FUNCTIONS.Build_Barracks_screen("now", target)
            return self._select_scv(unit_type)

        # [Action 6] 研發科技實驗室 (造掠奪者必備)
        elif action_id == 6:
            if player.minerals >= 50 and player.vespene >= 25:
                if actions.FUNCTIONS.Build_TechLab_quick.id in available:
                    return actions.FUNCTIONS.Build_TechLab_quick("now")
            return self._select_unit(unit_type, BARRACKS_ID)

        # [Action 7] 訓練掠奪者
        elif action_id == 7:
            if player.minerals >= 100 and player.vespene >= 25:
                if actions.FUNCTIONS.Train_Marauder_quick.id in available:
                    return actions.FUNCTIONS.Train_Marauder_quick("now")
            return self._select_unit(unit_type, BARRACKS_ID)

        # [Action 8] 中心擴散掃描 (偵察周邊)
        elif action_id == 8:
            target = self.scan_points[self.current_scan_idx]
            self.current_scan_idx = (self.current_scan_idx + 1) % len(self.scan_points)
            return actions.FUNCTIONS.move_camera(target)

        # [Action 9] 在視角中心建造二礦
        elif action_id == 9:
            if player.minerals >= 400 and actions.FUNCTIONS.Build_CommandCenter_screen.id in available:
                return actions.FUNCTIONS.Build_CommandCenter_screen("now", (42, 42))
            return self._select_scv(unit_type)

        return actions.FUNCTIONS.no_op()

    # --- 內部輔助函式 ---
    def _select_unit(self, unit_type, unit_id):
        y, x = (unit_type == unit_id).nonzero()
        if x.any():
            return actions.FUNCTIONS.select_point("select", (int(x.mean()), int(y.mean())))
        return actions.FUNCTIONS.no_op()

    def _select_scv(self, unit_type):
        y, x = (unit_type == SCV_ID).nonzero()
        if x.any():
            idx = random.randint(0, len(x) - 1)
            return actions.FUNCTIONS.select_point("select", (x[idx], y[idx]))
        return actions.FUNCTIONS.no_op()

    def _select_scv_filtered(self, unit_type, target):
        """ 選取遠離目標資源點的工兵，避免拉走正在採氣的人 """
        y, x = (unit_type == SCV_ID).nonzero()
        if x.any() and target:
            dist = np.sqrt((x - target[0])**2 + (y - target[1])**2)
            mask = dist > 15 
            if mask.any():
                idx = random.choice(np.where(mask)[0])
                return actions.FUNCTIONS.select_point("select", (x[idx], y[idx]))
        return self._select_scv(unit_type)

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
def main(argv):
    del argv
    agent = ProductionAI()
    with sc2_env.SC2Env(
        map_name="Simple64",
        players=[sc2_env.Agent(sc2_env.Race.terran), 
                 sc2_env.Agent(sc2_env.Race.terran)],
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
                # 隨機選擇動作測試 (0-9)
                action_id = random.randint(0, 9)
                sc2_action = agent.get_action(obs_list[0], action_id)
                obs_list = env.step([sc2_action])
                if obs_list[0].last():
                    break

if __name__ == "__main__":
    app.run(main)