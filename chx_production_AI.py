import os  # 操作系統相關功能
import random  # 隨機數生成
import numpy as np  # 數值計算庫
import csv  # CSV文件處理
import time  # 時間相關功能
import platform  # 平台檢測
from absl import app  # Google的命令行應用框架
from pysc2.env import sc2_env  # StarCraft II環境
from pysc2.lib import actions, features, units  # StarCraft II動作、特徵和單位定義

# 偵測現在是 Windows 還是 Mac
if platform.system() == "Windows":
    # 如果是 Windows，強制指定你桌機的路徑
    os.environ["SC2PATH"] = r"D:\StarCraft II"
else:
    # 如果是 Mac (Darwin) 或 Linux，通常不需要設定，
    # burnysc2 會自動去 /Applications/StarCraft II 找
    pass

# =========================================================
# 🏗️ 定義人族單位 ID (Constants)
# =========================================================
COMMAND_CENTER_ID = 18  # 指揮中心單位ID
SUPPLY_DEPOT_ID = 19  # 補給站單位ID
REFINERY_ID = 20  # 瓦斯廠單位ID
BARRACKS_ID = 21  # 兵營單位ID
BARRACKS_TECHLAB_ID = 37  # 兵營科技實驗室單位ID
SCV_ID = 45  # 工兵單位ID
MARAUDER_ID = 51  # 掠奪者單位ID
MINERAL_FIELD_ID = 341  # 礦物田單位ID
GEYSER_ID = 342  # 瓦斯泉單位ID

# =========================================================
# 📊 數據收集器: 紀錄資源與訓練狀態
# =========================================================
class DataCollector:
    def __init__(self):
        # 如果logs目錄不存在，則創建它
        if not os.path.exists("logs"):
            os.makedirs("logs")
        # 設置日誌文件名，包含時間戳
        self.filename = f"logs/terran_log_{int(time.time())}.csv"
        # 建立 CSV 標頭
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "Minerals", "Vespene", "Workers", "Ideal", "Action_ID", "Marauders_Produced"])

    def log_step(self, time_val, minerals, vespene, workers, ideal, action_id, marauders_produced):
        # 轉為 float 以避免 NumPy 類型在 round 時報錯
        display_time = float(time_val)
        # 將數據追加到CSV文件中
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([round(display_time, 2), minerals, vespene, workers, ideal, action_id, marauders_produced])

# =========================================================
# 🧠 生產大腦: 專注於生產五隻掠奪者
# =========================================================
class ProductionAI:
    def __init__(self):
        # 初始化數據收集器
        self.collector = DataCollector()
        # 已建造的補給站數量
        self.depots_built = 0
        # 瓦斯廠目標位置
        self.refinery_target = None

        # 畫面中心點預設值
        self.cc_x_screen = 42
        self.cc_y_screen = 42

        # 已指派的瓦斯工兵數量
        self.gas_workers_assigned = 0

        # 鏡頭管理座標
        self.base_minimap_coords = None
        self.scan_points = []
        self.current_scan_idx = 0

        # 掠奪者生產計數器 - 目標是生產5隻
        self.marauders_produced = 0
        self.marauder_production_complete = False

        # 追蹤建築物狀態
        self.barracks_built = False
        self.techlab_built = False
        self.refinery_built = False

    def get_action(self, obs, action_id):
        """
        專注於生產五隻掠奪者的決策映射:
        0:無動作, 1:造SCV, 2:蓋補給站, 3:蓋瓦斯廠, 4:採瓦斯,
        5:蓋兵營, 6:研發科技, 7:造掠奪者, 8:擴散掃描, 9:擴張開礦
        """
        # 獲取單位類型、玩家信息和可用動作
        unit_type = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
        player = obs.observation.player
        available = obs.observation.available_actions

        # --- 1. 座標與防禦型掃描點初始化 ---
        # 如果基地座標尚未初始化，則進行初始化
        if self.base_minimap_coords is None:
            player_relative_mini = obs.observation.feature_minimap[features.MINIMAP_FEATURES.player_relative.index]
            y_mini, x_mini = (player_relative_mini == features.PlayerRelative.SELF).nonzero()
            if x_mini.any():
                bx, by = int(x_mini.mean()), int(y_mini.mean())
                self.base_minimap_coords = (bx, by)
                # 以基地為中心擴散的掃描點
                offsets = [(0, 0), (20, 0), (-20, 0), (0, 20), (0, -20), (15, 15), (-15, -15)]
                self.scan_points = [(np.clip(bx + dx, 0, 63), np.clip(by + dy, 0, 63)) for dx, dy in offsets]

        # --- 2. 視角跳轉邏輯 ---
        # 获取指揮中心的座標
        cc_y, cc_x = (unit_type == COMMAND_CENTER_ID).nonzero()

        # Action 9 (開礦): 若畫面看得到主基，說明還沒跳轉到礦區位置，需要移動鏡頭
        if action_id == 9 and cc_x.any():
            if len(self.scan_points) > 1:
                return actions.FUNCTIONS.move_camera(self.scan_points[1]) # 跳轉到第一個擴散點嘗試開礦

        # Action 0-7 (基礎營運): 若畫面沒基地，強制拉回主基地
        if action_id <= 7 and not cc_x.any() and self.base_minimap_coords:
            return actions.FUNCTIONS.move_camera(self.base_minimap_coords)

        # 更新基地在螢幕中的座標 (用於計算相對建築位置)
        if cc_x.any():
            self.cc_x_screen, self.cc_y_screen = int(cc_x.mean()), int(cc_y.mean())

        # 動態工兵飽和計算
        current_workers = player.food_workers
        refinery_pixels = np.sum(unit_type == REFINERY_ID)
        refinery_count = int(refinery_pixels / 80) # 80 像素約為一個建築大小
        ideal_workers = 16 + (refinery_count * 3)

        # 更新建築物狀態
        self._update_building_status(unit_type)

        # 紀錄數據 (新增掠奪者計數)
        self.collector.log_step(obs.observation.game_loop, player.minerals,
                                player.vespene, current_workers, ideal_workers, action_id,
                                self.marauders_produced)

        # 如果已經生產完成5隻掠奪者，停止生產
        if self.marauder_production_complete:
            return actions.FUNCTIONS.no_op()

        # --- 3. 專注於生產五隻掠奪者的邏輯 ---

        # [Action 1] 訓練 SCV (維持基本經濟)
        if action_id == 1:
            # 如果當前工兵數量少於理想數量且礦物足夠，則訓練SCV
            if current_workers < ideal_workers and player.minerals >= 50:
                if actions.FUNCTIONS.Train_SCV_quick.id in available:
                    return actions.FUNCTIONS.Train_SCV_quick("now")
            # 選擇指揮中心以訓練SCV
            return self._select_unit(unit_type, COMMAND_CENTER_ID)

        # [Action 2] 建造補給站 (確保有足夠補給)
        elif action_id == 2:
            # 如果礦物足夠且可以建造補給站，則建造補給站
            if player.minerals >= 100 and actions.FUNCTIONS.Build_SupplyDepot_screen.id in available:
                target = self._calc_depot_pos()
                return actions.FUNCTIONS.Build_SupplyDepot_screen("now", target)
            # 選擇SCV以建造補給站
            return self._select_scv(unit_type)

        # [Action 3] 建造瓦斯廠 (掠奪者需要瓦斯)
        elif action_id == 3:
            # 如果瓦斯廠尚未建造且礦物足夠，則建造瓦斯廠
            if not self.refinery_built and player.minerals >= 75 and actions.FUNCTIONS.Build_Refinery_screen.id in available:
                self.refinery_target = self._find_geyser(unit_type)
                if self.refinery_target:
                    return actions.FUNCTIONS.Build_Refinery_screen("now", self.refinery_target)
            # 選擇SCV以建造瓦斯廠
            return self._select_scv(unit_type)

        # [Action 4] 指派採瓦斯 (確保有瓦斯生產)
        elif action_id == 4:
            # 計算最大允許的瓦斯工兵數量
            max_gas_allowed = refinery_count * 3
            # 如果瓦斯工兵數量不足且有瓦斯廠目標，則指派工兵採集瓦斯
            if self.gas_workers_assigned < max_gas_allowed and self.refinery_target:
                if actions.FUNCTIONS.Harvest_Gather_screen.id in available:
                    self.gas_workers_assigned += 1
                    return actions.FUNCTIONS.Harvest_Gather_screen("now", self.refinery_target)
                # 選擇遠離目標的SCV以避免干擾正在採氣的工兵
                return self._select_scv_filtered(unit_type, self.refinery_target)
            return actions.FUNCTIONS.no_op()

        # [Action 5] 建造兵營 (生產掠奪者的必要建築)
        elif action_id == 5:
            # 如果兵營尚未建造且礦物足夠，則建造兵營
            if not self.barracks_built and player.minerals >= 150 and actions.FUNCTIONS.Build_Barracks_screen.id in available:
                target = self._calc_barracks_pos(obs)
                return actions.FUNCTIONS.Build_Barracks_screen("now", target)
            # 選擇SCV以建造兵營
            return self._select_scv(unit_type)

        # [Action 6] 研發科技實驗室 (造掠奪者必備)
        elif action_id == 6:
            # 如果兵營已建造且科技實驗室尚未建造，且資源足夠，則建造科技實驗室
            if self.barracks_built and not self.techlab_built and player.minerals >= 50 and player.vespene >= 25:
                if actions.FUNCTIONS.Build_TechLab_quick.id in available:
                    return actions.FUNCTIONS.Build_TechLab_quick("now")
            # 選擇兵營以建造科技實驗室
            return self._select_unit(unit_type, BARRACKS_ID)

        # [Action 7] 訓練掠奪者 (主要目標 - 生產5隻)
        elif action_id == 7:
            # 如果兵營和科技實驗室都已建造，且資源足夠，且掠奪者數量少於5隻，則訓練掠奪者
            if (self.barracks_built and self.techlab_built and
                player.minerals >= 100 and player.vespene >= 25 and
                self.marauders_produced < 5):
                if actions.FUNCTIONS.Train_Marauder_quick.id in available:
                    self.marauders_produced += 1
                    print(f"生產掠奪者: {self.marauders_produced}/5")
                    if self.marauders_produced >= 5:
                        self.marauder_production_complete = True
                        print("✅ 已成功生產5隻掠奪者！")
                    return actions.FUNCTIONS.Train_Marauder_quick("now")
            # 選擇兵營以訓練掠奪者
            return self._select_unit(unit_type, BARRACKS_ID)

        # [Action 8] 中心擴散掃描 (偵察周邊)
        elif action_id == 8:
            # 如果有掃描點，則移動鏡頭到下一個掃描點
            if self.scan_points:
                target = self.scan_points[self.current_scan_idx]
                self.current_scan_idx = (self.current_scan_idx + 1) % len(self.scan_points)
                return actions.FUNCTIONS.move_camera(target)
            return actions.FUNCTIONS.no_op()

        # [Action 9] 在視角中心建造二礦 (經濟擴張)
        elif action_id == 9:
            # 如果礦物足夠且可以建造指揮中心，則建造二礦
            if player.minerals >= 400 and actions.FUNCTIONS.Build_CommandCenter_screen.id in available:
                # 嘗試在當前畫面中心建造
                return actions.FUNCTIONS.Build_CommandCenter_screen("now", (42, 42))
            # 選擇SCV以建造二礦
            return self._select_scv(unit_type)

        # 如果沒有匹配的動作，則執行無操作
        return actions.FUNCTIONS.no_op()

    def _update_building_status(self, unit_type):
        """更新建築物狀態"""
        # 檢查兵營是否存在
        barracks_pixels = np.sum(unit_type == BARRACKS_ID)
        self.barracks_built = barracks_pixels > 0

        # 檢查科技實驗室是否存在
        techlab_pixels = np.sum(unit_type == BARRACKS_TECHLAB_ID)
        self.techlab_built = techlab_pixels > 0

        # 檢查瓦斯廠是否存在
        refinery_pixels = np.sum(unit_type == REFINERY_ID)
        self.refinery_built = refinery_pixels > 0

        # 檢查現有掠奪者數量
        marauder_pixels = np.sum(unit_type == MARAUDER_ID)
        # 每個掠奪者約佔 20 像素，調整計數
        self.marauders_produced = int(marauder_pixels / 20)

    # --- 內部輔助函式 ---
    def _select_unit(self, unit_type, unit_id):
        """選擇指定類型的單位"""
        # 獲取指定單位類型的座標
        y, x = (unit_type == unit_id).nonzero()
        if x.any():
            # 選擇單位的平均位置
            return actions.FUNCTIONS.select_point("select", (int(x.mean()), int(y.mean())))
        # 如果沒有找到單位，則執行無操作
        return actions.FUNCTIONS.no_op()

    def _select_scv(self, unit_type):
        """隨機選擇一個SCV工兵"""
        # 獲取所有SCV的座標
        y, x = (unit_type == SCV_ID).nonzero()
        if x.any():
            # 隨機選擇一個SCV
            idx = random.randint(0, len(x) - 1)
            return actions.FUNCTIONS.select_point("select", (x[idx], y[idx]))
        # 如果沒有SCV，則執行無操作
        return actions.FUNCTIONS.no_op()

    def _select_scv_filtered(self, unit_type, target):
        """ 選取遠離目標資源點的工兵，避免拉走正在採氣的人 """
        # 獲取所有SCV的座標
        y, x = (unit_type == SCV_ID).nonzero()
        if x.any() and target:
            # 計算每個SCV到目標的距離
            dist = np.sqrt((x - target[0])**2 + (y - target[1])**2)
            # 選擇距離目標大於 15 的工兵
            mask = dist > 15
            if mask.any():
                # 從符合條件的SCV中隨機選擇一個
                valid_indices = np.where(mask)[0]
                idx = random.choice(valid_indices)
                return actions.FUNCTIONS.select_point("select", (x[idx], y[idx]))
        # 如果沒有符合條件的SCV，則使用普通選擇方法
        return self._select_scv(unit_type)

    def _calc_depot_pos(self):
        """ 三角形排列座標計算 """
        # 根據已建造的補給站數量計算下一個補給站的位置
        if self.depots_built == 0:
            target = (self.cc_x_screen + 15, self.cc_y_screen + 15)
        elif self.depots_built == 1:
            target = (self.cc_x_screen + 27, self.cc_y_screen + 15)
        else:
            target = (self.cc_x_screen + 21, self.cc_y_screen + 27)
        # 更新已建造的補給站數量
        self.depots_built = (self.depots_built + 1) % 3
        # 確保座標不超出畫面邊界 (0-83)
        return (np.clip(target[0], 0, 83), np.clip(target[1], 0, 83))

    def _calc_barracks_pos(self, obs):
        """ 根據出生點自動判斷兵營位移，避免蓋在礦區 """
        # 獲取玩家在小地圖上的相對位置
        player_relative = obs.observation.feature_minimap[features.MINIMAP_FEATURES.player_relative.index]
        y_mini, x_mini = (player_relative == features.PlayerRelative.SELF).nonzero()
        # 如果基地在左邊，往右蓋；在右邊，往左蓋
        offset_x = 30 if (x_mini.mean() if x_mini.any() else 0) < 32 else -30
        # 返回兵營的建造位置，確保不超出邊界
        return (np.clip(42 + offset_x, 0, 83), 42)

    def _find_geyser(self, unit_type):
        """ 局部像素遮罩：精確鎖定單一湧泉中心 """
        # 獲取所有瓦斯泉的座標
        y, x = (unit_type == GEYSER_ID).nonzero()
        if x.any():
            # 獲取第一個瓦斯泉的座標
            ax, ay = x[0], y[0]
            # 建立遮罩只取第一個瓦斯泉附近的像素，避免平均值飄移到兩座泉中間
            mask = (np.abs(x - ax) < 10) & (np.abs(y - ay) < 10)
            if mask.any():
                # 返回第一個瓦斯泉的平均位置
                return (int(x[mask].mean()), int(y[mask].mean()))
        # 如果沒有找到瓦斯泉，則返回None
        return None

# =========================================================
# 🎮 主程式啟動器 (專注於生產五隻掠奪者)
# =========================================================
def main(argv):
    """
    主程式啟動器 - 專注於生產五隻掠奪者
    程式流程:
    1. 初始化 StarCraft II 環境
    2. 建立生產 AI 代理
    3. 進入無限對局循環
    4. 每局重置狀態並專注生產掠奪者
    5. 根據建築物狀態選擇適當動作
    6. 完成5隻掠奪者後停止生產
    """
    # 刪除未使用的argv參數
    del argv
    # 創建生產AI代理
    agent = ProductionAI()
    try:
        # 初始化StarCraft II環境
        with sc2_env.SC2Env(
            map_name="Simple64",  # 使用Simple64地圖
            players=[sc2_env.Agent(sc2_env.Race.terran),  # 人族玩家
                     sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.easy)],  # 簡單難度的蟲族電腦
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64),  # 畫面和小地圖尺寸
                use_raw_units=False),  # 不使用原始單位數據
            step_mul=16,     # 動作頻率 (APM 控制)
            realtime=False,  # 加速模式
        ) as env:
            # 進入無限對局循環
            while True:
                print("--- 啟動新對局: 目標生產5隻掠奪者 ---")
                # 重置環境並開始新對局
                obs_list = env.reset()
                # 重置每局狀態
                agent.depots_built = 0
                agent.marauders_produced = 0
                agent.marauder_production_complete = False
                agent.gas_workers_assigned = 0

                # 遊戲主循環
                while True:
                    # 專注於生產掠奪者的動作優先級:
                    # 1. 先建立基本設施 (SCV, 補給站, 瓦斯廠, 兵營, 科技實驗室)
                    # 2. 然後專注生產掠奪者直到達到5隻
                    if agent.marauders_produced < 5:
                        # 如果科技實驗室尚未建造，隨機選擇建造相關動作
                        if not agent.techlab_built:
                            action_id = random.randint(1, 6)
                        else:
                            # 如果科技實驗室已建造，專注生產掠奪者
                            action_id = 7
                    else:
                        # 如果已生產5隻掠奪者，隨機選擇任何動作
                        action_id = random.randint(0, 9)

                    # 獲取AI動作並執行
                    sc2_action = agent.get_action(obs_list[0], action_id)
                    obs_list = env.step([sc2_action])

                    # 如果遊戲結束，跳出內層循環
                    if obs_list[0].last():
                        break
    except KeyboardInterrupt:
        # 如果用戶手動中斷程式，打印提示信息
        print("程式已手動停止")

# 如果腳本被直接執行，則運行main函數
if __name__ == "__main__":
    app.run(main)
