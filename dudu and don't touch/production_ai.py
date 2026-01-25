import os  # 操作系統相關功能
import random  # 隨機數生成
import numpy as np  # 數值計算庫
import csv  # CSV文件處理
import time  # 時間相關功能
import platform  # 平台檢測
from absl import app  # Google的命令行應用框架

# Fix for random.shuffle compatibility issue
import chx_fix_random_shuffle
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
# 這些常數用於識別 StarCraft II 中的人族單位類型
# 每個單位都有唯一的 ID，用於在遊戲中識別和操作
# =========================================================
COMMAND_CENTER_ID = 18  # 指揮中心單位ID - 主要建築，用於訓練工兵和建造其他建築
SUPPLY_DEPOT_ID = 19  # 補給站單位ID - 提供人口上限，建造更多單位的必要條件
REFINERY_ID = 20  # 瓦斯廠單位ID - 用於採集瓦斯資源，掠奪者生產所需
BARRACKS_ID = 21  # 兵營單位ID - 基本軍事建築，用於訓練地面部隊
BARRACKS_TECHLAB_ID = 37  # 兵營科技實驗室單位ID - 附加建築，用於解鎖高級單位如掠奪者
SCV_ID = 45  # 工兵單位ID - 基本工人單位，用於採集資源和建造建築
MARAUDER_ID = 51  # 掠奪者單位ID - 目標生產單位，強大的地面戰鬥單位
MINERAL_FIELD_ID = 341  # 礦物田單位ID - 礦物資源點，用於採集礦物
GEYSER_ID = 342  # 瓦斯泉單位ID - 瓦斯資源點，用於採集瓦斯

# =========================================================
# 📊 數據收集器: 紀錄資源與訓練狀態
# 這個類別負責收集和記錄遊戲過程中的關鍵數據，包括:
# - 時間戳記
# - 礦物和瓦斯資源數量
# - 工兵數量和理想工兵數量
# - 當前執行的動作ID
# - 已生產的掠奪者數量
# 這些數據用於後續分析和訓練改進
# =========================================================
class DataCollector:
    def __init__(self):
        """初始化數據收集器"""
        # 如果logs目錄不存在，則創建它
        if not os.path.exists("logs"):
            os.makedirs("logs")
        # 設置日誌文件名，包含時間戳，確保每次運行都有唯一的日誌文件
        self.filename = f"logs/terran_log_{int(time.time())}.csv"
        # 建立 CSV 標頭，定義數據結構
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "Minerals", "Vespene", "Workers", "Ideal", "Action_ID", "Marauders_Produced"])

    def log_step(self, time_val, minerals, vespene, workers, ideal, action_id, marauders_produced):
        """記錄每一步的遊戲狀態數據"""
        # 轉為 float 以避免 NumPy 類型在 round 時報錯
        display_time = float(time_val.item()) if hasattr(time_val, 'item') else float(time_val)
        # 將數據追加到CSV文件中，記錄當前遊戲狀態
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([round(display_time, 2), minerals, vespene, workers, ideal, action_id, marauders_produced])

# =========================================================
# 🧠 生產大腦: 專注於生產五隻掠奪者
# 這個類別是核心AI邏輯，負責:
# 1. 資源管理和工兵分配
# 2. 建築物建造和科技樹發展
# 3. 掠奪者生產和目標達成
# 4. 鏡頭控制和地圖偵察
# 5. 數據收集和狀態追蹤
# =========================================================
class ProductionAI:
    def __init__(self):
        """初始化生產AI，設置初始狀態和變數"""
        # 初始化數據收集器，用於記錄遊戲過程中的關鍵數據
        self.collector = DataCollector()
        # 已建造的補給站數量，用於三角形排列計算
        self.depots_built = 0
        # 瓦斯廠目標位置列表，用於工兵採集瓦斯的導航
        self.refinery_targets = []
        # 已建造的瓦斯廠數量
        self.refineries_built = 0

        # 畫面中心點預設值，用於建築物位置計算
        self.cc_x_screen = 42
        self.cc_y_screen = 42

        # 已指派的瓦斯工兵數量，每個瓦斯廠最多需要3個工兵
        self.gas_workers_assigned = 0

        # 鏡頭管理座標，用於基地定位和偵察
        self.base_minimap_coords = None
        # 偵察掃描點列表，用於周邊偵察
        self.scan_points = []
        # 當前掃描點索引
        self.current_scan_idx = 0

        # 掠奪者生產計數器 - 目標是生產5隻
        self.marauders_produced = 0
        # 是否完成掠奪者生產目標
        self.marauder_production_complete = False

        # 追蹤建築物狀態，用於科技樹決策
        self.barracks_built = False
        self.techlab_built = False

        # 追蹤已嘗試的瓦斯泉位置，避免重複嘗試
        self.attempted_geyser_positions = set()
        # 當前正在建造的瓦斯廠位置
        self.current_refinery_target = None
        # 瓦斯工人分配計時器
        self.gas_worker_timer = 0

    def get_action(self, obs, action_id):
        """
        專注於生產五隻掠奪者的決策映射:
        0:無動作, 1:造SCV, 2:蓋補給站, 3:蓋瓦斯廠, 4:採瓦斯,
        5:蓋兵營, 6:研發科技, 7:造掠奪者, 8:擴散掃描, 9:擴張開礦

        這個方法是核心決策引擎，根據當前遊戲狀態和指定的動作ID，
        返回適當的StarCraft II動作。它處理以下關鍵邏輯:
        1. 鏡頭和視角管理
        2. 資源和工兵管理
        3. 建築物建造和科技樹發展
        4. 掠奪者生產和目標達成
        5. 數據收集和狀態更新
        """
        # 獲取單位類型、玩家信息和可用動作
        unit_type = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
        player = obs.observation.player
        available = obs.observation.available_actions

        # --- 1. 座標與防禦型掃描點初始化 ---
        # 如果基地座標尚未初始化，則進行初始化
        # 這個初始化過程只會在遊戲開始時執行一次
        if self.base_minimap_coords is None:
            player_relative_mini = obs.observation.feature_minimap[features.MINIMAP_FEATURES.player_relative.index]
            y_mini, x_mini = (player_relative_mini == features.PlayerRelative.SELF).nonzero()
            if x_mini.any():
                bx, by = int(x_mini.mean()), int(y_mini.mean())
                self.base_minimap_coords = (bx, by)
                # 以基地為中心擴散的掃描點，用於偵察和視角跳轉
                # 這些掃描點形成一個星形模式，覆蓋基地周邊區域
                offsets = [(0, 0), (20, 0), (-20, 0), (0, 20), (0, -20), (15, 15), (-15, -15)]
                self.scan_points = [(np.clip(bx + dx, 0, 63), np.clip(by + dy, 0, 63)) for dx, dy in offsets]

        # --- 2. 視角跳轉邏輯 ---
        # 获取指揮中心的座標，用於視角管理和建築位置計算
        cc_y, cc_x = (unit_type == COMMAND_CENTER_ID).nonzero()

        # Action 9 (開礦): 若畫面看得到主基，說明還沒跳轉到礦區位置，需要移動鏡頭
        # 這個邏輯用於經濟擴張階段，當需要建造二礦時，自動跳轉視角到擴張位置
        if action_id == 9 and cc_x.any():
            if len(self.scan_points) > 1:
                return actions.FUNCTIONS.move_camera(self.scan_points[1]) # 跳轉到第一個擴散點嘗試開礦

        # Action 0-7 (基礎營運): 若畫面沒基地，強制拉回主基地
        # 這個邏輯確保在執行基本操作時，視角始終能看到基地
        if action_id <= 7 and not cc_x.any() and self.base_minimap_coords:
            return actions.FUNCTIONS.move_camera(self.base_minimap_coords)

        # 更新基地在螢幕中的座標 (用於計算相對建築位置)
        # 這個座標用於計算補給站和兵營的建造位置
        if cc_x.any():
            self.cc_x_screen, self.cc_y_screen = int(cc_x.mean()), int(cc_y.mean())

        # 動態工兵飽和計算
        # 根據當前瓦斯廠數量計算理想工兵數量
        # 基本工兵數量為16個（採集礦物），每個瓦斯廠額外需要3個工兵
        current_workers = player.food_workers
        refinery_pixels = np.sum(unit_type == REFINERY_ID)
        refinery_count = int(refinery_pixels / 80) # 80 像素約為一個建築大小
        ideal_workers = 16 + (refinery_count * 3)

        # 計算當前實際在採集瓦斯的工兵數量
        # 找到所有在瓦斯廠附近的 SCV 工兵
        gas_workers_actual = 0
        if self.refinery_targets:
            # 計算所有瓦斯廠附近的 SCV 數量
            scv_y, scv_x = (unit_type == SCV_ID).nonzero()
            if scv_x.any() and scv_y.any():
                for refinery_target in self.refinery_targets:
                    if refinery_target:
                        distances = np.sqrt((scv_x - refinery_target[0])**2 + (scv_y - refinery_target[1])**2)
                        gas_workers_actual += np.sum(distances < 10)
        self.gas_workers_assigned = int(gas_workers_actual)  # 更新實際瓦斯工兵數量

        # 瓦斯工人分配 - 更頻繁地檢查和分配工人
        self.gas_worker_timer = (self.gas_worker_timer + 1) % 10
        if self.gas_worker_timer == 0:
            # 每10步檢查一次瓦斯工人分配，更頻繁地維護工人數量
            self._assign_gas_workers_if_needed(obs, unit_type)

        # 立即檢查並補足瓦斯工人數量，確保兩個瓦斯泉都有3個工人
        if refinery_count > 0 and self.refinery_targets:
            max_gas_allowed = refinery_count * 3
            if gas_workers_actual < max_gas_allowed and actions.FUNCTIONS.Harvest_Gather_screen.id in available:
                # 立即嘗試補足工人數量 - 直接執行工人分配邏輯
                scv_y, scv_x = (unit_type == SCV_ID).nonzero()
                if scv_x.any() and scv_y.any():
                    # 找到工人最少的瓦斯廠並優先補足
                    min_workers = float('inf')
                    target_refinery = None

                    for refinery_target in self.refinery_targets:
                        if refinery_target:
                            distances = np.sqrt((scv_x - refinery_target[0])**2 + (scv_y - refinery_target[1])**2)
                            workers_here = np.sum(distances < 10)
                            if workers_here < min_workers:
                                min_workers = workers_here
                                target_refinery = refinery_target

                    # 如果找到目標瓦斯廠，則指派工兵
                    if target_refinery:
                        # 選擇遠離目標的SCV以避免干擾正在採氣的工兵
                        dist = np.sqrt((scv_x - target_refinery[0])**2 + (scv_y - target_refinery[1])**2)
                        mask = dist > 15
                        if mask.any():
                            valid_indices = np.where(mask)[0]
                            idx = random.choice(valid_indices)
                            self.gas_workers_assigned += 1
                            return actions.FUNCTIONS.Harvest_Gather_screen("now", target_refinery)
                        else:
                            # 如果沒有遠離的工兵，選擇任何工兵
                            idx = random.randint(0, len(scv_x) - 1)
                            self.gas_workers_assigned += 1
                            return actions.FUNCTIONS.Harvest_Gather_screen("now", target_refinery)
                    elif self.refinery_targets:
                        # 如果沒有找到最優目標，使用第一個瓦斯廠目標
                        target_refinery = self.refinery_targets[0]
                        # 選擇遠離目標的SCV以避免干擾正在採氣的工兵
                        dist = np.sqrt((scv_x - target_refinery[0])**2 + (scv_y - target_refinery[1])**2)
                        mask = dist > 15
                        if mask.any():
                            valid_indices = np.where(mask)[0]
                            idx = random.choice(valid_indices)
                            self.gas_workers_assigned += 1
                            return actions.FUNCTIONS.Harvest_Gather_screen("now", target_refinery)
                        else:
                            # 如果沒有遠離的工兵，選擇任何工兵
                            idx = random.randint(0, len(scv_x) - 1)
                            self.gas_workers_assigned += 1
                            return actions.FUNCTIONS.Harvest_Gather_screen("now", target_refinery)

        # 更新建築物狀態，檢查哪些建築已經建造完成
        self._update_building_status(unit_type)

        # 紀錄數據 (新增掠奪者計數)，用於後續分析和訓練改進
        self.collector.log_step(obs.observation.game_loop, player.minerals,
                                player.vespene, current_workers, ideal_workers, action_id,
                                self.marauders_produced)

        # 如果已經生產完成5隻掠奪者，停止生產，執行無操作
        if self.marauder_production_complete:
            return actions.FUNCTIONS.no_op()

        # --- 3. 專注於生產五隻掠奪者的邏輯 ---
        # 以下是核心生產邏輯，根據不同的動作ID執行不同的操作
        # 每個動作都有明確的目標和條件檢查

        # =========================================================
        # 2. 派遣空閒工兵去挖礦: 選取空閒的工兵 >>> 尋找哪裡有礦 >>> 派遣工兵去挖
        # 4. 派遣工兵挖瓦礦: 選取空閒的工兵 >>> 尋找空閒的礦 >>> 開始挖礦
        # =========================================================
        # [Action 1] 訓練 SCV (維持基本經濟)
        if action_id == 1:
            # 如果當前工兵數量少於理想數量且礦物足夠，則訓練SCV
            if current_workers < ideal_workers and player.minerals >= 50:
                if actions.FUNCTIONS.Train_SCV_quick.id in available:
                    return actions.FUNCTIONS.Train_SCV_quick("now")
            # 選擇指揮中心以訓練SCV
            return self._select_unit(unit_type, COMMAND_CENTER_ID)

        # =========================================================
        # 1. 建造建築物: 檢查科技樹 >>> 確認資源足夠 >>> 選取空閒的工兵 >>> 選取在挖礦的工兵 >>> 使用技能 >>> 尋找可放置的地點 >>> 派遣工兵建造
        # =========================================================
        # [Action 2] 建造補給站 (確保有足夠補給)
        elif action_id == 2:
            # 如果礦物足夠且可以建造補給站，則建造補給站
            if player.minerals >= 100 and actions.FUNCTIONS.Build_SupplyDepot_screen.id in available:
                target = self._calc_depot_pos(unit_type)
                return actions.FUNCTIONS.Build_SupplyDepot_screen("now", target)
            # 選擇SCV以建造補給站
            return self._select_scv(unit_type)

        # =========================================================
        # 1. 建造建築物: 檢查科技樹 >>> 確認資源足夠 >>> 選取空閒的工兵 >>> 選取在挖礦的工兵 >>> 使用技能 >>> 尋找可放置的地點 >>> 派遣工兵建造
        # 3. 派遣工兵挖瓦斯: 選取空閒的工兵 >>> 選取在挖礦的工兵 >>> 尋找未達上限的瓦斯 >>> 派遣工兵建造瓦斯 >>> 開始挖瓦斯 >>> 補足挖瓦斯人數（三人）
        # =========================================================
        # [Action 3] 建造瓦斯廠 (掠奪者需要瓦斯) - 確保兩個瓦斯泉都建造
        elif action_id == 3:
            # 尋找所有瓦斯泉
            all_geysers = self._find_all_geysers(unit_type)

            # 如果沒有找到任何瓦斯泉，嘗試移動相機來尋找
            if not all_geysers and self.base_minimap_coords:
                # 獲取下一個相機位置來系統地搜索地圖
                next_camera_pos = self._get_next_camera_position_for_geysers()
                return actions.FUNCTIONS.move_camera(next_camera_pos)

            # 如果找到瓦斯泉，檢查哪些瓦斯泉還沒有建造瓦斯廠
            if all_geysers and player.minerals >= 75 and actions.FUNCTIONS.Build_Refinery_screen.id in available:
                # 找到還沒有瓦斯廠的瓦斯泉
                geysers_without_refineries = []
                for geyser_pos in all_geysers:
                    # 檢查這個瓦斯泉附近是否已經有瓦斯廠
                    has_refinery = False
                    for refinery_target in self.refinery_targets:
                        if refinery_target and np.sqrt((geyser_pos[0] - refinery_target[0])**2 + (geyser_pos[1] - refinery_target[1])**2) < 15:
                            has_refinery = True
                            break
                    if not has_refinery:
                        geysers_without_refineries.append(geyser_pos)

                # 如果有瓦斯泉沒有瓦斯廠，建造在第一個這樣的瓦斯泉上
                if geysers_without_refineries:
                    target_geyser = geysers_without_refineries[0]
                    # 添加到目標列表
                    if target_geyser not in self.refinery_targets:
                        self.refinery_targets.append(target_geyser)
                    return actions.FUNCTIONS.Build_Refinery_screen("now", target_geyser)

            # 如果所有瓦斯泉都已經有瓦斯廠，或者沒有資源，選擇SCV以備後續操作
            return self._select_scv(unit_type)

        # =========================================================
        # 3. 派遣工兵挖瓦斯: 選取空閒的工兵 >>> 選取在挖礦的工兵 >>> 尋找未達上限的瓦斯 >>> 派遣工兵建造瓦斯 >>> 開始挖瓦斯 >>> 補足挖瓦斯人數（三人）
        # =========================================================
        # [Action 4] 指派採瓦斯 (確保有瓦斯生產) - 更積極地維護工人數量
        elif action_id == 4:
            # 計算最大允許的瓦斯工兵數量
            max_gas_allowed = refinery_count * 3
            # 如果瓦斯工兵數量不足且有瓦斯廠目標，則指派工兵採集瓦斯
            if self.gas_workers_assigned < max_gas_allowed and self.refinery_targets:
                if actions.FUNCTIONS.Harvest_Gather_screen.id in available:
                    # 找到工人最少的瓦斯廠並優先補足
                    min_workers = float('inf')
                    target_refinery = None

                    scv_y, scv_x = (unit_type == SCV_ID).nonzero()
                    if scv_x.any() and scv_y.any():
                        for refinery_target in self.refinery_targets:
                            if refinery_target:
                                distances = np.sqrt((scv_x - refinery_target[0])**2 + (scv_y - refinery_target[1])**2)
                                workers_here = np.sum(distances < 10)
                                if workers_here < min_workers:
                                    min_workers = workers_here
                                    target_refinery = refinery_target

                    # 如果找到目標瓦斯廠，則指派工兵
                    if target_refinery:
                        self.gas_workers_assigned += 1
                        return actions.FUNCTIONS.Harvest_Gather_screen("now", target_refinery)
                    elif self.refinery_targets:
                        # 如果沒有找到最優目標，使用第一個瓦斯廠目標
                        self.gas_workers_assigned += 1
                        return actions.FUNCTIONS.Harvest_Gather_screen("now", self.refinery_targets[0])

                # 選擇遠離目標的SCV以避免干擾正在採氣的工兵
                if self.refinery_targets:
                    return self._select_scv_filtered(unit_type, self.refinery_targets[0])
            return actions.FUNCTIONS.no_op()

        # =========================================================
        # 1. 建造建築物: 檢查科技樹 >>> 確認資源足夠 >>> 選取空閒的工兵 >>> 選取在挖礦的工兵 >>> 使用技能 >>> 尋找可放置的地點 >>> 派遣工兵建造
        # =========================================================
        # [Action 5] 建造兵營 (生產掠奪者的必要建築)
        elif action_id == 5:
            # 如果兵營尚未建造且礦物足夠，則建造兵營
            if not self.barracks_built and player.minerals >= 150 and actions.FUNCTIONS.Build_Barracks_screen.id in available:
                target = self._calc_barracks_pos(obs)
                return actions.FUNCTIONS.Build_Barracks_screen("now", target)
            # 選擇SCV以建造兵營
            return self._select_scv(unit_type)

        # =========================================================
        # 6. 檢查科技樹: 查詢科技樹 >>> 尋找建築物 >>> 由淺至深檢查（迴圈） >>> 發現遺漏建築回傳 >>> 蓋被遺漏之建築
        # =========================================================
        # [Action 6] 研發科技實驗室 (造掠奪者必備)
        elif action_id == 6:
            # 如果兵營已建造且科技實驗室尚未建造，且資源足夠，則建造科技實驗室
            if self.barracks_built and not self.techlab_built and player.minerals >= 50 and player.vespene >= 25:
                if actions.FUNCTIONS.Build_TechLab_quick.id in available:
                    return actions.FUNCTIONS.Build_TechLab_quick("now")
            # 選擇兵營以建造科技實驗室
            return self._select_unit(unit_type, BARRACKS_ID)

        # =========================================================
        # 8. 建造所需單位: 檢查資源是否足夠 >>> 檢查有無該建築物
        # =========================================================
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

        # =========================================================
        # 5. 建造新主堡: 選取空閒的工兵 >>> 選取在挖礦的工兵 >>> 派遣工兵移動到最近的未開發礦點 >>> 等到資源足夠 >>> 使用技能 >>> 蓋主堡
        # =========================================================
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
        """
        更新建築物狀態

        這個方法負責檢查當前遊戲狀態中各個關鍵建築物的存在狀態，
        並更新相應的狀態變數。這些狀態變數用於後續的決策邏輯，
        確保AI知道哪些建築已經建造完成，哪些還需要建造。

        科技樹檢查流程:
        1. 檢查兵營是否存在（基本軍事建築）
        2. 檢查科技實驗室是否存在（高級單位解鎖）
        3. 檢查瓦斯廠是否存在（瓦斯資源採集）
        4. 檢查現有掠奪者數量（目標進度追蹤）
        """
        # 檢查兵營是否存在，兵營是生產地面部隊的基本建築
        barracks_pixels = np.sum(unit_type == BARRACKS_ID)
        self.barracks_built = barracks_pixels > 0

        # 檢查科技實驗室是否存在，科技實驗室是解鎖掠奪者等高級單位的必要條件
        techlab_pixels = np.sum(unit_type == BARRACKS_TECHLAB_ID)
        self.techlab_built = techlab_pixels > 0

        # 檢查瓦斯廠是否存在，瓦斯廠用於採集瓦斯資源，是生產掠奪者的必要條件
        refinery_pixels = np.sum(unit_type == REFINERY_ID)
        self.refinery_built = refinery_pixels > 0

        # 檢查現有掠奪者數量，用於追蹤目標進度
        marauder_pixels = np.sum(unit_type == MARAUDER_ID)
        # 每個掠奪者約佔 20 像素，調整計數，這個值可能需要根據實際遊戲情況調整
        self.marauders_produced = int(marauder_pixels / 20)

    # --- 內部輔助函式 ---
    # 這些輔助方法用於處理常見的單位選擇和位置計算任務，
    # 使得主要邏輯更加清晰和模組化

    def _select_unit(self, unit_type, unit_id):
        """
        選擇指定類型的單位

        這個方法用於選擇特定類型的單位，例如指揮中心、兵營等。
        它會找到所有該類型單位的位置，然後選擇它們的平均位置作為目標點。

        參數:
        - unit_type: 單位類型陣列，包含所有單位的類型信息
        - unit_id: 要選擇的單位類型ID

        返回:
        - 選擇該單位類型的動作，或者無操作（如果沒有找到該類型單位）
        """
        # 獲取指定單位類型的座標
        y, x = (unit_type == unit_id).nonzero()
        if x.any():
            # 選擇單位的平均位置，這樣可以避免選擇到單個單位可能被阻擋的問題
            return actions.FUNCTIONS.select_point("select", (int(x.mean()), int(y.mean())))
        # 如果沒有找到單位，則執行無操作
        return actions.FUNCTIONS.no_op()

    def _select_scv(self, unit_type):
        """
        隨機選擇一個SCV工兵

        這個方法用於選擇一個隨機的SCV工兵，通常用於建造任務或資源採集任務。
        隨機選擇可以避免總是選擇同一個工兵，從而更均勻地分配工作負載。

        參數:
        - unit_type: 單位類型陣列，包含所有單位的類型信息

        返回:
        - 選擇一個SCV的動作，或者無操作（如果沒有SCV）
        """
        # 獲取所有SCV的座標
        y, x = (unit_type == SCV_ID).nonzero()
        if x.any():
            # 隨機選擇一個SCV，確保工作負載均勻分配
            idx = random.randint(0, len(x) - 1)
            return actions.FUNCTIONS.select_point("select", (x[idx], y[idx]))
        # 如果沒有SCV，則執行無操作
        return actions.FUNCTIONS.no_op()

    def _select_scv_filtered(self, unit_type, target):
        """
        選取遠離目標資源點的工兵，避免拉走正在採氣的人

        這個方法用於選擇遠離特定目標（通常是瓦斯泉）的SCV工兵。
        這可以避免干擾正在採集瓦斯的工兵，確保資源採集的連續性。

        參數:
        - unit_type: 單位類型陣列，包含所有單位的類型信息
        - target: 目標位置（通常是瓦斯泉的位置）

        返回:
        - 選擇一個遠離目標的SCV的動作，或者使用普通選擇方法
        """
        # 獲取所有SCV的座標
        y, x = (unit_type == SCV_ID).nonzero()
        if x.any() and target:
            # 計算每個SCV到目標的距離
            dist = np.sqrt((x - target[0])**2 + (y - target[1])**2)
            # 選擇距離目標大於 15 的工兵，避免干擾正在工作的工兵
            mask = dist > 15
            if mask.any():
                # 從符合條件的SCV中隨機選擇一個
                valid_indices = np.where(mask)[0]
                idx = random.choice(valid_indices)
                return actions.FUNCTIONS.select_point("select", (x[idx], y[idx]))
        # 如果沒有符合條件的SCV，則使用普通選擇方法
        return self._select_scv(unit_type)

    def _find_safe_building_position(self, unit_type, target_pos, max_attempts=5):
        """
        尋找安全的建築位置，避免放置在礦物和指揮中心中間

        這個方法檢查目標位置是否會阻擋礦物採集或指揮中心操作，如果會，則尋找替代位置。

        參數:
        - unit_type: 單位類型陣列，用於檢測礦物和指揮中心位置
        - target_pos: 目標建築位置 (x, y)
        - max_attempts: 最大尋找替代位置的嘗試次數

        返回:
        - 安全的建築位置 (x, y)
        """
        # 獲取礦物和指揮中心的位置
        mineral_y, mineral_x = (unit_type == MINERAL_FIELD_ID).nonzero()
        cc_y, cc_x = (unit_type == COMMAND_CENTER_ID).nonzero()

        # 如果目標位置是安全的，直接返回
        if self._is_position_safe(unit_type, target_pos, mineral_x, mineral_y, cc_x, cc_y):
            return target_pos

        # 如果目標位置不安全，尋找替代位置
        for attempt in range(max_attempts):
            # 生成隨機偏移
            offset_x = random.randint(-10, 10)
            offset_y = random.randint(-10, 10)

            # 計算新的目標位置
            new_target = (target_pos[0] + offset_x, target_pos[1] + offset_y)

            # 檢查新位置是否安全
            if self._is_position_safe(unit_type, new_target, mineral_x, mineral_y, cc_x, cc_y):
                return new_target

        # 如果多次尋找都沒有找到安全位置，返回原始位置
        return target_pos

    def _is_position_safe(self, unit_type, target_pos, mineral_x, mineral_y, cc_x, cc_y):
        """
        檢查建築位置是否安全

        這個方法檢查目標位置是否會阻擋礦物採集或指揮中心操作。

        參數:
        - unit_type: 單位類型陣列
        - target_pos: 目標建築位置 (x, y)
        - mineral_x, mineral_y: 礦物位置座標
        - cc_x, cc_y: 指揮中心位置座標

        返回:
        - True 如果位置安全，False 如果位置不安全
        """
        # 定義安全距離
        safe_distance = 15  # 像素

        # 檢查是否有礦物在附近
        if mineral_x.any() and mineral_y.any():
            distances_to_minerals = np.sqrt((mineral_x - target_pos[0])**2 + (mineral_y - target_pos[1])**2)
            if np.any(distances_to_minerals < safe_distance):
                return False

        # 檢查是否有指揮中心在附近
        if cc_x.any() and cc_y.any():
            distances_to_cc = np.sqrt((cc_x - target_pos[0])**2 + (cc_y - target_pos[1])**2)
            if np.any(distances_to_cc < safe_distance):
                return False

        # 檢查是否在指揮中心和礦物之間
        if (mineral_x.any() and mineral_y.any() and cc_x.any() and cc_y.any()):
            # 計算指揮中心到礦物的向量
            cc_to_mineral_x = mineral_x.mean() - cc_x.mean()
            cc_to_mineral_y = mineral_y.mean() - cc_y.mean()

            # 計算指揮中心到目標位置的向量
            cc_to_target_x = target_pos[0] - cc_x.mean()
            cc_to_target_y = target_pos[1] - cc_y.mean()

            # 計算點積和向量長度
            dot_product = cc_to_mineral_x * cc_to_target_x + cc_to_mineral_y * cc_to_target_y
            mineral_distance = np.sqrt(cc_to_mineral_x**2 + cc_to_mineral_y**2)
            target_distance = np.sqrt(cc_to_target_x**2 + cc_to_target_y**2)

            # 如果目標位置在指揮中心和礦物之間，則不安全
            if (target_distance < mineral_distance and
                dot_product > 0 and
                abs(dot_product) > 0.8 * target_distance * mineral_distance):
                return False

        return True

    def _calc_depot_pos(self, unit_type):
        """
        三角形排列座標計算 - 避免放置在礦物和指揮中心中間

        這個方法用於計算補給站的建造位置，採用三角形排列模式，並避免放置在礦物和指揮中心中間。
        這種排列方式可以最大化空間利用，同時確保補給站不會阻擋彼此或資源採集。

        參數:
        - unit_type: 單位類型陣列，用於檢測礦物和指揮中心位置

        返回:
        - 下一個補給站的建造位置（x, y座標）
        """
        # 根據已建造的補給站數量計算下一個補給站的位置
        # 這種三角形排列可以確保補給站不會阻擋彼此
        if self.depots_built == 0:
            # 第一個補給站位於基地的右下方
            target = (self.cc_x_screen + 15, self.cc_y_screen + 15)
        elif self.depots_built == 1:
            # 第二個補給站位於基地的右方，與第一個補給站形成水平線
            target = (self.cc_x_screen + 27, self.cc_y_screen + 15)
        else:
            # 第三個補給站位於第一個和第二個補給站之間的下方，形成三角形
            target = (self.cc_x_screen + 21, self.cc_y_screen + 27)

        # 更新已建造的補給站數量，循環使用0-2
        self.depots_built = (self.depots_built + 1) % 3

        # 避免放置在礦物和指揮中心中間 - 尋找替代位置
        target = self._find_safe_building_position(unit_type, target)

        # 確保座標不超出畫面邊界 (0-83)
        return (np.clip(target[0], 0, 83), np.clip(target[1], 0, 83))

    def _calc_barracks_pos(self, obs):
        """
        根據出生點自動判斷兵營位移，避免蓋在礦區

        這個方法用於計算兵營的建造位置，會根據玩家的出生點位置
        自動調整兵營的位置，避免建造在礦區上方。

        參數:
        - obs: 當前遊戲觀察狀態

        返回:
        - 兵營的建造位置（x, y座標）
        """
        # 獲取玩家在小地圖上的相對位置
        player_relative = obs.observation.feature_minimap[features.MINIMAP_FEATURES.player_relative.index]
        y_mini, x_mini = (player_relative == features.PlayerRelative.SELF).nonzero()
        # 如果基地在左邊（x < 32），往右蓋；在右邊，往左蓋
        # 這樣可以避免兵營建造在礦區上方
        offset_x = 30 if (x_mini.mean() if x_mini.any() else 0) < 32 else -30
        # 返回兵營的建造位置，確保不超出邊界
        return (np.clip(42 + offset_x, 0, 83), 42)

    def _find_geyser(self, unit_type):
        """
        局部像素遮罩：精確鎖定單一湧泉中心

        這個方法用於找到瓦斯泉的精確位置，使用局部像素遮罩技術。
        這可以避免當有多個瓦斯泉時，平均位置可能落在兩個泉中間的問題。

        參數:
        - unit_type: 單位類型陣列，包含所有單位的類型信息

        返回:
        - 第一个瓦斯泉的精確位置（x, y座標），或者None（如果沒有找到瓦斯泉）
        """
        # =========================================================
        # 7. 尋找特定建築物: 在地圖上掃描有無需求建築物 >>> 發現遺漏建築回傳（有無都需回傳）
        # =========================================================
        # 獲取所有瓦斯泉的座標
        y, x = (unit_type == GEYSER_ID).nonzero()
        if x.any():
            # 獲取第一個瓦斯泉的座標
            ax, ay = x[0], y[0]
            # 建立遮罩只取第一個瓦斯泉附近的像素，避免平均值飄移到兩座泉中間
            # 這個遮罩只考慮距離第一個瓦斯泉10像素以內的區域
            mask = (np.abs(x - ax) < 10) & (np.abs(y - ay) < 10)
            if mask.any():
                # 返回第一個瓦斯泉的平均位置，這個位置更精確
                return (int(x[mask].mean()), int(y[mask].mean()))
        # 如果沒有找到瓦斯泉，則返回None
        return None

    def _find_all_geysers(self, unit_type):
        """
        找到所有瓦斯泉的位置

        這個方法用於找到地圖上所有瓦斯泉的精確位置，用於建造多個瓦斯廠。

        參數:
        - unit_type: 單位類型陣列，包含所有單位的類型信息

        返回:
        - 所有瓦斯泉的精確位置列表（x, y座標），或者空列表（如果沒有找到瓦斯泉）
        """
        # 獲取所有瓦斯泉的座標
        y, x = (unit_type == GEYSER_ID).nonzero()
        geysers = []

        if x.any():
            # 找到所有獨立的瓦斯泉
            visited = set()
            for i in range(len(x)):
                if i not in visited:
                    # 獲取當前瓦斯泉的座標
                    ax, ay = x[i], y[i]
                    # 建立遮罩只取當前瓦斯泉附近的像素
                    mask = (np.abs(x - ax) < 10) & (np.abs(y - ay) < 10)
                    if mask.any():
                        # 計算這個瓦斯泉的平均位置
                        geyser_pos = (int(x[mask].mean()), int(y[mask].mean()))
                        geysers.append(geyser_pos)
                        # 標記這個瓦斯泉的所有像素為已訪問
                        visited.update(np.where(mask)[0])

        return geysers

    def _assign_gas_workers_if_needed(self, obs, unit_type):
        """
        自動分配瓦斯工人到所有瓦斯廠

        這個方法定期檢查所有瓦斯廠並分配工人，確保每個瓦斯廠都有足夠的工人。

        參數:
        - obs: 當前遊戲觀察狀態
        - unit_type: 單位類型陣列
        """
        player = obs.observation.player
        available = obs.observation.available_actions

        # 計算當前瓦斯廠數量
        refinery_pixels = np.sum(unit_type == REFINERY_ID)
        refinery_count = int(refinery_pixels / 80)  # 80 像素約為一個建築大小

        if refinery_count > 0 and self.refinery_targets:
            # 計算每個瓦斯廠需要的工人數量
            max_gas_allowed = refinery_count * 3

            # 計算當前實際在採集瓦斯的工兵數量
            gas_workers_actual = 0
            scv_y, scv_x = (unit_type == SCV_ID).nonzero()
            if scv_x.any() and scv_y.any():
                for refinery_target in self.refinery_targets:
                    if refinery_target:
                        distances = np.sqrt((scv_x - refinery_target[0])**2 + (scv_y - refinery_target[1])**2)
                        gas_workers_actual += np.sum(distances < 10)

            # 如果瓦斯工兵數量不足，嘗試分配更多工人
            if gas_workers_actual < max_gas_allowed and actions.FUNCTIONS.Harvest_Gather_screen.id in available:
                # 找到工人最少的瓦斯廠
                min_workers = float('inf')
                target_refinery = None

                for refinery_target in self.refinery_targets:
                    if refinery_target:
                        distances = np.sqrt((scv_x - refinery_target[0])**2 + (scv_y - refinery_target[1])**2)
                        workers_here = np.sum(distances < 10)
                        if workers_here < min_workers:
                            min_workers = workers_here
                            target_refinery = refinery_target

                if target_refinery:
                    # 選擇遠離目標的SCV以避免干擾正在採氣的工兵
                    y, x = (unit_type == SCV_ID).nonzero()
                    if x.any() and target_refinery:
                        dist = np.sqrt((x - target_refinery[0])**2 + (y - target_refinery[1])**2)
                        mask = dist > 15
                        if mask.any():
                            valid_indices = np.where(mask)[0]
                            idx = random.choice(valid_indices)
                            # 直接嘗試採集瓦斯
                            return actions.FUNCTIONS.Harvest_Gather_screen("now", target_refinery)

    def _get_next_camera_position_for_geysers(self):
        """
        獲取下一個相機位置來尋找瓦斯泉

        這個方法返回預定義的相機位置，用於系統地搜索地圖上的瓦斯泉。

        返回:
        - 下一個相機位置（x, y座標），或者None如果所有位置都已嘗試
        """
        # 預定義的相機位置，覆蓋地圖的不同區域
        camera_positions = [
            (10, 50),  # 左下
            (50, 10),  # 右下
            (10, 10),  # 左上
            (50, 50),  # 右上
            (30, 30),  # 中間
        ]

        # 返回下一個未嘗試的位置
        for pos in camera_positions:
            pos_key = f"{pos[0]}_{pos[1]}"
            if pos_key not in self.attempted_geyser_positions:
                self.attempted_geyser_positions.add(pos_key)
                return pos

        # 所有位置都已嘗試，重置並返回第一個位置
        self.attempted_geyser_positions.clear()
        return camera_positions[0]

# =========================================================
# 🎮 主程式啟動器 (專注於生產五隻掠奪者)
# 這個函數是整個程式的入口點，負責:
# 1. 初始化遊戲環境和AI代理
# 2. 管理遊戲循環和對局流程
# 3. 協調AI決策和遊戲執行
# 4. 處理異常和用戶中斷
# =========================================================
def main(argv):
    """
    主程式啟動器 - 專注於生產五隻掠奪者

    程式流程:
    1. 初始化 StarCraft II 環境，設置地圖、玩家和遊戲參數
    2. 建立生產 AI 代理，負責決策和動作執行
    3. 進入無限對局循環，每局都重置狀態並專注生產掠奪者
    4. 在每局中，根據建築物狀態和目標進度選擇適當動作
    5. 完成5隻掠奪者生產後，隨機選擇動作進行其他操作
    6. 直到用戶手動中斷程式為止

    遊戲設置:
    - 地圖: Simple64（64x64的簡單地圖，適合AI訓練）
    - 玩家: 人族玩家 vs 蟲族電腦（簡單難度）
    - 畫面尺寸: 84x84像素
    - 小地圖尺寸: 64x64像素
    - 動作頻率: 16步/秒（APM控制）
    - 模式: 加速模式（非實時）
    """
    # 刪除未使用的argv參數，避免警告
    del argv
    # 創建生產AI代理，這個代理負責所有決策和動作執行
    agent = ProductionAI()
    try:
        # 初始化StarCraft II環境，使用上下文管理器確保資源正確釋放
        with sc2_env.SC2Env(
            map_name="Simple64",  # 使用Simple64地圖，適合AI訓練和測試
            players=[sc2_env.Agent(sc2_env.Race.terran),  # 人族玩家，由AI控制
                     sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.easy)],  # 簡單難度的蟲族電腦對手
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64),  # 畫面和小地圖尺寸設置
                use_raw_units=False),  # 不使用原始單位數據，使用特徵層數據
            step_mul=16,     # 動作頻率設置，16步/秒，控制AI的APM（每分鐘動作數）
            realtime=False,  # 加速模式，非實時運行，適合AI訓練
        ) as env:
            # 進入無限對局循環，直到用戶手動中斷
            while True:
                print("--- 啟動新對局: 目標生產5隻掠奪者 ---")
                # 重置環境並開始新對局，獲取初始觀察狀態
                obs_list = env.reset()
                # 重置每局狀態，確保每局都從相同的初始狀態開始
                agent.depots_built = 0
                agent.marauders_produced = 0
                agent.marauder_production_complete = False
                agent.gas_workers_assigned = 0
                agent.refinery_targets = []  # 重置瓦斯廠目標列表，確保每局都會重新建造瓦斯廠
                agent.attempted_geyser_positions = set()  # 重置瓦斯泉尋找狀態

                # 遊戲主循環，每次迭代執行一個動作
                while True:
                    # 專注於生產掠奪者的動作優先級和決策邏輯:
                    # 1. 先建立基本設施 (SCV, 補給站, 瓦斯廠, 兵營, 科技實驗室)
                    # 2. 然後專注生產掠奪者直到達到5隻目標
                    # 3. 完成目標後，隨機選擇動作進行其他操作
                    '''if agent.marauders_produced < 5:
                        # 如果科技實驗室尚未建造，表示還在基礎設施建設階段
                        # 隨機選擇建造相關動作（1-6: SCV, 補給站, 瓦斯廠, 兵營, 科技實驗室）
                        if not agent.techlab_built:
                            action_id = random.randint(1, 6)
                        else:
                            # 如果科技實驗室已建造，表示已經準備好生產掠奪者
                            # 專注生產掠奪者（動作7）
                            action_id = 7
                    else:'''
                        # 如果已生產5隻掠奪者，表示目標已達成
                        # 隨機選擇任何動作（0-9）進行其他操作或探索
                    action_id = random.randint(0, 9)

                    # 獲取AI動作並執行，將當前觀察狀態和選擇的動作ID傳遞給AI
                    sc2_action = agent.get_action(obs_list[0], action_id)
                    # 執行動作並獲取新的觀察狀態
                    obs_list = env.step([sc2_action])

                    # 如果遊戲結束（勝利或失敗），跳出內層循環，開始新對局
                    if obs_list[0].last():
                        break
    except KeyboardInterrupt:
        # 如果用戶手動中斷程式（通常是Ctrl+C），打印提示信息
        print("程式已手動停止")
        # 這裡可以添加其他清理工作，例如保存最終數據等

# 如果腳本被直接執行（而不是被導入為模組），則運行main函數
if __name__ == "__main__":
    # 使用absl.app.run來運行main函數，這個函數提供了命令行參數解析和其他實用功能
    app.run(main)
