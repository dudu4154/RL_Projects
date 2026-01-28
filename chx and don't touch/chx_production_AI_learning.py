"""
StarCraft II Q-Learning AI for Producing 5 Marauders (掠奪者)
使用Q-learning算法訓練AI生產5隻掠奪者

這個腳本實現了完整的Q-learning系統，包括：
1. 狀態表示 - 將StarCraft II遊戲狀態轉換為Q-learning可用的狀態向量
2. Q-learning代理 - 實現Q-learning算法，包括epsilon-greedy策略
3. 獎勵系統 - 設計鼓勵高效生產掠奪者的獎勵機制
4. 訓練循環 - 完整的訓練基礎架構，包括數據記錄和性能追蹤
5. 數據導出 - 使用pandas將訓練數據導出為CSV格式，方便Excel分析

作者: Cline
日期: 2026/1/27
"""

import os
import random
import numpy as np
import pandas as pd
import csv
import time
import platform
from absl import app
from collections import deque
from datetime import datetime
import json

# Fix for random.shuffle compatibility issue
import chx_fix_random_shuffle
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

# =========================================================
# 🏗️ 定義人族單位 ID (與現有代碼保持一致)
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
# 📊 狀態表示類 - 將StarCraft II遊戲狀態轉換為Q-learning狀態向量
# =========================================================
class StateRepresentation:
    """
    狀態表示類，負責將複雜的StarCraft II遊戲狀態轉換為Q-learning算法可用的狀態向量。

    狀態向量包含以下關鍵信息：
    - 資源狀態（礦物、瓦斯）
    - 建築狀態（補給站、瓦斯廠、兵營、科技實驗室）
    - 單位狀態（SCV工兵、掠奪者）
    - 工人飽和程度
    - 當前動作可用性
    """

    def __init__(self):
        """初始化狀態表示類"""
        self.state_dim = 22  # 狀態向量的維度 (12 base features + 10 one-hot action encoding)
        self.previous_state = None
        self.state_history = []

    def get_state_vector(self, obs, action_id):
        """
        從觀察狀態和當前動作ID生成狀態向量

        參數:
        - obs: 當前遊戲觀察狀態
        - action_id: 當前動作ID

        返回:
        - state_vector: 正規化後的狀態向量
        """
        unit_type = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
        player = obs.observation.player
        available = obs.observation.available_actions

        # 1. 資源狀態（0-1正規化）
        minerals_norm = min(player.minerals / 1000.0, 1.0)  # 礦物正規化
        vespene_norm = min(player.vespene / 500.0, 1.0)    # 瓦斯正規化

        # 2. 建築狀態（二進制表示）
        # 檢查建築物是否存在
        barracks_built = 1.0 if np.sum(unit_type == BARRACKS_ID) > 0 else 0.0
        techlab_built = 1.0 if np.sum(unit_type == BARRACKS_TECHLAB_ID) > 0 else 0.0
        refinery_built = 1.0 if np.sum(unit_type == REFINERY_ID) > 0 else 0.0

        # 3. 單位狀態（正規化計數）
        scv_count = min(np.sum(unit_type == SCV_ID) / 50.0, 1.0)  # 工兵數量正規化
        marauder_count = min(np.sum(unit_type == MARAUDER_ID) / 20.0, 1.0)  # 掠奪者數量正規化

        # 4. 工人飽和程度（0-1正規化）
        current_workers = player.food_workers
        refinery_pixels = np.sum(unit_type == REFINERY_ID)
        refinery_count = int(refinery_pixels / 80)  # 80像素約為一個建築大小
        ideal_workers = 16 + (refinery_count * 3)
        worker_saturation = min(current_workers / max(ideal_workers, 1), 1.0)

        # 5. 供應狀態（0-1正規化）
        supply_used_norm = player.food_used / 200.0  # 供應使用比例
        supply_cap_norm = player.food_cap / 200.0    # 供應上限比例

        # 6. 動作可用性（二進制表示）
        action_available = 1.0 if self._is_action_available(action_id, available) else 0.0

        # 7. 當前動作ID（one-hot編碼）
        action_onehot = np.zeros(10)
        if 0 <= action_id <= 9:
            action_onehot[action_id] = 1.0

        # 8. 時間進度（0-1正規化）
        # 確保從numpy數組中提取標量值
        game_loop_scalar = obs.observation.game_loop.item() if hasattr(obs.observation.game_loop, 'item') else obs.observation.game_loop
        time_progress = min(game_loop_scalar / (60 * 60 * 10), 1.0)  # 10分鐘遊戲時間正規化

        # 構建完整狀態向量 - 逐步構建以避免數組形狀問題
        base_state = np.array([
            float(minerals_norm),           # 0: 礦物狀態
            float(vespene_norm),            # 1: 瓦斯狀態
            float(barracks_built),          # 2: 兵營是否建造
            float(techlab_built),           # 3: 科技實驗室是否建造
            float(refinery_built),          # 4: 瓦斯廠是否建造
            float(scv_count),               # 5: 工兵數量
            float(marauder_count),          # 6: 掠奪者數量
            float(worker_saturation),       # 7: 工人飽和程度
            float(supply_used_norm),        # 8: 供應使用比例
            float(supply_cap_norm),         # 9: 供應上限比例
            float(action_available),        # 10: 動作是否可用
            float(time_progress)            # 11: 時間進度
        ], dtype=np.float32)

        # 添加動作one-hot編碼
        state_vector = np.concatenate([base_state, action_onehot.astype(np.float32)])

        # 存儲狀態歷史以供分析
        self.state_history.append(state_vector.copy())

        # 限制狀態歷史長度
        if len(self.state_history) > 1000:
            self.state_history.pop(0)

        return state_vector

    def _is_action_available(self, action_id, available_actions):
        """檢查指定動作是否可用"""
        action_mapping = {
            1: actions.FUNCTIONS.Train_SCV_quick.id,
            2: actions.FUNCTIONS.Build_SupplyDepot_screen.id,
            3: actions.FUNCTIONS.Build_Refinery_screen.id,
            4: actions.FUNCTIONS.Harvest_Gather_screen.id,
            5: actions.FUNCTIONS.Build_Barracks_screen.id,
            6: actions.FUNCTIONS.Build_TechLab_quick.id,
            7: actions.FUNCTIONS.Train_Marauder_quick.id,
            8: actions.FUNCTIONS.move_camera.id,
            9: actions.FUNCTIONS.Build_CommandCenter_screen.id
        }

        if action_id == 0:
            return True  # no_op總是可用

        target_action_id = action_mapping.get(action_id, None)
        return target_action_id is not None and target_action_id in available_actions

    def get_state_dimension(self):
        """返回狀態向量的維度"""
        return self.state_dim

    def clear_history(self):
        """清除狀態歷史"""
        self.state_history = []

# =========================================================
# 🧠 Q-Learning代理 - 核心學習算法
# =========================================================
class QLearningAgent:
    """
    Q-Learning代理，實現完整的Q-learning算法，包括：
    - Q表管理
    - Epsilon-greedy策略
    - 經驗回放
    - 學習率衰減
    - 探索率衰減
    """

    def __init__(self, state_dim, action_dim=10, learning_rate=0.1, discount_factor=0.95,
                 exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay=0.995):
        """
        初始化Q-learning代理

        參數:
        - state_dim: 狀態向量維度
        - action_dim: 動作空間大小（默認10個動作）
        - learning_rate: 學習率（alpha）
        - discount_factor: 折扣因子（gamma）
        - exploration_rate: 初始探索率（epsilon）
        - min_exploration_rate: 最小探索率
        - exploration_decay: 探索率衰減速度
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay = exploration_decay

        # 初始化Q表
        self.q_table = {}
        self.experience_buffer = deque(maxlen=10000)  # 經驗回放緩衝區

        # 訓練統計
        self.episode_count = 0
        self.total_rewards = []
        self.episode_lengths = []
        self.marauders_produced_history = []

        # 狀態離散化參數
        self.state_bins = 20  # 每個狀態維度的離散化bin數量

    def discretize_state(self, state_vector):
        """
        將連續狀態向量離散化為可用於Q表的離散狀態

        參數:
        - state_vector: 連續狀態向量

        返回:
        - discretized_state: 離散化後的狀態元組
        """
        # 將狀態向量離散化為整數bin
        discretized = []
        for i, val in enumerate(state_vector):
            if i < 12:  # 前12個維度是連續值
                bin_index = int(val * self.state_bins)
                discretized.append(bin_index)
            else:  # 後面是one-hot編碼，直接使用
                discretized.append(int(val))

        return tuple(discretized)

    def get_q_value(self, state, action):
        """
        获取指定狀態和動作的Q值

        參數:
        - state: 離散化狀態
        - action: 動作ID

        返回:
        - Q值
        """
        if state not in self.q_table:
            # 初始化新狀態的Q值
            self.q_table[state] = np.zeros(self.action_dim)
        return self.q_table[state][action]

    def update_q_value(self, state, action, reward, next_state, done):
        """
        使用Q-learning更新規則更新Q值

        參數:
        - state: 當前狀態
        - action: 當前動作
        - reward: 立即獎勵
        - next_state: 下一個狀態
        - done: 是否為終止狀態
        """
        current_q = self.get_q_value(state, action)

        if done:
            # 終止狀態，沒有下一個狀態的Q值
            max_next_q = 0
        else:
            # 獲取下一個狀態的最大Q值
            next_q_values = self.q_table.get(next_state, np.zeros(self.action_dim))
            max_next_q = np.max(next_q_values)

        # Q-learning更新規則：Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q

    def select_action(self, state, available_actions=None):
        """
        使用epsilon-greedy策略選擇動作

        參數:
        - state: 當前狀態
        - available_actions: 可用動作列表（可選）

        返回:
        - selected_action: 選擇的動作ID
        - is_exploration: 是否為探索動作
        """
        # 探索率衰減
        self.exploration_rate = max(self.min_exploration_rate,
                                  self.exploration_rate * self.exploration_decay)

        if random.random() < self.exploration_rate:
            # 探索：隨機選擇動作
            if available_actions is not None and len(available_actions) > 0:
                selected_action = random.choice(available_actions)
            else:
                selected_action = random.randint(0, self.action_dim - 1)
            return selected_action, True
        else:
            # 利用：選擇Q值最高的動作
            if state not in self.q_table:
                # 如果狀態不在Q表中，隨機選擇動作
                if available_actions is not None and len(available_actions) > 0:
                    selected_action = random.choice(available_actions)
                else:
                    selected_action = random.randint(0, self.action_dim - 1)
                return selected_action, False

            q_values = self.q_table[state]

            # 如果有可用動作列表，只考慮可用動作
            if available_actions is not None and len(available_actions) > 0:
                # 過濾不可用動作
                available_q_values = [q_values[a] if a in available_actions else -np.inf for a in range(self.action_dim)]
                best_action = np.argmax(available_q_values)
            else:
                best_action = np.argmax(q_values)

            return best_action, False

    def add_experience(self, state, action, reward, next_state, done):
        """
        將經驗添加到經驗回放緩衝區

        參數:
        - state: 當前狀態
        - action: 當前動作
        - reward: 立即獎勵
        - next_state: 下一個狀態
        - done: 是否為終止狀態
        """
        self.experience_buffer.append((state, action, reward, next_state, done))

    def train_from_experience(self, batch_size=32):
        """
        從經驗回放緩衝區中訓練

        參數:
        - batch_size: 每批訓練的經驗數量
        """
        if len(self.experience_buffer) < batch_size:
            return  # 經驗不足，不訓練

        # 隨機採樣經驗
        batch = random.sample(self.experience_buffer, batch_size)

        for state, action, reward, next_state, done in batch:
            self.update_q_value(state, action, reward, next_state, done)

    def save_model(self, filename):
        """
        保存Q表模型到文件

        參數:
        - filename: 保存文件名
        """
        with open(filename, 'w') as f:
            json.dump({str(k): v.tolist() for k, v in self.q_table.items()}, f)

    def load_model(self, filename):
        """
        從文件加載Q表模型

        參數:
        - filename: 加載文件名
        """
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                self.q_table = {eval(k): np.array(v) for k, v in data.items()}

    def get_exploration_rate(self):
        """獲取當前探索率"""
        return self.exploration_rate

    def increment_episode(self):
        """增加回合計數"""
        self.episode_count += 1

    def add_reward(self, reward):
        """添加獎勵到歷史記錄"""
        self.total_rewards.append(reward)

    def add_episode_length(self, length):
        """添加回合長度到歷史記錄"""
        self.episode_lengths.append(length)

    def add_marauders_produced(self, count):
        """添加掠奪者生產數量到歷史記錄"""
        self.marauders_produced_history.append(count)

# =========================================================
# 🎁 獎勵系統 - 設計鼓勵高效生產掠奪者的獎勵機制
# =========================================================
class RewardSystem:
    """
    獎勵系統，設計用於鼓勵AI高效生產5隻掠奪者的獎勵機制。

    獎勵設計原則：
    - 正向獎勵：完成關鍵步驟和目標
    - 負向獎勵：資源浪費和低效行為
    - 時間懲罰：鼓勵快速完成目標
    - 終止獎勵：完成5隻掠奪者生產
    """

    def __init__(self):
        """初始化獎勵系統"""
        self.previous_marauders = 0
        self.previous_minerals = 0
        self.previous_vespene = 0
        self.start_time = None
        self.episode_start_time = None
        # 追蹤建築物歷史最大數量
        self.max_supply_depots = 0
        self.max_barracks = 0
        self.max_techlabs = 0
        self.max_refineries = 0
        # 追蹤補給站數量（用於上限機制）
        self.supply_depot_count = 0
        # 追蹤建築物完成狀態
        self.barracks_completed = False
        self.techlab_completed = False
        self.refinery_completed = False

    def calculate_reward(self, obs, action_id, marauders_produced, done=False):
        """
        計算當前步驟的獎勵

        參數:
        - obs: 當前遊戲觀察狀態
        - action_id: 當前動作ID
        - marauders_produced: 當前掠奪者生產數量
        - done: 是否為終止狀態

        返回:
        - reward: 計算得到的獎勵值
        """
        player = obs.observation.player
        unit_type = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]

        # 初始化獎勵
        reward = 0

        # 1. 時間懲罰（每步小懲罰，鼓勵快速完成）
        reward -= 0.1

        # 2. 計算當前建築物數量
        current_supply_depots = np.sum(unit_type == SUPPLY_DEPOT_ID)
        current_barracks = np.sum(unit_type == BARRACKS_ID)
        current_techlabs = np.sum(unit_type == BARRACKS_TECHLAB_ID)
        current_refineries = np.sum(unit_type == REFINERY_ID)

        # 3. 更新歷史最大值
        if current_supply_depots > self.max_supply_depots:
            self.max_supply_depots = current_supply_depots
        if current_barracks > self.max_barracks:
            self.max_barracks = current_barracks
        if current_techlabs > self.max_techlabs:
            self.max_techlabs = current_techlabs
        if current_refineries > self.max_refineries:
            self.max_refineries = current_refineries

        # 4. 新的獎勵系統 - 只有當現在數量 > 歷史最高數量時才給分
        # 造出一隻掠奪者 +50 (大獎) - 終極目標，分數最高
        if marauders_produced > self.previous_marauders:
            reward += 50 * (marauders_produced - self.previous_marauders)

        # 蓋出兵營 +10 (中獎) - 關鍵路徑
        if current_barracks > self.max_barracks:  # 只有新建成的兵營才給分
            reward += 10

        # 蓋出科技實驗室 +10 (中獎) - 解鎖掠奪者的鑰匙
        if current_techlabs > self.max_techlabs:  # 只有新建成的科技實驗室才給分
            reward += 10

        # 蓋出瓦斯廠 +5 (小獎) - 有瓦斯才能造兵
        if current_refineries > self.max_refineries:  # 只有新建成的瓦斯廠才給分
            reward += 5

        # 蓋出補給站 +2 (小獎) - 有人口才能造兵，但不要給太高
        # 只有前3個補給站給分，第4個開始 +0 分（上限機制）
        if current_supply_depots > self.max_supply_depots:  # 只有新建成的補給站才給分
            if self.max_supply_depots < 3:  # 上限機制：只有前3個補給站給分
                reward += 2
            else:
                # For depots beyond 3, give negative reward to discourage overbuilding
                reward -= 1

        # 造出一隻工兵 (SCV) +1 (小小獎) - 經濟基礎
        current_scvs = np.sum(unit_type == SCV_ID)
        if current_scvs > np.sum(self.previous_scvs if hasattr(self, 'previous_scvs') else 0):
            reward += 1
        self.previous_scvs = current_scvs

        # 5. 無效動作 (錢不夠亂按) -1 (懲罰)
        # 檢查動作是否因為資源不足而失敗
        if action_id != 0:  # 排除no_op
            # 檢查常見的資源不足情況
            if action_id == 1 and player.minerals < 50:  # 訓練SCV需要50礦物
                reward -= 1
            elif action_id == 2 and player.minerals < 100:  # 建造補給站需要100礦物
                reward -= 1
            elif action_id == 3 and player.minerals < 75:  # 建造瓦斯廠需要75礦物
                reward -= 1
            elif action_id == 5 and player.minerals < 150:  # 建造兵營需要150礦物
                reward -= 1
            elif action_id == 6 and (player.minerals < 50 or player.vespene < 25):  # 科技實驗室需要50礦物+25瓦斯
                reward -= 1
            elif action_id == 7 and (player.minerals < 100 or player.vespene < 25):  # 掠奪者需要100礦物+25瓦斯
                reward -= 1

        # 6. 完成5隻掠奪者的終極獎勵（保留原有邏輯）
        if done and marauders_produced >= 5:
            reward += 50.0  # 完成目標的大獎勵

        # 7. 完成目標但用時過長的懲罰（保留原有邏輯）
        if done and marauders_produced >= 5:
            current_time = time.time()
            if self.episode_start_time is not None:
                episode_duration = current_time - self.episode_start_time
                # 每秒額外時間懲罰
                time_penalty = max(0, episode_duration - 300) * 0.01  # 5分鐘以上開始懲罰
                reward -= time_penalty

        # 更新之前的狀態
        self.previous_marauders = marauders_produced
        self.previous_minerals = player.minerals
        self.previous_vespene = player.vespene

        return reward

    def reset(self):
        """重置獎勵系統狀態

        在每個回合開始時調用，重置所有狀態跟蹤變量
        確保每個回合的獎勵計算是獨立的，不受上一回合影響
        """
        self.previous_marauders = 0  # 重置掠奪者計數
        self.previous_minerals = 0   # 重置礦物跟蹤
        self.previous_vespene = 0    # 重置瓦斯跟蹤
        self.previous_scvs = 0       # 重置工兵跟蹤
        self.episode_start_time = time.time()  # 記錄新回合的開始時間

        # 重置建築物歷史最大數量
        self.max_supply_depots = 0
        self.max_barracks = 0
        self.max_techlabs = 0
        self.max_refineries = 0

        # 重置建築物完成狀態
        self.barracks_completed = False
        self.techlab_completed = False
        self.refinery_completed = False

# =========================================================
# 📊 數據記錄器 - 使用pandas記錄訓練數據並導出CSV
# =========================================================
class DataLogger:
    """
    數據記錄器，使用pandas記錄完整的訓練數據，並導出為CSV格式供Excel分析。

    記錄的數據包括：
    - 訓練指標（獎勵、Q值、回合長度）
    - 性能統計（掠奪者生產時間、成功率）
    - 學習曲線
    - 動作分佈
    """

    def __init__(self):
        """初始化數據記錄器"""
        # 創建logs目錄（如果不存在）
        if not os.path.exists("logs"):
            os.makedirs("logs")

        # 創建AI_learning目錄（如果不存在）
        if not os.path.exists("logs/AI_learning"):
            os.makedirs("logs/AI_learning")

        # 設置文件名，包含時間戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f"logs/AI_learning/ql_training_{timestamp}.csv"
        self.json_filename = f"logs/AI_learning/ql_stats_{timestamp}.json"

        # 初始化數據框架 - 只記錄回合級別統計數據，適合Excel分析
        self.episode_data = pd.DataFrame(columns=[
            'Episode', 'Total_Reward', 'Episode_Length',
            'Marauders_Produced', 'Success', 'Exploration_Rate',
            'Timestamp', 'Training_Time'
        ])

        # 統計數據
        self.stats = {
            'episodes': [],
            'total_rewards': [],
            'avg_rewards': [],
            'max_rewards': [],
            'min_rewards': [],
            'episode_lengths': [],
            'marauders_produced': [],
            'success_rate': [],
            'avg_exploration_rate': [],
            'training_time': [],
            'timestamp': []
        }

        # 計時器
        self.start_time = time.time()
        self.episode_start_time = time.time()

    def log_step(self, episode, step, obs, action_id, reward, q_value, exploration_rate):
        """
        記錄單步訓練數據 - 現在只記錄回合級別統計，不記錄每步詳細數據

        參數:
        - episode: 當前回合數
        - step: 當前步數
        - obs: 當前遊戲觀察狀態
        - action_id: 當前動作ID
        - reward: 當前獎勵
        - q_value: 當前Q值
        - exploration_rate: 當前探索率
        """
        # 不再記錄每步數據，只在回合結束時記錄統計數據
        pass

    def log_episode_stats(self, episode, total_reward, episode_length, marauders_produced, success):
        """
        記錄回合統計數據

        參數:
        - episode: 回合數
        - total_reward: 總獎勵
        - episode_length: 回合長度
        - marauders_produced: 生產的掠奪者數量
        - success: 是否成功完成目標
        """
        # 計算統計數據
        training_duration = time.time() - self.start_time
        episode_duration = time.time() - self.episode_start_time

        # 記錄回合級別數據到episode_data DataFrame
        episode_data_row = {
            'Episode': episode + 1,  # 使用1-based indexing
            'Total_Reward': total_reward,
            'Episode_Length': episode_length,
            'Marauders_Produced': marauders_produced,
            'Success': 1 if success else 0,
            'Exploration_Rate': self.stats.get('exploration_rates', [0.1])[-1] if 'exploration_rates' in self.stats and self.stats['exploration_rates'] else 0.1,
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Training_Time': training_duration
        }

        # 添加到episode_data DataFrame
        self.episode_data = pd.concat([
            self.episode_data,
            pd.DataFrame([episode_data_row])
        ], ignore_index=True)

        # 更新統計數據
        self.stats['episodes'].append(episode)
        self.stats['total_rewards'].append(total_reward)
        self.stats['episode_lengths'].append(episode_length)
        self.stats['marauders_produced'].append(marauders_produced)
        self.stats['success_rate'].append(1.0 if success else 0.0)
        self.stats['training_time'].append(training_duration)
        self.stats['timestamp'].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # 重置回合計時器
        self.episode_start_time = time.time()

    def save_to_csv(self):
        """
        將訓練數據保存為CSV文件 - 現在保存回合級別統計數據
        """
        if not self.episode_data.empty:
            # 保存回合級別統計數據到CSV文件，適合Excel打開
            self.episode_data.to_csv(self.csv_filename, index=False, encoding='utf-8-sig')
            print(f"✅ 回合統計數據已保存到: {self.csv_filename}（共{len(self.episode_data)}筆數據，可用Excel打開）")
        else:
            print(f"⚠️ 沒有回合數據可保存")

    def save_stats_to_json(self):
        """
        將統計數據保存為JSON文件
        """
        # 計算額外統計數據
        if self.stats['total_rewards']:
            self.stats['avg_rewards'] = np.mean(self.stats['total_rewards'])
            self.stats['max_rewards'] = np.max(self.stats['total_rewards'])
            self.stats['min_rewards'] = np.min(self.stats['total_rewards'])
            self.stats['avg_exploration_rate'] = np.mean(self.stats.get('exploration_rates', [0.1]))

        with open(self.json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
            print(f"✅ 統計數據已保存到: {self.json_filename}")

    def get_training_summary(self):
        """
        获取訓練摘要信息

        返回:
        - summary: 訓練摘要字典
        """
        if len(self.stats['episodes']) == 0:
            return {}

        summary = {
            'total_episodes': len(self.stats['episodes']),
            'total_steps': len(self.episode_data),  # 現在使用回合數據長度
            'avg_reward': np.mean(self.stats['total_rewards']),
            'max_reward': np.max(self.stats['total_rewards']),
            'min_reward': np.min(self.stats['total_rewards']),
            'avg_episode_length': np.mean(self.stats['episode_lengths']),
            'total_marauders': np.sum(self.stats['marauders_produced']),
            'success_rate': np.mean(self.stats['success_rate']),
            'total_training_time': time.time() - self.start_time,
            'csv_file': self.csv_filename,
            'json_file': self.json_filename
        }

        return summary

    def add_exploration_rate(self, exploration_rate):
        """
        添加探索率到統計數據

        參數:
        - exploration_rate: 當前探索率
        """
        if 'exploration_rates' not in self.stats:
            self.stats['exploration_rates'] = []
        self.stats['exploration_rates'].append(exploration_rate)

# =========================================================
# 🤖 Q-Learning生產AI - 整合Q-learning與現有ProductionAI
# =========================================================
class QLearningProductionAI:
    """
    整合Q-learning與現有ProductionAI的完整AI系統。

    這個類整合了：
    - 狀態表示
    - Q-learning代理
    - 獎勵系統
    - 數據記錄
    - 與StarCraft II環境的交互
    """

    def __init__(self):
        """初始化Q-learning生產AI"""
        # 初始化組件
        self.state_representation = StateRepresentation()
        self.q_agent = QLearningAgent(self.state_representation.get_state_dimension())
        self.reward_system = RewardSystem()
        self.data_logger = DataLogger()

        # 從現有ProductionAI繼承的狀態
        self.depots_built = 0
        self.refinery_targets = []
        self.cc_x_screen = 42
        self.cc_y_screen = 42
        self.gas_workers_assigned = 0
        self.base_minimap_coords = None
        self.scan_points = []
        self.current_scan_idx = 0
        self.marauders_produced = 0
        self.marauder_production_complete = False
        self.barracks_built = False
        self.techlab_built = False
        self.attempted_geyser_positions = set()
        self.current_refinery_target = None
        self.gas_worker_timer = 0

        # 訓練參數
        self.current_episode = 0
        self.current_step = 0
        self.total_reward = 0

    def get_action(self, obs, action_id=None):
        """
        獲取基於Q-learning的動作

        參數:
        - obs: 當前遊戲觀察狀態
        - action_id: 可選的動作ID（用於測試）

        返回:
        - sc2_action: StarCraft II動作
        """
        # 獲取當前狀態
        state_vector = self.state_representation.get_state_vector(obs, action_id or 0)
        discretized_state = self.q_agent.discretize_state(state_vector)

        # 獲取可用動作
        available_actions = self._get_available_actions(obs)

        # 選擇動作（使用Q-learning或指定動作）
        if action_id is None:
            selected_action, is_exploration = self.q_agent.select_action(discretized_state, available_actions)
        else:
            selected_action = action_id
            is_exploration = False

        # 獲取Q值
        q_value = self.q_agent.get_q_value(discretized_state, selected_action)

        # 獲取獎勵
        reward = self.reward_system.calculate_reward(
            obs, selected_action, self.marauders_produced
        )

        # 記錄數據
        self.data_logger.log_step(
            self.current_episode, self.current_step, obs,
            selected_action, reward, q_value, self.q_agent.get_exploration_rate()
        )

        # 更新總獎勵
        self.total_reward += reward

        # 獲取StarCraft II動作
        sc2_action = self._get_sc2_action(obs, selected_action)

        # 更新狀態
        self.current_step += 1

        return sc2_action, selected_action, reward, q_value

    def _get_available_actions(self, obs):
        """
        获取當前可用的動作列表

        參數:
        - obs: 當前遊戲觀察狀態

        返回:
        - available_actions: 可用動作ID列表
        """
        available = obs.observation.available_actions
        available_actions = []

        # 檢查每個動作是否可用
        if actions.FUNCTIONS.Train_SCV_quick.id in available:
            available_actions.append(1)
        if actions.FUNCTIONS.Build_SupplyDepot_screen.id in available:
            available_actions.append(2)
        if actions.FUNCTIONS.Build_Refinery_screen.id in available:
            available_actions.append(3)
        if actions.FUNCTIONS.Harvest_Gather_screen.id in available:
            available_actions.append(4)
        if actions.FUNCTIONS.Build_Barracks_screen.id in available:
            available_actions.append(5)
        if actions.FUNCTIONS.Build_TechLab_quick.id in available:
            available_actions.append(6)
        if actions.FUNCTIONS.Train_Marauder_quick.id in available:
            available_actions.append(7)
        if actions.FUNCTIONS.move_camera.id in available:
            available_actions.append(8)
        if actions.FUNCTIONS.Build_CommandCenter_screen.id in available:
            available_actions.append(9)

        # 總是可用的動作
        available_actions.append(0)  # no_op

        return available_actions

    def _get_sc2_action(self, obs, action_id):
        """
        將動作ID轉換為StarCraft II動作

        參數:
        - obs: 當前遊戲觀察狀態
        - action_id: 動作ID

        返回:
        - sc2_action: StarCraft II動作
        """
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
                # 以基地為中心擴散的掃描點
                offsets = [(0, 0), (20, 0), (-20, 0), (0, 20), (0, -20), (15, 15), (-15, -15)]
                self.scan_points = [(np.clip(bx + dx, 0, 63), np.clip(by + dy, 0, 63)) for dx, dy in offsets]

        # --- 2. 視角跳轉邏輯 ---
        cc_y, cc_x = (unit_type == COMMAND_CENTER_ID).nonzero()

        # Action 9 (開礦): 若畫面看得到主基，說明還沒跳轉到礦區位置，需要移動鏡頭
        if action_id == 9 and cc_x.any():
            if len(self.scan_points) > 1:
                return actions.FUNCTIONS.move_camera(self.scan_points[1])

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

        # 計算當前實際在採集瓦斯的工兵數量
        gas_workers_actual = 0
        if self.refinery_targets:
            scv_y, scv_x = (unit_type == SCV_ID).nonzero()
            if scv_x.any() and scv_y.any():
                for refinery_target in self.refinery_targets:
                    if refinery_target:
                        distances = np.sqrt((scv_x - refinery_target[0])**2 + (scv_y - refinery_target[1])**2)
                        gas_workers_actual += np.sum(distances < 10)
        self.gas_workers_assigned = int(gas_workers_actual)

        # 瓦斯工人分配 - 更頻繁地檢查和分配工人
        self.gas_worker_timer = (self.gas_worker_timer + 1) % 10
        if self.gas_worker_timer == 0:
            self._assign_gas_workers_if_needed(obs, unit_type)

        # 更新建築物狀態
        self._update_building_status(unit_type)

        # --- 3. 專注於生產五隻掠奪者的邏輯 ---
        # [Action 1] 訓練 SCV (維持基本經濟)
        if action_id == 1:
            if current_workers < ideal_workers and player.minerals >= 50:
                if actions.FUNCTIONS.Train_SCV_quick.id in available:
                    return actions.FUNCTIONS.Train_SCV_quick("now")
            return self._select_unit(unit_type, COMMAND_CENTER_ID)

        # [Action 2] 建造補給站 (確保有足夠補給)
        elif action_id == 2:
            if player.minerals >= 100 and actions.FUNCTIONS.Build_SupplyDepot_screen.id in available:
                target = self._calc_depot_pos(unit_type)
                return actions.FUNCTIONS.Build_SupplyDepot_screen("now", target)
            return self._select_scv(unit_type)

        # [Action 3] 建造瓦斯廠 (掠奪者需要瓦斯)
        elif action_id == 3:
            all_geysers = self._find_all_geysers(unit_type)

            # 如果沒有找到任何瓦斯泉，嘗試移動相機來尋找
            if not all_geysers and self.base_minimap_coords:
                next_camera_pos = self._get_next_camera_position_for_geysers()
                return actions.FUNCTIONS.move_camera(next_camera_pos)

            # 如果找到瓦斯泉，檢查哪些瓦斯泉還沒有建造瓦斯廠
            if all_geysers and player.minerals >= 75 and actions.FUNCTIONS.Build_Refinery_screen.id in available:
                geysers_without_refineries = []
                for geyser_pos in all_geysers:
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
                    if target_geyser not in self.refinery_targets:
                        self.refinery_targets.append(target_geyser)
                    return actions.FUNCTIONS.Build_Refinery_screen("now", target_geyser)

            return self._select_scv(unit_type)

        # [Action 4] 指派採瓦斯 (確保有瓦斯生產)
        elif action_id == 4:
            max_gas_allowed = refinery_count * 3
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
                        self.gas_workers_assigned += 1
                        return actions.FUNCTIONS.Harvest_Gather_screen("now", self.refinery_targets[0])

                return self._select_scv_filtered(unit_type, self.refinery_targets[0] if self.refinery_targets else None)
            return actions.FUNCTIONS.no_op()

        # [Action 5] 建造兵營 (生產掠奪者的必要建築)
        elif action_id == 5:
            if not self.barracks_built and player.minerals >= 150 and actions.FUNCTIONS.Build_Barracks_screen.id in available:
                target = self._calc_barracks_pos(obs)
                return actions.FUNCTIONS.Build_Barracks_screen("now", target)
            return self._select_scv(unit_type)

        # [Action 6] 研發科技實驗室 (造掠奪者必備)
        elif action_id == 6:
            if self.barracks_built and not self.techlab_built and player.minerals >= 50 and player.vespene >= 25:
                if actions.FUNCTIONS.Build_TechLab_quick.id in available:
                    return actions.FUNCTIONS.Build_TechLab_quick("now")
            return self._select_unit(unit_type, BARRACKS_ID)

        # [Action 7] 訓練掠奪者 (主要目標 - 生產5隻)
        elif action_id == 7:
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
            return self._select_unit(unit_type, BARRACKS_ID)

        # [Action 8] 中心擴散掃描 (偵察周邊)
        elif action_id == 8:
            if self.scan_points:
                target = self.scan_points[self.current_scan_idx]
                self.current_scan_idx = (self.current_scan_idx + 1) % len(self.scan_points)
                return actions.FUNCTIONS.move_camera(target)
            return actions.FUNCTIONS.no_op()

        # [Action 9] 在視角中心建造二礦 (經濟擴張)
        elif action_id == 9:
            if player.minerals >= 400 and actions.FUNCTIONS.Build_CommandCenter_screen.id in available:
                return actions.FUNCTIONS.Build_CommandCenter_screen("now", (42, 42))
            return self._select_scv(unit_type)

        # 如果沒有匹配的動作，則執行無操作
        return actions.FUNCTIONS.no_op()

    def _update_building_status(self, unit_type):
        """更新建築物狀態"""
        barracks_pixels = np.sum(unit_type == BARRACKS_ID)
        self.barracks_built = barracks_pixels > 0

        techlab_pixels = np.sum(unit_type == BARRACKS_TECHLAB_ID)
        self.techlab_built = techlab_pixels > 0

        refinery_pixels = np.sum(unit_type == REFINERY_ID)
        self.refinery_built = refinery_pixels > 0

        marauder_pixels = np.sum(unit_type == MARAUDER_ID)
        self.marauders_produced = int(marauder_pixels / 20)

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
        y, x = (unit_type == SCV_ID).nonzero()
        if x.any() and target:
            dist = np.sqrt((x - target[0])**2 + (y - target[1])**2)
            mask = dist > 15
            if mask.any():
                idx = random.choice(np.where(mask)[0])
                return actions.FUNCTIONS.select_point("select", (x[idx], y[idx]))
        return self._select_scv(unit_type)

    def _calc_depot_pos(self, unit_type):
        if self.depots_built == 0:
            target = (self.cc_x_screen + 15, self.cc_y_screen + 15)
        elif self.depots_built == 1:
            target = (self.cc_x_screen + 27, self.cc_y_screen + 15)
        else:
            target = (self.cc_x_screen + 21, self.cc_y_screen + 27)

        self.depots_built = (self.depots_built + 1) % 3
        return (np.clip(target[0], 0, 83), np.clip(target[1], 0, 83))

    def _calc_barracks_pos(self, obs):
        player_relative = obs.observation.feature_minimap[features.MINIMAP_FEATURES.player_relative.index]
        y_mini, x_mini = (player_relative == features.PlayerRelative.SELF).nonzero()
        offset_x = 30 if (x_mini.mean() if x_mini.any() else 0) < 32 else -30
        return (np.clip(42 + offset_x, 0, 83), 42)

    def _find_all_geysers(self, unit_type):
        y, x = (unit_type == GEYSER_ID).nonzero()
        geysers = []

        if x.any():
            visited = set()
            for i in range(len(x)):
                if i not in visited:
                    ax, ay = x[i], y[i]
                    mask = (np.abs(x - ax) < 10) & (np.abs(y - ay) < 10)
                    if mask.any():
                        geyser_pos = (int(x[mask].mean()), int(y[mask].mean()))
                        geysers.append(geyser_pos)
                        visited.update(np.where(mask)[0])

        return geysers

    def _assign_gas_workers_if_needed(self, obs, unit_type):
        player = obs.observation.player
        available = obs.observation.available_actions

        refinery_pixels = np.sum(unit_type == REFINERY_ID)
        refinery_count = int(refinery_pixels / 80)

        if refinery_count > 0 and self.refinery_targets:
            max_gas_allowed = refinery_count * 3

            gas_workers_actual = 0
            scv_y, scv_x = (unit_type == SCV_ID).nonzero()
            if scv_x.any() and scv_y.any():
                for refinery_target in self.refinery_targets:
                    if refinery_target:
                        distances = np.sqrt((scv_x - refinery_target[0])**2 + (scv_y - refinery_target[1])**2)
                        gas_workers_actual += np.sum(distances < 10)

            if gas_workers_actual < max_gas_allowed and actions.FUNCTIONS.Harvest_Gather_screen.id in available:
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
                    y, x = (unit_type == SCV_ID).nonzero()
                    if x.any() and target_refinery:
                        dist = np.sqrt((x - target_refinery[0])**2 + (y - target_refinery[1])**2)
                        mask = dist > 15
                        if mask.any():
                            valid_indices = np.where(mask)[0]
                            idx = random.choice(valid_indices)
                            return actions.FUNCTIONS.Harvest_Gather_screen("now", target_refinery)

    def _get_next_camera_position_for_geysers(self):
        camera_positions = [
            (10, 50),  # 左下
            (50, 10),  # 右下
            (10, 10),  # 左上
            (50, 50),  # 右上
            (30, 30),  # 中間
        ]

        for pos in camera_positions:
            pos_key = f"{pos[0]}_{pos[1]}"
            if pos_key not in self.attempted_geyser_positions:
                self.attempted_geyser_positions.add(pos_key)
                return pos

        self.attempted_geyser_positions.clear()
        return camera_positions[0]

    def reset_episode(self):
        """重置回合狀態"""
        self.depots_built = 0
        self.refinery_targets = []
        self.gas_workers_assigned = 0
        self.base_minimap_coords = None
        self.scan_points = []
        self.current_scan_idx = 0
        self.marauders_produced = 0
        self.marauder_production_complete = False
        self.barracks_built = False
        self.techlab_built = False
        self.attempted_geyser_positions = set()
        self.current_refinery_target = None
        self.gas_worker_timer = 0

        self.current_step = 0
        self.total_reward = 0

        # 重置獎勵系統
        self.reward_system.reset()

        # 清除狀態歷史
        self.state_representation.clear_history()

    def end_episode(self, success=False):
        """
        結束回合並記錄統計數據

        參數:
        - success: 是否成功完成目標
        """
        # 記錄回合統計數據
        self.data_logger.log_episode_stats(
            self.current_episode,
            self.total_reward,
            self.current_step,
            self.marauders_produced,
            success
        )

        # 添加探索率到統計數據
        self.data_logger.add_exploration_rate(self.q_agent.get_exploration_rate())

        # 增加回合計數
        self.current_episode += 1

        # 訓練Q-learning代理
        self.q_agent.train_from_experience(batch_size=32)

        # 增加Q-learning代理的回合計數
        self.q_agent.increment_episode()

        # 添加獎勵到Q-learning代理歷史
        self.q_agent.add_reward(self.total_reward)
        self.q_agent.add_episode_length(self.current_step)
        self.q_agent.add_marauders_produced(self.marauders_produced)

        print(f"回合 {self.current_episode} 完成: 獎勵={self.total_reward:.2f}, "
              f"步數={self.current_step}, 掠奪者={self.marauders_produced}, "
              f"探索率={self.q_agent.get_exploration_rate():.3f}")

    def save_model(self, filename=None):
        """
        保存Q-learning模型

        參數:
        - filename: 保存文件名（可選）
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/AI_learning/ql_model_{timestamp}.json"

        self.q_agent.save_model(filename)
        print(f"✅ Q-learning模型已保存到: {filename}")

    def load_model(self, filename):
        """
        加載Q-learning模型

        參數:
        - filename: 加載文件名
        """
        self.q_agent.load_model(filename)
        print(f"✅ Q-learning模型已從 {filename} 加載")

    def save_training_data(self):
        """保存訓練數據到CSV和JSON文件"""
        self.data_logger.save_to_csv()
        self.data_logger.save_stats_to_json()

    def get_training_summary(self):
        """獲取訓練摘要信息"""
        return self.data_logger.get_training_summary()

# =========================================================
# 🎮 主訓練函數 - Q-learning訓練循環
# =========================================================
def train_ql_agent(argv, episodes=50, max_steps=5000):
    """
    主訓練函數，執行完整的Q-learning訓練循環

    參數:
    - argv: 命令行參數（未使用）
    - episodes: 訓練回合數（默認50）
    - max_steps: 每回合最大步數（默認5000）
    """
    del argv  # 刪除未使用的參數

    print("🚀 開始Q-learning訓練...")
    print(f"目標: 訓練AI生產5隻掠奪者")
    print(f"訓練參數: {episodes}回合, 每回合最多{max_steps}步")

    # 初始化Q-learning生產AI
    ql_agent = QLearningProductionAI()

    # 偵測現在是Windows還是Mac
    if platform.system() == "Windows":
        os.environ["SC2PATH"] = r"D:\StarCraft II"
    else:
        pass

    try:
        # 初始化StarCraft II環境
        with sc2_env.SC2Env(
            map_name="Simple64",
            players=[sc2_env.Agent(sc2_env.Race.terran),
                     sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.easy)],
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64),
                use_raw_units=False),
            step_mul=16,
            realtime=False,
        ) as env:
            # 訓練循環
            for episode in range(episodes):
                print(f"\n🎮 開始回合 {episode + 1}/{episodes}")

                # 重置環境
                obs_list = env.reset()
                ql_agent.reset_episode()

                # 回合循環
                for step in range(max_steps):
                    # 獲取動作
                    sc2_action, action_id, reward, q_value = ql_agent.get_action(obs_list[0])

                    # 執行動作
                    obs_list = env.step([sc2_action])

                    # 檢查是否完成目標
                    if ql_agent.marauders_produced >= 5:
                        ql_agent.end_episode(success=True)
                        break

                    # 檢查是否遊戲結束
                    if obs_list[0].last():
                        success = ql_agent.marauders_produced >= 5
                        ql_agent.end_episode(success=success)
                        break

                # 每10回合保存一次模型和數據
                if (episode + 1) % 10 == 0:
                    ql_agent.save_model()
                    ql_agent.save_training_data()

                    # 顯示訓練進度
                    summary = ql_agent.get_training_summary()
                    if summary:
                        print(f"\n📊 訓練進度（{episode + 1}回合）：")
                        print(f"   平均獎勵: {summary.get('avg_reward', 0):.2f}")
                        print(f"   最大獎勵: {summary.get('max_reward', 0):.2f}")
                        print(f"   成功率: {summary.get('success_rate', 0) * 100:.1f}%")
                        print(f"   總掠奪者生產: {summary.get('total_marauders', 0)}")
                        print(f"   平均回合長度: {summary.get('avg_episode_length', 0):.0f}步")

            # 訓練完成，保存最終模型和數據
            ql_agent.save_model()
            ql_agent.save_training_data()

            # 顯示最終訓練摘要
            final_summary = ql_agent.get_training_summary()
            print(f"\n🎉 訓練完成！")
            print(f"總回合數: {final_summary.get('total_episodes', 0)}")
            print(f"總步數: {final_summary.get('total_steps', 0)}")
            print(f"平均獎勵: {final_summary.get('avg_reward', 0):.2f}")
            print(f"最大獎勵: {final_summary.get('max_reward', 0):.2f}")
            print(f"成功率: {final_summary.get('success_rate', 0) * 100:.1f}%")
            print(f"總掠奪者生產: {final_summary.get('total_marauders', 0)}")
            print(f"總訓練時間: {final_summary.get('total_training_time', 0):.0f}秒")
            print(f"數據已保存到: {final_summary.get('csv_file', '未知')}")
            print(f"統計已保存到: {final_summary.get('json_file', '未知')}")

    except KeyboardInterrupt:
        print("\n⏹️ 訓練被手動中斷")
        # 保存中斷時的模型和數據
        ql_agent.save_model()
        ql_agent.save_training_data()
        print("✅ 模型和數據已保存")

# =========================================================
# 🏁 主程式入口
# =========================================================
if __name__ == "__main__":
    # 使用absl.app.run來運行主訓練函數
    app.run(train_ql_agent)
