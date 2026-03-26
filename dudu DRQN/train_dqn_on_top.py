import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import time
from collections import deque
from pysc2.env import sc2_env
from pysc2.lib import actions, features

# 匯入底層腳本
# --- train_dqn_on_top.py 的最上方 ---
import production_ai
from production_ai import ProductionAI
import logging
from absl import logging as absl_logging

# 屏蔽 features.py 產出的警告訊息
absl_logging.set_verbosity(absl_logging.ERROR)
# --- 1. 定義 Action ID 與 Unit ID 的對應  ---
TARGET_UNIT_MAP = {
    14: production_ai.SCV_ID,       16: 48,  # Marine
    17: 49,  # Reaper              18: production_ai.MARAUDER_ID,
    19: 50,  # Ghost               20: 53,  # Hellion
    21: 484, # Hellbat             22: 498, # WidowMine
    23: 33,  # SiegeTank           24: 692, # Cyclone
    25: 52,  # Thor                26: 34,  # Viking
    27: 54,  # Medivac             28: 689, # Liberator
    29: 56,  # Raven               30: 57,  # Battlecruiser
    31: 55,  # Banshee             32: production_ai.PLANETARY_FORTRESS_ID
}

PIXELS_PER_UNIT = {
    production_ai.SCV_ID: 15,
    48: 10,  # Marine
    49: 15,  # Reaper
    production_ai.MARAUDER_ID: 22, # Marauder 體型較大，約 20-25 像素
    50: 15,  # Ghost
    33: 150, # Siege Tank (建築/重型單位像素較多)
    # 建築物類建議只要像素 > 0 就算 1 棟，或是給予較大除數
}

# --- 📍 新增：任務映射配置表 (人族 18 單位完整版) ---
REWARD_CONFIG = {
    14: {"name": "SCV", "id": 45, "first": 20.0, "repeat": 20.0, "pixel": 10.0, "req": []},
    16: {"name": "陸戰隊", "id": 48, "first": 30.0, "repeat": 30.0, "pixel": 10.0, "req": ['barracks']},
    18: {"name": "掠奪者", "id": 51, "first": 300.0, "repeat": 300.0, "pixel": 10.0, "req": ['techlab']},
    'depot': 50.0,      # 造出補給站
    'refinery': 80.0,    # 造出瓦斯廠
    'barracks': 120.0,   # 造出軍營
    'techlab': 150.0     # 造出科技實驗室
}

# =========================================================
# 🐒 路徑設定
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(current_dir, "log")

def patched_data_collector_init(self):
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    self.filename = os.path.join(log_dir, f"terran_log_{int(time.time())}.csv")
    with open(self.filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 【同步】加入 Barracks
        writer.writerow(["Game_Loop", "Minerals", "Vespene", "Workers", "Ideal", "Barracks", "Action_ID"])

production_ai.DataCollector.__init__ = patched_data_collector_init

# =========================================================
# 📊 訓練紀錄器 (已修正參數數量與整數轉換)
# =========================================================
# =========================================================
# 📊 訓練紀錄器 (更新為 7 個核心欄位)
# =========================================================
class TrainingLogger:
    def __init__(self):
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        self.filename = os.path.join(log_dir, f"dqn_training_log_{int(time.time())}.csv")
        # 寫入新的 7 個標題
        with open(self.filename, mode='w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Task", "Epsilon", "Total_Reward", "End_Loop", "Marauders", "Marines"])

    # 更新參數順序以對應新標題
    def log_episode(self, ep, task_name, eps, reward, end_loop, marauders_cnt, marines_cnt):
        if hasattr(reward, "item"): 
            reward = reward.item()
        
        with open(self.filename, mode='a', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow([ep, task_name, f"{eps:.3f}", int(reward), end_loop, marauders_cnt, marines_cnt])
class Logger:
    def __init__(self, filename):
        self.filename = filename
        # 標題列增加 Target_Count
        header = ['Episode', 'Task', 'Epsilon', 'Total_Reward', 'Barracks', 'TechLabs', 'Target_Count', 'End_Loop', 'Reason', 'Is_Bottom_Right']
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        with open(self.filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def log_episode(self, ep, task, eps, reward, b_count, t_count, target_count, loop, reason, is_br):
        with open(self.filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([ep, task, eps, reward, b_count, t_count, target_count, loop, reason, is_br])
# =========================================================
# 🧠 深度學習模型 (DQN)
# =========================================================
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        
        # ✨ 植入海馬迴：新增 LSTM 記憶層
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        
        self.fc_action = nn.Linear(128, action_size)
        self.fc_param = nn.Linear(128, 64)

    # ✨ forward 必須接收上一步的記憶 (hidden)
    def forward(self, x, hidden):
        x = torch.relu(self.fc1(x))
        
        # LSTM 需要 3D 維度 (batch_size, sequence_length, features)
        x = x.unsqueeze(1) 
        
        # 通過 LSTM，並吐出「新的記憶」
        x, new_hidden = self.lstm(x, hidden)
        
        x = x.squeeze(1) # 轉回 2D 交給輸出層
        
        action_q = self.fc_action(x)
        param_q = self.fc_param(x)
        
        return action_q, param_q, new_hidden
    
    
    
def get_state_vector(obs, current_block, target_project_id, last_action_id, last_action_success, current_loop):
    player = obs.observation.player
    m_unit = obs.observation.feature_minimap.unit_type
    m_relative = obs.observation.feature_minimap.player_relative
    float(player.minerals) / 2000.0,  # 晶礦儲量
    # 判斷選中狀態
    is_scv_selected = 1.0 if any(u.unit_type == 45 for u in obs.observation.multi_select) else 0.0
    is_cc_selected = 1.0 if (len(obs.observation.single_select) > 0 and 
                             obs.observation.single_select[0].unit_type == 18) else 0.0

    # 計算單位數量 (從迷霧/小地圖計算)
    def count_unit(unit_id):
        return np.sum((m_unit == unit_id) & (m_relative == 1))

# 建立 15 維特徵向量
    # 建立 15 維特徵向量
    state_list = [
        min(float(player.food_workers) / 50.0, 1.0),   # 1. 工兵數量 (最高算到 50 隻)
        min(float(player.minerals) / 2000.0, 1.0),     # 2. 晶礦儲量 (大於 2000 一律算 1.0)
        min(float(player.vespene) / 1000.0, 1.0),      # 3. 瓦斯儲量
        min(float(player.food_used) / 200.0, 1.0),     # 4. 目前人口
        min(float(count_unit(19)) / 10.0, 1.0),        # 5. 補給站數量
        min(float(count_unit(20)) / 2.0, 1.0),         # 6. 瓦斯廠數量
        min(float(count_unit(21)) / 5.0, 1.0),         # 7. 軍營數量
        min(float(count_unit(37)) / 5.0, 1.0),         # 8. 科技實驗室
        min(float(is_scv_selected), 1.0),              # 9. 選中工兵
        min(float(is_cc_selected), 1.0),               # 10. 選中主堡 (順序調換以維持一致性)
        min(float(count_unit(48)) / 50.0, 1.0),        # 11. 陸戰隊數量
        min(float(count_unit(51)) / 30.0, 1.0),        # 12. 掠奪者數量
        min(float(target_project_id) / 40.0, 1.0),     # 13. 任務 ID
        min(float(current_block) / 64.0, 1.0),         # 14. 目前視角區塊
        min(float(last_action_success), 1.0),           # 15. 上一個動作是
        min(float(current_loop) / 13440.0, 1.0)        # ✨ 16. 遊戲時間進度 (0.0 = 開局, 1.0 = 時間到)否成功
    ]
    return state_list



# =========================================================
# 🎮 訓練主程式
# =========================================================
def main(argv):
    del argv
    state_size = 16
    VALID_ACTIONS = [1, 2, 11, 14, 16, 18, 34, 41, 42, 44, 45]
    action_size = len(VALID_ACTIONS)
    batch_size = 64  # ✨ 新增：每次訓練抓取的樣本數
    train_step_counter = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    brain_model = QNetwork(state_size, action_size).to(device)
    optimizer = optim.Adam(brain_model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()
    memory = deque(maxlen=100000)
    success_memory = deque(maxlen=20000)

    gamma = 0.99
    logger = TrainingLogger() # 使用 TrainingLogger 紀錄產量
    learn_min = 0.01
    last_action_id = 0   
    current_block = 1 
    # ✨ 1. 創造第二大腦 (Target Model)，並複製第一大腦的初始智慧
    target_model = QNetwork(state_size, action_size).to(device)
    target_model.load_state_dict(brain_model.state_dict())
    target_model.eval() # 設定為評估模式，不參與梯度更新
    
    optimizer = optim.Adam(brain_model.parameters(), lr=0.0005)   
    
    task_info = REWARD_CONFIG[18] 
    CURRENT_TRAIN_TASK = 18
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True) # 確保資料夾一定存在
    model_path = os.path.join(log_dir, "dqn_model.pth")
    if os.path.exists(model_path):
        # ✨ 真正的載入舊記憶
        brain_model.load_state_dict(torch.load(model_path, map_location=device))
        # 載入後，記得也要同步給第二大腦 (Target Model)
        target_model.load_state_dict(brain_model.state_dict())
        print("✅ 載入成功！接續之前的記憶繼續訓練...")

    epsilon = 1.00; epsilon_decay = 0.995; gamma = 0.99 
    def train_model():
        if len(memory) < batch_size:
            return
            
        half_batch = batch_size // 2
        
        # 你的黃金記憶庫邏輯...
        if len(success_memory) >= half_batch:
            batch_normal = random.sample(memory, half_batch)
            batch_success = random.sample(success_memory, half_batch)
            batch = batch_normal + batch_success
            # ✨ 2. 記得加上這行！把兩種記憶打亂，避免 AI 死背順序
            random.shuffle(batch) 
        else:
            batch = random.sample(memory, batch_size)

        states = torch.FloatTensor(np.array([x[0] for x in batch])).to(device)
        actions = torch.LongTensor(np.array([x[1] for x in batch])).to(device)
        rewards = torch.FloatTensor(np.array([x[2] for x in batch])).to(device)
        next_states = torch.FloatTensor(np.array([x[3] for x in batch])).to(device)
        # ...(前面 states, actions, rewards 的解包保持不變)...
        dones = torch.FloatTensor(np.array([x[4] for x in batch])).to(device)
        
        # ✨ N-Step 新增：讀取包裹裡的 Gamma 衰減乘數 (第 7 個元素)
        gamma_mults = torch.FloatTensor(np.array([x[7] for x in batch])).to(device)

        # 把記憶包從 Memory 中解開
        h_in = torch.cat([x[5][0] for x in batch], dim=1)
        c_in = torch.cat([x[5][1] for x in batch], dim=1)
        next_h_in = torch.cat([x[6][0] for x in batch], dim=1)
        next_c_in = torch.cat([x[6][1] for x in batch], dim=1)

        q_actions, _, _ = brain_model(states, (h_in, c_in))
        q_values = q_actions.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # 1. 第一大腦 (Brain) 負責挑選未來最棒的「動作」
            next_q_actions_brain, _, _ = brain_model(next_states, (next_h_in, next_c_in))
            best_actions = next_q_actions_brain.max(1)[1].unsqueeze(1) 
            
            # 2. 第二大腦 (Target) 負責幫這個動作「客觀打分數」
            next_q_actions_target, _, _ = target_model(next_states, (next_h_in, next_c_in))
            max_next_q = next_q_actions_target.gather(1, best_actions).squeeze(1)
            
            # ✨ 3. N-Step 結合：使用包裹裡的 gamma_mults 算最終期望值
            target_q = rewards + (gamma_mults * max_next_q * (1 - dones))
            

        loss = nn.MSELoss()(q_values, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with sc2_env.SC2Env(
        map_name="Simple96",
        players=[sc2_env.Agent(sc2_env.Race.terran),sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy)], #sc2_env.Agent(sc2_env.Race.terran)
        agent_interface_format=sc2_env.AgentInterfaceFormat(
            feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64), use_raw_units=False),
        step_mul=16, realtime=False
        
    ) as env:
        
        for ep in range(3000):
            agent = ProductionAI()
            obs_list = env.reset()
            obs = obs_list[0]
            
            episode_memory = []  
            # ✨ 1. N-Step 新增：準備一個長度為 3 的滑動視窗
            n_step_buffer = deque(maxlen=7) 
            total_reward = 0
            episode_memory = []
            current_block = 1       # 目前地圖區塊
            last_action_id = 0      
            last_action_success = 1.0 # ✨ 新增：預設第一步為成功
            marine_count = 0        # 預設陸戰隊數量
            last_target_count = 0   # 追蹤掠奪者數量
            last_marine_count = 0   # ✨ 新增這行：追蹤陸戰隊數量
            milestones = {}
            next_obs = obs
            last_depot_count = 0    # 用於判斷補給站是否蓋過頭
            off_screen_steps = 0
            
            agent.locked_action = None
            agent.lock_timer = 0
            agent.cc_is_bound = False
            achieved_milestones = set()  # ✨ 加入這行，解決 NameError

            hidden_state = (
                torch.zeros(1, 1, 128).to(device),
                torch.zeros(1, 1, 128).to(device)
            )
            print(f"\n--- 啟動第 {ep + 1} 局 ---")
            while True:
                
                
                
                
                # 根據任務設定更新 ID
                task_cfg = REWARD_CONFIG.get(CURRENT_TRAIN_TASK, {})
                target_project_id = 18 
                
                train_step_counter += 1

                # ==========================================
                # ✨ 動態動作遮罩 (Dynamic Action Masking)
                # 取得「當前」畫面的特徵與玩家狀態
                player = obs.observation.player
                s_unit = obs.observation.feature_screen.unit_type
                s_player = obs.observation.feature_screen.player_relative
                
                # 取得當下時間
                current_time_loop = int(obs.observation.game_loop[0])
                
                # (1) 取得當前狀態 (補上 current_time_loop)
                state = get_state_vector(obs, current_block, target_project_id, last_action_id, last_action_success, current_time_loop)
                state_t = torch.FloatTensor(np.array(state)).to(device)
               # --- ✨ 統一計算真實的建築數量 (基於執行動作前的畫面 s_unit) ---
                barracks_pixels = np.sum((s_unit == 21) & (s_player == 1))
                current_barracks = int(np.round(barracks_pixels / 137.0)) # 真實兵營數
                
                refinery_pixels = np.sum((s_unit == 20) & (s_player == 1))
                current_refineries = int(np.round(refinery_pixels / 97.0)) # 真實瓦斯廠數
                
                # 👇 加入科技實驗室的真實數量計算
                techlab_pixels = np.sum((s_unit == 37) & (s_player == 1))
                current_techlabs = int(np.round(techlab_pixels / 85.0)) # 真實科技室數
                # 👇 新增：計算補給站數量 (防呆用)
                depot_pixels = np.sum((s_unit == 19) & (s_player == 1))
                current_depots = int(np.round(depot_pixels / 69.0))
                
                # ==========================================
                # ✨ 升級版：科技樹動態動作遮罩 (Tech-Tree Masking)
                # ==========================================
                allowed_actions = [1, 2, 11, 14, 16, 18, 34, 41, 42, 44, 45]

                # B. 補給站防呆：人口足夠時，或已經有 3 座補給站，絕對不准再蓋 (Action 1)
                supply_surplus = float(player.food_cap) - float(player.food_used)
                if (supply_surplus >= 16 or current_depots >= 3) and 1 in allowed_actions:
                    allowed_actions.remove(1)
                # A. 工兵數量鎖定：一個基地最多只要 22 隻工兵，超過絕對不准再造 (Action 14)
                if float(player.food_workers) >= 22 and 14 in allowed_actions:
                    allowed_actions.remove(14)

                # B. 補給站防呆：人口足夠時，不准蓋補給站 (Action 1)
                supply_surplus = player.food_cap - player.food_used
                if supply_surplus >= 16 and 1 in allowed_actions:
                    allowed_actions.remove(1)
                    
                # C. 兵營防呆：兵營 >= 3 座，不准再蓋兵營 (Action 2)
                if current_barracks >= 3 and 2 in allowed_actions:
                    allowed_actions.remove(2)

                # D. 【關鍵】科技室與造兵防呆：如果沒有兵營，絕對不准蓋科技室或造兵！
                if current_barracks == 0:
                    for act in [16, 18, 34]: # 16陸戰隊, 18掠奪者, 34科技室
                        if act in allowed_actions: allowed_actions.remove(act)

                # E. 【關鍵】掠奪者防呆：如果沒有科技室，絕對不准造掠奪者！
                if current_techlabs == 0:
                    if 18 in allowed_actions: allowed_actions.remove(18)

                # F. 瓦斯廠防呆：瓦斯廠 >= 2 座，不准再蓋瓦斯廠 (Action 11)
                if current_refineries >= 1:
                    if 11 in allowed_actions: allowed_actions.remove(11)
                has_geyser = np.any(s_unit == 342) or np.any(s_unit == 341)
                
                if not has_geyser:
                    if 11 in allowed_actions: 
                        allowed_actions.remove(11)

                # G. 【關鍵】採瓦斯防呆：如果沒有瓦斯廠，絕對不准派工兵去採瓦斯 (Action 42)
                if current_refineries == 0:
                    if 42 in allowed_actions: allowed_actions.remove(42)

                # ==========================================
                # H. ✨ 終極 UI 反灰機制 (沒有資源/目標，按鈕絕對不亮)
                # ==========================================
                # 1. 沒建築/沒對象 絕對不准選
                if current_barracks == 0:
                    if 44 in allowed_actions: allowed_actions.remove(44) # 沒兵營不准選兵營
                    
                if player.idle_worker_count == 0:
                    if 41 in allowed_actions: allowed_actions.remove(41) # 沒閒置工兵不准選閒置工兵
                
                # 2. 資源與人口不足鎖定 (沒錢沒人口就不准造)
                minerals = float(player.minerals)
                vespene = float(player.vespene)
                supply_left = float(player.food_cap) - float(player.food_used)

                # --- 人口不足防呆 ---
                if supply_left < 1:
                    for act in [14, 16]: # SCV, 陸戰隊 (需 1 人口)
                        if act in allowed_actions: allowed_actions.remove(act)
                if supply_left < 2:
                    if 18 in allowed_actions: allowed_actions.remove(18) # 掠奪者 (需 2 人口)

                # --- 晶礦不足防呆 ---
                if minerals < 50:
                    for act in [14, 16, 34]: # SCV, 陸戰隊, 科技室
                        if act in allowed_actions: allowed_actions.remove(act)
                if minerals < 75:
                    if 11 in allowed_actions: allowed_actions.remove(11) # 瓦斯廠
                if minerals < 100:
                    for act in [1, 18]: # 補給站, 掠奪者
                        if act in allowed_actions: allowed_actions.remove(act)
                if minerals < 150:
                    if 2 in allowed_actions: allowed_actions.remove(2)   # 兵營

                # --- 瓦斯不足防呆 ---
                if vespene < 25:
                    if 18 in allowed_actions: allowed_actions.remove(18) # 掠奪者 (需 25 瓦斯)
                if vespene < 50:
                    if 34 in allowed_actions: allowed_actions.remove(34) # 科技室 (需 50 瓦斯)
                # ==========================================

                # H. 資源不足鎖定：沒錢就不要想著蓋兵營 (需要 150 礦)
                if float(player.minerals) < 150 and 2 in allowed_actions:
                    allowed_actions.remove(2)
                # ==========================================
                # ✨ 拘束器生效：將星海真實的 Action ID 轉換成神經網路的 0~10 索引
                # ==========================================
                # 取得目前「合法動作」在 0~10 裡面的對應位置
                allowed_indices = [VALID_ACTIONS.index(act) for act in allowed_actions]

                # 選擇動作 (降維版 + 拘束器發威)
                if random.random() <= epsilon:
                    # ✨ 從「合法」的選項中隨機挑選索引！(不再是全部盲選)
                    action_index = random.choice(allowed_indices)
                    p_id = random.randint(1, 64)
                    with torch.no_grad():
                        _, _, next_hidden_state = brain_model(state_t.unsqueeze(0), hidden_state)
                else:
                    with torch.no_grad():
                        # q_actions 輸出 11 個分數
                        q_actions, q_params, next_hidden_state = brain_model(state_t.unsqueeze(0), hidden_state)
                        
                        # ✨ 套用拘束器遮罩：把不合法動作的分數變成負無限大 (-inf)！
                        mask = torch.full_like(q_actions, float('-inf'))
                        mask[0, allowed_indices] = 0
                        masked_q_actions = q_actions + mask
                        
                        # 從合法的動作中，選出分數最高的那一個索引
                        action_index = masked_q_actions.argmax().item()
                        p_id = q_params.argmax().item() + 1

                # ✨ 翻譯：把選出的 0~10 索引，轉回星海真實的 Action ID 給引擎執行
                a_id = VALID_ACTIONS[action_index]
                # ==========================================
                # ==========================================
                # ✨ 核心防彈機制：在一開始就先給定預設發呆動作！
                sc2_action = actions.FUNCTIONS.no_op() 

                # 這裡是你原本的呼叫邏輯 (無論你有沒有包在 if 裡面都沒關係了)
                try:
                    sc2_action = agent.get_action(obs, a_id, parameter=p_id)
                except Exception as e:
                    print(f"⚠️ 錯誤: {e}")
                    sc2_action = actions.FUNCTIONS.no_op()
                except Exception as e:
                    # 如果腳本內部發生任何不可預期的錯誤，印出警告但不崩潰
                    print(f"⚠️ 執行動作 {a_id} 時發生錯誤: {e}，自動轉為發呆 (no_op)")

                # 執行動作並進入下一幀
                obs_list = env.step([sc2_action, actions.FUNCTIONS.no_op()])
                next_obs = obs_list[0]
                
                if sc2_action.function != 0:
                    current_action_success = 1.0
                else:
                    current_action_success = 0.0
                # --- 計算獎勵 (Reward) ---
                # 使用 next_obs 取得最新的玩家狀態與單位數量
                next_s_unit = next_obs.observation.feature_screen.unit_type
                next_s_player = next_obs.observation.feature_screen.player_relative
                
                # 計算陸戰隊數量 (ID: 48，每隻約佔 10 像素)
                marine_pixels = np.sum((next_s_unit == 48) & (next_s_player == 1))
                marine_count = int(np.round(float(marine_pixels) / 10.0))
                
                # 計算掠奪者數量 (ID: 51，每隻約佔 10 像素)
                u_pixels = np.sum((next_s_unit == 51) & (next_s_player == 1))
                current_real_count = int(np.round(float(u_pixels) / 10.0))
                

                
                actual_id = agent.locked_action if agent.locked_action is not None else a_id
                updated_block = getattr(agent, 'active_parameter', 1)
                player = next_obs.observation.player
                next_barracks_pixels = np.sum((next_obs.observation.feature_screen.unit_type == 21) & (next_obs.observation.feature_screen.player_relative == 1))
                real_barracks_count = int(np.round(next_barracks_pixels / 137.0))
                count_barracks = np.sum((next_obs.observation.feature_screen.unit_type == 21))
                agent.collector.log_step(
                    next_obs.observation.game_loop[0],
                    player.minerals,
                    player.vespene,
                    player.food_workers,
                    0, 
                    real_barracks_count, # ✨ 這裡傳入正確的數量
                    a_id
                )

                # ==========================================
                # 🏆 穩定版全局計分系統 (Dense Reward Shaping)
                # ==========================================
                step_reward = 0.0
                
                # 1. 基礎時間壓力：活著的每一幀都微扣一點點，逼迫它快點做事
                step_reward -= 0.1
                
                # 2. 裝忙懲罰：一直切換視角或框選，微扣分避免局部最佳解
                if a_id in [41, 44, 45]:
                    step_reward -= 0.5

                # 3. 防呆懲罰：超建兵營 (超過3座) 或人口過剩蓋補給站
                if a_id == 2 and current_barracks >= 3:
                    step_reward -= 50.0
                if a_id == 1:
                    supply_surplus = float(player.food_cap) - float(player.food_used)
                    if supply_surplus > 30:
                        step_reward -= 5.0

                # 4. 🍞 科技樹麵包屑 (Milestones)：每局只給一次的永久獎勵
                # (請確保在 while True 迴圈最上方有宣告 achieved_milestones = set())
                current_depots = int(np.round(np.sum((s_unit == 19) & (s_player == 1)) / 69.0))
                
                if "depot" not in achieved_milestones and current_depots >= 1:
                    step_reward += 100.0
                    achieved_milestones.add("depot")
                    
                if "barracks" not in achieved_milestones and current_barracks >= 1:
                    step_reward += 200.0
                    achieved_milestones.add("barracks")
                    
                if "refinery" not in achieved_milestones and current_refineries >= 1:
                    step_reward += 150.0
                    achieved_milestones.add("refinery")
                    
                if "techlab" not in achieved_milestones and current_techlabs >= 1:
                    step_reward += 300.0
                    achieved_milestones.add("techlab")

                # 5. ⚔️ 漸進式產量獎勵 (限制陸戰隊獎勵次數，避免刷分)
                if marine_count > getattr(agent, 'last_marine_count', 0):
                    agent.last_marine_count = marine_count
                    
                    # ✨ 修正：最多只允許 5 隻防守，而且分數降為 10 分！
                    if marine_count <= 10:  
                        step_reward += 10.0  
                        print(f"🔫 產出陸戰隊作防守！目前數量: {marine_count}")
                    else:
                        step_reward -= 50.0  # 嚴懲超量暴兵！
                        print("⚡ 警告：浪費資源造太多陸戰隊！扣 50 分")
                    
                if current_real_count > getattr(agent, 'last_target_count', 0):
                    # ✨ 修正：把掠奪者的大獎翻倍成 1000 分，製造巨大吸引力！
                    step_reward += 1000.0 
                    agent.last_target_count = current_real_count
                    print(f"🎯 產出掠奪者！目前數量: {current_real_count}")

                # 6. 🏁 終局結算：規則永遠不變
                current_loop = int(next_obs.observation.game_loop[0])
                done = next_obs.last() or current_loop >= 13440 or current_real_count >= 5

                # ==========================================
                # ⚠️ 第一步：先結算遊戲終局的龐大分數 (加分或扣分)
                # ==========================================
                if done:
                    if current_real_count >= 5:
                        step_reward += 3000.0 # 完美達成目標
                        print(f"✅ 任務成功！產出 5 隻掠奪者 (耗時: {current_loop} 幀)")
                        if current_loop <= 6720:
                            step_reward += 2000.0 # 5分鐘內極速加碼
                    else:
                        step_reward -= 1000.0 # 沒達成目標統一扣分
                        print(f"❌ 任務失敗：時間到未達成目標，結算懲罰")

                # ==========================================
                # ⚠️ 第二步：取得下一步狀態與壓縮分數
                # ==========================================
                scaled_reward = step_reward / 1000.0  # (避免梯度爆炸)
                next_time_loop = int(next_obs.observation.game_loop[0])
                next_state = get_state_vector(next_obs, getattr(agent, 'active_parameter', 1), 18, a_id, current_action_success, next_time_loop)

                # ==========================================
                # ⚠️ 第三步：N-Step Bootstrapping 緩衝與寫入
                # ==========================================
                N_STEP = 7
                GAMMA = 0.99
                
                # 1. 先把當前這一步丟進緩衝區
                current_transition = (state, action_index, scaled_reward, next_state, done, hidden_state, next_hidden_state)
                n_step_buffer.append(current_transition)

                # 2. 如果視窗滿了，結算過去 N 步的總報酬，並存入大腦
                if len(n_step_buffer) == N_STEP:
                    # 總報酬 = r_0 + (gamma * r_1) + (gamma^2 * r_2)
                    n_reward = sum([n_step_buffer[i][2] * (GAMMA ** i) for i in range(N_STEP)])
                    
                    # 拿「第 0 步」的狀態，配上「第 N 步」的結果
                    n_state = n_step_buffer[0][0]
                    n_action = n_step_buffer[0][1]
                    n_next_state = n_step_buffer[-1][3]
                    n_done = n_step_buffer[-1][4]
                    n_hidden_in = n_step_buffer[0][5]
                    n_hidden_out = n_step_buffer[-1][6]
                    
                    # ✨ 打包成新的 N-Step 包裹 (多存入了一個 Gamma 衰減乘數)
                    n_transition = (n_state, n_action, n_reward, n_next_state, n_done, n_hidden_in, n_hidden_out, GAMMA ** N_STEP)
                    
                    
                    # 滑動視窗：移除最舊的一步，讓下一步進來
                    n_step_buffer.popleft() 

                # 3. 遊戲結束時，強制清空並結算緩衝區裡剩下的尾巴
                if done:
                    while len(n_step_buffer) > 0:
                        actual_n = len(n_step_buffer)
                        n_reward = sum([n_step_buffer[i][2] * (GAMMA ** i) for i in range(actual_n)])
                        n_state = n_step_buffer[0][0]
                        n_action = n_step_buffer[0][1]
                        n_next_state = n_step_buffer[-1][3]
                        n_done = n_step_buffer[-1][4]
                        n_hidden_in = n_step_buffer[0][5]
                        n_hidden_out = n_step_buffer[-1][6]
                        
                        n_transition = (n_state, n_action, n_reward, n_next_state, n_done, n_hidden_in, n_hidden_out, GAMMA ** actual_n)
                        memory.append(n_transition)
                        episode_memory.append(n_transition)
                        n_step_buffer.popleft()
                # ==========================================
                # ==========================================
                # ⚠️ 第四步：黃金記憶守門員 (只有贏家能進)
                # ==========================================
                if done:
                    if current_real_count >= 5:
                        success_memory.extend(episode_memory)
                        print(f"🌟 黃金記憶寫入！已將 {len(episode_memory)} 步成功經驗刻入大腦深層！")

                # ==========================================
                # ⚠️ 第五步：觸發訓練與更新迴圈變數
                # ==========================================
                if len(memory) > batch_size:
                    train_model()

                hidden_state = next_hidden_state
                obs = next_obs
                total_reward += step_reward
                last_action_id = a_id
                last_action_success = current_action_success
                current_block = getattr(agent, 'active_parameter', 1)

                # --- 終局判定：跳出迴圈寫入日誌 ---

                
                # --- 終局判定：沒及格就扣分 ---
                if done:
                    
                    
                    # 1. 先從配置表中取得當前任務的設定
                    task_cfg = REWARD_CONFIG.get(CURRENT_TRAIN_TASK, {})
                    task_name = task_cfg.get('name', "未知任務")
                    
                    # 2. 初始化統計數據
                    final_b_count = 0 
                    final_t_count = 0
                    allowed_tech = task_cfg.get("req", [])
                    
                    stats_map = [
                        (production_ai.BARRACKS_ID, "兵營", "barracks"),
                        (production_ai.BARRACKS_TECHLAB_ID, "科技室", "techlab"),
                        (production_ai.SUPPLY_DEPOT_ID, "補給站", "depot"),
                        (production_ai.REFINERY_ID, "瓦斯廠", "refinery"),
                    ]
                    
                    # 3. 根據里程碑更新紀錄數據
                    for u_id, name, key in stats_map:
                        if key in allowed_tech:
                            exists = milestones.get(key, False) 
                            if key == "barracks" and exists: final_b_count = 1
                            if key == "techlab" and exists: final_t_count = 1
                    
                    # 4. 寫入 CSV 紀錄
                    logger.log_episode(ep + 1, "掠奪者任務", epsilon, total_reward, current_loop, current_real_count, marine_count)
                    break # ✨ 記得加上 break 跳出 while 迴圈
                
            
            # 回合結束後更新 epsilon
            # 回合結束後更新 epsilon
            epsilon = max(learn_min, epsilon * epsilon_decay)
            
            # ✨ 4. 大腦同步機制：每隔 5 局，讓第二大腦吸收第一大腦的最新知識！
            if ep % 5 == 0:
                target_model.load_state_dict(brain_model.state_dict())
                print(f"🔄 雙大腦同步完成 (Target Model Updated)！")
            torch.save(brain_model.state_dict(), model_path)

if __name__ == "__main__":
    from absl import app
    app.run(main)