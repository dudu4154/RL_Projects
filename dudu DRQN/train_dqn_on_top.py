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

import os
os.environ["SC2PATH"] = r"D:\StarCraft II"
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
    is_scv_selected = 1.0 if any(u.unit_type == 45 for u in obs.observation.multi_select) else 0.0
    is_cc_selected = 1.0 if (len(obs.observation.single_select) > 0 and 
                             obs.observation.single_select[0].unit_type == 18) else 0.0
    is_barracks_selected = 1.0 if (len(obs.observation.single_select) > 0 and 
                                   obs.observation.single_select[0].unit_type == 21) else 0.0
    # 計算單位數量 (從迷霧/小地圖計算)
    def count_unit(unit_id):
        return np.sum((m_unit == unit_id) & (m_relative == 1))

    state_list = [
        min(float(player.food_workers) / 50.0, 1.0),   # 1. 工兵數量
        min(float(player.minerals) / 2000.0, 1.0),     # 2. 晶礦儲量 
        min(float(player.vespene) / 1000.0, 1.0),      # 3. 瓦斯儲量
        min(float(player.food_used) / 200.0, 1.0),     # 4. 目前人口
        min(float(count_unit(19)) / 10.0, 1.0),        # 5. 補給站數量
        min(float(count_unit(20)) / 2.0, 1.0),         # 6. 瓦斯廠數量
        min(float(count_unit(21)) / 5.0, 1.0),         # 7. 軍營數量
        min(float(count_unit(37)) / 5.0, 1.0),         # 8. 科技室數量
        min(float(is_scv_selected), 1.0),              # 9. 選中工兵
        min(float(is_cc_selected), 1.0),               # 10. 選中主堡
        min(float(is_barracks_selected), 1.0),         # ✨ 11. 新增：選中兵營
        min(float(count_unit(48)) / 50.0, 1.0),        # 12. 陸戰隊數量
        min(float(count_unit(51)) / 30.0, 1.0),        # 13. 掠奪者數量
        min(float(target_project_id) / 40.0, 1.0),     # 14. 任務 ID
        min(float(current_block) / 64.0, 1.0),         # 15. 目前視角區塊
        min(float(last_action_success), 1.0),          # 16. 上一個動作成功與否
        min(float(current_loop) / 13440.0, 1.0)        # 17. 遊戲時間進度
    ]
    return state_list



# =========================================================
# 🎮 訓練主程式
# =========================================================
def main(argv):
    del argv
    state_size = 17  
    VALID_ACTIONS = [1, 2, 11, 14, 16, 18, 34, 41, 42, 44, 45]
    action_size = len(VALID_ACTIONS)
    batch_size = 64  # ✨ 新增：每次訓練抓取的樣本數
    train_step_counter = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    brain_model = QNetwork(state_size, action_size).to(device)
    optimizer = optim.Adam(brain_model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()
    # 一般記憶：維持 deque，只記住最近 300 局的真實血淚史
    memory = deque(maxlen=500) 

    # 菁英記憶：改成 List，用來做排行榜
    success_memory = [] 
    MAX_ELITE_MEMORY = 300 # 設定排行榜最多收錄 1000 局

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

    epsilon = 1.00; epsilon_decay = 0.99; gamma = 0.99

    def get_action_mask(target_obs):
        """ 根據當前畫面狀態，回傳合法的 Action 索引列表 """
        player = target_obs.observation.player
        s_unit = target_obs.observation.feature_screen.unit_type
        s_player = target_obs.observation.feature_screen.player_relative
        
        barracks = int(np.round(np.sum((s_unit == 21) & (s_player == 1)) / 137.0))
        refineries = int(np.round(np.sum((s_unit == 20) & (s_player == 1)) / 97.0))
        techlabs = int(np.round(np.sum((s_unit == 37) & (s_player == 1)) / 85.0))
        depots = int(np.round(np.sum((s_unit == 19) & (s_player == 1)) / 69.0))
        
        # 👉 修正 1：移除 44 (發呆)，不准 AI 躺平
        allowed_acts = [1, 2, 11, 14, 16, 18, 34, 41, 42, 45]
        supply_surplus = float(player.food_cap) - float(player.food_used)
        minerals = float(player.minerals)
        vespene = float(player.vespene)
        
        # --- 套用防呆規則 ---
        if (supply_surplus >= 16 or depots >= 3) and 1 in allowed_acts: allowed_acts.remove(1)
        if float(player.food_workers) >= 22 and 14 in allowed_acts: allowed_acts.remove(14)
        
        # 👉 修正 2：兵營只要有 2 座就夠了，禁止再蓋，避免卡建築網格
        if barracks >= 2 and 2 in allowed_acts: allowed_acts.remove(2)
        if depots == 0 and 2 in allowed_acts: allowed_acts.remove(2)
            
        if barracks == 0:
            if 11 in allowed_acts: allowed_acts.remove(11)
            # 👉 移除 44 的連帶修改
            for act in [16, 18, 34]: 
                if act in allowed_acts: allowed_acts.remove(act)
                
        if techlabs == 0 and 18 in allowed_acts: allowed_acts.remove(18)
        if refineries >= 1 and 11 in allowed_acts: allowed_acts.remove(11)
        has_geyser = np.any(s_unit == 342) or np.any(s_unit == 341)
        if not has_geyser and 11 in allowed_acts: allowed_acts.remove(11)
        if refineries == 0 and 42 in allowed_acts: allowed_acts.remove(42)
        
        # 👉 修正 3：如果瓦斯已經大於 150，禁止再瘋狂派兵去採瓦斯 (截斷 Action 42 的無限迴圈)
        if vespene >= 150 and 42 in allowed_acts: allowed_acts.remove(42)
            
        if player.idle_worker_count == 0 and 41 in allowed_acts: allowed_acts.remove(41)
        
        if supply_surplus < 1:
            for act in [14, 16]: 
                if act in allowed_acts: allowed_acts.remove(act)
        if supply_surplus < 2 and 18 in allowed_acts: allowed_acts.remove(18)
        
        if minerals < 50:
            for act in [14, 16, 34]: 
                if act in allowed_acts: allowed_acts.remove(act)
        if minerals < 75 and 11 in allowed_acts: allowed_acts.remove(11)
        if minerals < 100:
            for act in [1, 18]: 
                if act in allowed_acts: allowed_acts.remove(act)
        if minerals < 150 and 2 in allowed_acts: allowed_acts.remove(2)
        
        if vespene < 25 and 18 in allowed_acts: allowed_acts.remove(18)
        if vespene < 50 and 34 in allowed_acts: allowed_acts.remove(34)
        
        return [VALID_ACTIONS.index(act) for act in allowed_acts]

    def train_model():
        seq_len = 8 # 🎬 關鍵設定：每次讓 LSTM 回想連續的 8 步
        batch_episodes_count = 16 # 每次隨機抽取 16 局遊戲來學習
        
        # 確保記憶庫裡有足夠的「對局」
        if len(memory) < batch_episodes_count:
            return
            
        half_batch = batch_episodes_count // 2
        
        # 1. 抽取整局遊戲 (Episodes)
        if len(success_memory) >= half_batch:
            batch_episodes = random.sample(memory, half_batch) + random.sample(success_memory, half_batch)
        else:
            batch_episodes = random.sample(memory, batch_episodes_count)
            
        # 打亂抽出來的對局順序
        random.shuffle(batch_episodes)
        
        optimizer.zero_grad()
        total_loss = 0
        valid_sequences = 0
        
        # 2. 對每一局遊戲抽出的「連續片段」進行學習
        for ep in batch_episodes:
            if len(ep) < seq_len:
                continue # 忽略太短的遊戲記憶
                
            valid_sequences += 1
            # 隨機挑選這局裡面的一段連續時間 (例如第 10 步 ~ 第 17 步)
            start_idx = random.randint(0, len(ep) - seq_len)
            sequence = ep[start_idx : start_idx + seq_len]
            
            # 拿出這段記憶「最一開始」的隱藏狀態，作為 LSTM 的回憶起點
            # sequence[0][5] 對應的是當時存入的 hidden_state (h_in, c_in)
            h_in, c_in = sequence[0][5] 
            hidden = (h_in.to(device), c_in.to(device))
            
            # 3. 順著時間線，一步一步推演這段記憶
            for step_data in sequence:
                # 將資料解包並轉為 Tensor (加上 unsqueeze(0) 模擬 batch_size=1)
                state = torch.FloatTensor([step_data[0]]).to(device)
                action = torch.LongTensor([step_data[1]]).to(device)
                reward = torch.FloatTensor([step_data[2]]).to(device)
                next_state = torch.FloatTensor([step_data[3]]).to(device)
                done = torch.FloatTensor([step_data[4]]).to(device)
                gamma_mult = torch.FloatTensor([step_data[7]]).to(device)
                next_allowed = step_data[8]  # 👈 拿出這一步的合法動作名單
                # 第一大腦 (Brain) 回想這一步，並產生「新的隱藏狀態」給下一步用
                q_actions, _, next_hidden = brain_model(state, hidden)
                q_value = q_actions.gather(1, action.unsqueeze(1)).squeeze(1)
                
                with torch.no_grad():
                    # 第一大腦預測下一步的最佳動作
                    next_q_actions_brain, _, _ = brain_model(next_state, next_hidden)
                    
                    mask = torch.full_like(next_q_actions_brain, float('-inf'))
                    mask[0, next_allowed] = 0
                    masked_next_q_brain = next_q_actions_brain + mask

                    # 👉 修正這行：從「套用過遮罩 (masked) 」的分數裡面挑選最大值！
                    # 原本是 next_q_actions_brain.max...，現在改成 masked_next_q_brain.max...
                    best_next_action = masked_next_q_brain.max(1)[1].unsqueeze(1)
                    
                    # 第二大腦 (Target) 針對該動作打客觀分數
                    next_q_actions_target, _, _ = target_model(next_state, next_hidden)
                    max_next_q = next_q_actions_target.gather(1, best_next_action).squeeze(1)
                    
                    # 計算最終期望值 (Bellman Equation)
                    target_q = reward + (gamma_mult * max_next_q * (1 - done))
                # 累加這一步的誤差
                loss = nn.SmoothL1Loss()(q_value, target_q) # ✨ 改成這個！
                total_loss += loss
                
                # ✨ 關鍵：將更新後的大腦記憶傳遞給時間線的「下一步」
                hidden = next_hidden
                
        # 4. 根據這批連續記憶的總誤差，更新神經網路
        if valid_sequences > 0 and isinstance(total_loss, torch.Tensor):
            # 取平均避免梯度爆炸
            mean_loss = total_loss / (valid_sequences * seq_len) 
            mean_loss.backward()
            optimizer.step()

    with sc2_env.SC2Env(
        map_name="Simple96",
        players=[sc2_env.Agent(sc2_env.Race.terran),sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy)], #sc2_env.Agent(sc2_env.Race.terran)
        agent_interface_format=sc2_env.AgentInterfaceFormat(
            feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64), use_raw_units=False),
        step_mul=16, realtime=False
        
    ) as env:
        
        current_session_file = f"d:/RL_Projects/dudu DRQN/log/dqn_training_log_{int(time.time())}.csv"
        
        for ep in range(10000):
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



            episode_reward = 0.0  
            step_reward = 0.0
            achieved_milestones = set()
            

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
                if agent.locked_action is not None:
                    # 讓底層代理人繼續完成他未完成的動作，傳入 0 (no_op) 代表 DQN 不給新指令
                    sc2_action = agent.get_action(obs, 0, parameter=1)
                    obs_list = env.step([sc2_action, actions.FUNCTIONS.no_op()])
                    obs = obs_list[0]
                    # 累積一點微小的時間扣分到全域，但不存入 DQN 記憶
                    total_reward -= 0.1 
                    continue
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
                
                # ==========================================
                # ✨ 升級版：科技樹動態動作遮罩 (Tech-Tree Masking)
                # ==========================================
                
                # 👉 呼叫我們寫好的神級防呆函數，直接取得 0~10 的合法索引！
                allowed_indices = get_action_mask(obs)

                
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
                next_allowed_indices = get_action_mask(next_obs)

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
                '''step_reward = 0.0
                
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

                    agent.last_target_count = current_real_count
                
                    # 💡 核心邏輯：計算剩餘時間比例 (開局 1.0 -> 結束 0.0)
                    # 總限時 13440 幀
                    time_ratio = 1.0 - (current_loop / 13440.0)
                
                    # 基礎分 500 + 時間紅利 (最高 500)
                    # 目的：誘導 AI 提早掛科技室、早採瓦斯，早產出的單位分拿更多
                    unit_reward = 500.0 + (time_ratio * 500.0)
                    step_reward += unit_reward
                    print(f"🎯 掠奪者第 {current_real_count} 隻產出！獲得時效獎勵: +{unit_reward:.0f}")

                # 6. 🏁 終局結算：規則永遠不變
                current_loop = int(next_obs.observation.game_loop[0])
                done = next_obs.last() or current_loop >= 13440 or current_real_count >= 5

                # ==========================================
                # ⚠️ 第一步：先結算遊戲終局的龐大分數 (加分或扣分)
                # ==========================================
                if done:
                    if current_real_count >= 5:
                        # ✅ 成功結算：保底 3000 分
                        step_reward += 3000.0
                        
                        # 💰 剩餘時間大獎金：最高可達 5000 分
                        # 公式：(剩餘時間百分比) * 5000
                        # 💡 目的：這是最強驅動力，逼迫 AI 壓縮所有建築時間，尋找最短路徑
                        time_bonus = (1.0 - current_loop / 13440.0) * 5000.0
                        step_reward += time_bonus
                        print(f"✅ 任務成功！花費 {current_loop} 幀，剩餘時間獎金: +{time_bonus:.0f}")
                    else:
                        # ❌ 失敗結算：引入「完成度補償」
                        # 💡 技巧：產越多隻扣越少 (例如產 4 隻只扣 360)
                        # 讓 AI 在學會產 5 隻前，先努力產出 1 隻來減少懲罰，維持學習動力
                        progress_ratio = current_real_count / 5.0
                        failure_penalty = -1000.0 + (progress_ratio * 800.0)
                        step_reward += failure_penalty
                        print(f"❌ 任務失敗：完成度 {current_real_count}/5，最終結算懲罰: {failure_penalty:.0f}")

                # ==========================================
                # ⚠️ 第二步：取得下一步狀態與壓縮分數
                # ==========================================
                scaled_reward = step_reward / 13000.0
                
                # 更新狀態向量，準備進入下一輪
                next_time_loop = int(next_obs.observation.game_loop[0])
                next_state = get_state_vector(
                    next_obs, 
                    getattr(agent, 'active_parameter', 1), 
                    18, 
                    a_id, 
                    current_action_success, 
                    next_time_loop
                )'''

                # 0. ✨ 初始化：每一步開始前歸零 (確保穩定收斂)
                step_reward = 0.0
                

                # 1.科技樹里程碑 (麵包屑導航：數值極小化，只給方向不給刷分空間)
                # 嚴厲邏輯：這些只是過程，不是目標。分數調低防止 AI 卡在「蓋完建築就發呆」。
                #邏輯：當畫面上出現補給站（Depot）、兵營（Barracks）、瓦斯廠（Refinery）]或科技室（Techlab）時，給予 1.0 ~ 3.0 的微量加分。
                if "depot" not in achieved_milestones and current_depots >= 1:
                    step_reward += 1.0; achieved_milestones.add("depot")
                if "barracks" not in achieved_milestones and current_barracks >= 1:
                    step_reward += 2.0; achieved_milestones.add("barracks")
                if "refinery" not in achieved_milestones and current_refineries >= 1:
                    step_reward += 1.5; achieved_milestones.add("refinery")
                if "techlab" not in achieved_milestones and current_techlabs >= 1:
                    step_reward += 3.0; achieved_milestones.add("techlab")

                # 2.核心產量獎勵 (掠奪者 Marauder)
                if current_real_count > getattr(agent, 'last_target_count', 0):
                    agent.last_target_count = current_real_count
                    # 設計理念：我們不使用原本的 1000 分，而是改用 10 分。
                    # 這能讓神經網路在更新權重時更穩定，不會因為一次巨大的脈衝訊號導致模型參數「飛掉」。
                    step_reward += 10.0
                    print(f"🎯 掠奪者產出！目前進度: {current_real_count}/5")

                # 3.終局判定：由「幀數」主導的勝負邏輯
                current_loop = int(next_obs.observation.game_loop[0])
                # 定義回合結束的三個條件：遊戲結束、時間耗盡（13440 幀）、或達成 5 隻目標。
                done = next_obs.last() or current_loop >= 13440 or current_real_count >= 5

                # ==========================================
                # ⚠️ 第一步：執行高壓的時間效率結算
                # ==========================================
                if done:
                    if current_real_count >= 5:
                        # 公式：(剩餘時間佔總時間的比例) * 50 分
                        time_bonus = (1.0 - current_loop / 13440.0) * 50.0
                        #time_bonus：這是一個線性獎勵
                        # 如果 AI 在第 1 幀就完成，獎金接近 50；如果等到最後 1 幀才完成，獎金接近 0。
                        #意義：強制 AI 優化路徑。AI 會發現發呆是很昂貴的，「省下的時間 = 賺到的分數」。
                        step_reward += (50.0 + time_bonus)
                        print(f"✅ 任務成功！剩餘時間獎金: +{time_bonus:.1f}")
                    else:
                        # 失敗結算：嚴厲倒扣，不給任何安慰分。
                        # 嚴厲邏輯：沒產滿就是失敗，強制 AI 去探索「成功」的那條路。
                        step_reward -= 20.0
                        print(f"❌ 任務失敗：效率不足或時間耗盡，最終懲罰 -20")

                # ==========================================
                # ⚠️ 第二步：分數壓縮 (Normalization)
                # ==========================================
                # 💡 總分上限現在約為 100~110 (科技加分 + 5隻產量 + 成功獎金 + 時間獎金)
                # 除以 160 可以讓 reward 完美落在1~-1附近，這是 DQN 最喜歡的區間。
                scaled_reward = step_reward / 160.0
                episode_reward += scaled_reward

                # 🚀 加入這行測試！
                if step_reward != 0:
                    print(f"DEBUG: 這一幀加了 {step_reward} 分，累計總分: {episode_reward}")

                next_time_loop = int(next_obs.observation.game_loop[0])
                next_state = get_state_vector(
                    next_obs, 
                    getattr(agent, 'active_parameter', 1), 
                    18, 
                    a_id, 
                    current_action_success, 
                    next_time_loop
                )
                current_transition = (state, action_index, scaled_reward, next_state, done, hidden_state, next_hidden_state, next_allowed_indices)
                n_step_buffer.append(current_transition)
                


                obs = next_obs
                state = next_state
                hidden_state = next_hidden_state
                last_action_id = a_id
                last_action_success = current_action_success



                # ==========================================
                # ⚠️ 第三步：N-Step Bootstrapping 緩衝與寫入
                # ==========================================
                N_STEP = 7
                GAMMA = 0.99
                
                # 1. 先把當前這一步丟進緩衝區
                current_transition = (state, action_index, scaled_reward, next_state, done, hidden_state, next_hidden_state, next_allowed_indices)
                n_step_buffer.append(current_transition)

                # 2. 如果視窗滿了，結算過去 N 步的總報酬，並存入大腦
                if len(n_step_buffer) == N_STEP:
                    n_reward = sum([n_step_buffer[i][2] * (GAMMA ** i) for i in range(N_STEP)])
                    
                    n_state = n_step_buffer[0][0]
                    n_action = n_step_buffer[0][1]
                    n_next_state = n_step_buffer[-1][3]
                    n_done = n_step_buffer[-1][4]
                    n_hidden_in = n_step_buffer[0][5]
                    n_hidden_out = n_step_buffer[-1][6]
                    n_next_allowed = n_step_buffer[-1][7]

                    # ✨ 打包成新的 N-Step 包裹
                    n_transition = (n_state, n_action, n_reward, n_next_state, n_done, n_hidden_in, n_hidden_out, GAMMA ** N_STEP, n_next_allowed)
                    #agent.store_transition(*n_transition)
                    # 👉 加上這行！把這一刻的記憶確實寫入整局的歷史中
                    episode_memory.append(n_transition)
                    
                    # 滑動視窗：移除最舊的一步，讓下一步進來
                    n_step_buffer.popleft() 

                # 3. 遊戲結束時，強制清空並結算緩衝區裡剩下的尾巴
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
                        
                        # 👉 新增這行：拿出最後一步的合法名單
                        n_next_allowed = n_step_buffer[-1][7]
                        
                        # 👉 修改這行：把 n_next_allowed 打包進去 (變成 9 個參數，與上方統一)
                        n_transition = (n_state, n_action, n_reward, n_next_state, n_done, n_hidden_in, n_hidden_out, GAMMA ** actual_n, n_next_allowed)
                        
                        episode_memory.append(n_transition)
                        n_step_buffer.popleft()
               
                if done:
                    # 👉 新增這行：把這整局的連續記憶打包存進一般記憶庫
                    memory.append(episode_memory) 

                    if current_real_count >= 5:
                        # 👉 修改這行：將 extend 改為 append，確保它存入的是一個完整的 List
                        success_memory.append(episode_memory) 
                        print(f"🌟 黃金記憶寫入！已將長度為 {len(episode_memory)} 步的完整經驗刻入大腦深層！")

                # ==========================================
                # ⚠️ 第五步：觸發訓練與更新迴圈變數
                # ==========================================
                if len(memory) >= 10:
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
                    print(f"!!! 即將寫入 CSV 的分數是: {episode_reward} (型別: {type(episode_reward)})")
                    #logger.log_episode(ep + 1, "掠奪者任務", epsilon, float(episode_reward), current_loop, current_real_count, marine_count)
                    #log_file = f"d:/RL_Projects/dudu DRQN/log/dqn_training_log_{int(time.time())}.csv"
                    with open(current_session_file, "a", encoding="utf-8") as f:
                        line = f"{ep+1},掠奪者任務,{epsilon:.3f},{episode_reward:.4f},{current_loop},{current_real_count},{marine_count}\n"
                        f.write(line)
                    print(f"✅ 已手動強制寫入 CSV: {episode_reward:.4f}")
                    break # ✨ 記得加上 break 跳出 while 迴圈

                state = next_state
                hidden_state = next_hidden_state
                
            
            # 回合結束後更新 epsilon
            
            epsilon = max(learn_min, epsilon * epsilon_decay)#
            
            # ✨ 4. 大腦同步機制：每隔 5 局，讓第二大腦吸收第一大腦的最新知識！
            if ep % 5 == 0:
                target_model.load_state_dict(brain_model.state_dict())
                print(f"🔄 雙大腦同步完成 (Target Model Updated)！")
            torch.save(brain_model.state_dict(), model_path)

if __name__ == "__main__":
    from absl import app
    app.run(main)