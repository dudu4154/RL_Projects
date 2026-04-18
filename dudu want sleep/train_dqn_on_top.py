import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import time
from collections import deque
from pysc2.env import sc2_env
from pysc2.lib import actions, features
import pickle
import threading # 加上這行
import copy
import pygame
import numpy as np

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
            writer.writerow([ep, task_name, f"{eps:.3f}", f"{reward:.4f}", end_loop, marauders_cnt, marines_cnt])
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
        min(float(current_loop) / 13440, 1.0)        # 17. 遊戲時間進度
    ]
    return state_list
def get_action_mask(obs, agent):
    """
    根據當前資源與狀態，過濾出目前「合法可執行」的動作 ID 索引。
    這能防止 AI 試圖在沒錢的時候蓋建築，減少無效的學習。
    """
    player = obs.observation.player
    allowed_indices = []
    
    # 這是你主程式中定義的有效動作清單
    VALID_ACTIONS = [1, 2, 11, 14, 16, 18, 34, 41, 42, 44, 45, 46, 47, 48, 50]
    
    for i, a_id in enumerate(VALID_ACTIONS):
        is_allowed = False
        
        # Action 1: 蓋補給站 (100 礦)
        if a_id == 1:
            if player.minerals >= 100: is_allowed = True
                
        # Action 2: 蓋兵營 (150 礦)
        elif a_id == 2:
            if player.minerals >= 150: is_allowed = True
                
        # Action 11: 蓋瓦斯廠 (75 礦)
        elif a_id == 11:
            if player.minerals >= 75: is_allowed = True
                
        # Action 14: 造 SCV (50 礦)
        elif a_id == 14:
            if player.minerals >= 50: is_allowed = True
                
        # Action 16: 造陸戰隊 (50 礦)
        elif a_id == 16:
            if player.minerals >= 50: is_allowed = True
                
        # Action 18: 造掠奪者 (100 礦, 25 瓦斯)
        elif a_id == 18:
            if player.minerals >= 100 and player.vespene >= 25: is_allowed = True
                
        # Action 34: 蓋兵營科技實驗室 (50 礦, 25 瓦斯)
        elif a_id == 34:
            if player.minerals >= 50 and player.vespene >= 25: is_allowed = True
                
        # 其他不需要資源條件的動作 (採礦、發呆、編隊、視角切換等) 預設允許執行
        elif a_id in [41, 42, 44, 45, 46, 47, 48, 50]:
            is_allowed = True

        # 如果資源條件符合，把該動作的「索引值」加進名單
        if is_allowed:
            allowed_indices.append(i)
            
    # 絕對防呆機制：如果莫名其妙什麼動作都不能做，強制允許「發呆(44)」
    if not allowed_indices:
        if 44 in VALID_ACTIONS:
            allowed_indices.append(VALID_ACTIONS.index(44))
        else:
            allowed_indices.append(0)

    return allowed_indices


# =========================================================
# 🎮 訓練主程式
# =========================================================
def main(argv):
    del argv
    RENDER_UI = True
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("DRQN 決策中樞可視化")
    clock = pygame.time.Clock()
    state_size = 17  
    IS_TRAINING = True
    # 確保清單包含生產、戰鬥與切換(50)
    VALID_ACTIONS = [1, 2, 11, 14, 16, 18, 34, 41, 42, 44, 45, 46, 47, 48, 50]
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

    # 宣告排行榜路徑
    elite_memory_path = os.path.join(log_dir, "elite_memory.pkl")
    
    # 嘗試載入舊的排行榜
    if os.path.exists(elite_memory_path):
        with open(elite_memory_path, "rb") as f:
            success_memory = pickle.load(f)
        print(f"📖 成功載入歷史菁英排行榜！目前收錄 {len(success_memory)} 局神操作。")
    else:
        success_memory = []
        
    MAX_ELITE_MEMORY = 1000

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

    epsilon = 1.00; epsilon_decay = 0.998; gamma = 0.998

    
    def draw_dual_head_network(surface, model, state_vector, hidden_state=None):
        import pygame
        import numpy as np
        import torch

        # ==========================================================
        # 1. 畫布基礎設定：乾淨的白底與字體
        # ==========================================================
        surface.fill((250, 250, 250)) 
        try:
            font = pygame.font.SysFont("arial", 13, bold=True)
            num_font = pygame.font.SysFont("arial", 10)
        except:
            font = pygame.font.SysFont("arial", 13)
            num_font = pygame.font.SysFont("arial", 10)

        # 標籤定義 (嚴格對應 17維輸入 與 15種動作)
        input_labels = ["Workers", "Minerals", "Vespene", "Supply", "Depot", "Gas", 
                        "Barracks", "TechLab", "SCV Sel", "CC Sel", "Rax Sel", 
                        "Marine", "Marauder", "Task ID", "Block", "Last OK", "Time"]
        action_labels = ["Depot", "Barrack", "Refinery", "Train SCV", "Marine", 
                        "Marauder", "TechLab", "Mine", "Gas", "Wait", 
                        "CC Ctrl", "Army Ctrl", "A-move", "Focus", "Switch"]

        # ==========================================================
        # 2. 獲取當前決策狀態
        # ==========================================================
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(np.array(state_vector)).unsqueeze(0).to(device)
            if hidden_state is None:
                hidden_state = (torch.zeros(1, 1, 128).to(device), torch.zeros(1, 1, 128).to(device))
            
            q_actions, _, _ = model(state_t, hidden_state)
            q_vals = q_actions.cpu().numpy()[0]
            # 找出當前 AI 最想做的動作
            best_action_idx = int(np.argmax(q_vals))

        # ==========================================================
        # 3. 座標佈局 (恢復為乾淨的直線排列)
        # ==========================================================
        in_pos = [(180, y * 32 + 60) for y in range(17)]
        hid_pos = [(400, y * 34 + 80) for y in range(16)] # 👉 恢復垂直置中對齊
        act_pos = [(620, y * 30 + 50) for y in range(15)]

        # ==========================================================
        # 4. 提取權重矩陣
        # ==========================================================
        fc1_w = model.fc1.weight.data.cpu().numpy()       # [128, 17]
        act_w = model.fc_action.weight.data.cpu().numpy() # [15, 128]

        # ==========================================================
        # 🌟 5. 繪製連線 (Top-K 強制稀疏化實線)
        # ==========================================================

        # A. 輸入 -> 隱藏層
        for h_idx in range(16):
            actual_h = h_idx * 8
            weights = fc1_w[actual_h, :]
            # ✨ 找出對這個隱藏節點影響力最大的 2 個輸入特徵
            top_2_inputs = np.argsort(np.abs(weights))[-2:]
            
            for i in range(17):
                w = weights[i]
                if i in top_2_inputs and abs(w) > 0.1:
                    color = (0, 0, 220) if w > 0 else (220, 0, 0)
                    
                    # ✨ 動態粗細：活躍特徵線條變粗
                    active_multiplier = 2.0 if state_vector[i] > 0.1 else 0.5
                    width = max(1, int(abs(w) * 4 * active_multiplier))
                    
                    pygame.draw.line(surface, color, in_pos[i], hid_pos[h_idx], width)

        # B. 隱藏層 -> 動作層
        for a_idx in range(15):
            weights = act_w[a_idx, :]
            sampled_weights = [weights[h * 8] for h in range(16)]
            # ✨ 找出對這個動作影響力最大的 2 個隱藏節點
            top_2_hidden = np.argsort(np.abs(sampled_weights))[-2:]
            
            for h_idx in range(16):
                actual_h = h_idx * 8
                w = act_w[a_idx, actual_h]
                if h_idx in top_2_hidden and abs(w) > 0.1:
                    color = (0, 0, 220) if w > 0 else (220, 0, 0)
                    
                    # 若是當前選擇的動作，加粗該路徑
                    is_active_path = (a_idx == best_action_idx)
                    width = max(1, int(abs(w) * (6 if is_active_path else 2)))
                    
                    pygame.draw.line(surface, color, hid_pos[h_idx], act_pos[a_idx], width)

        # ==========================================================
        # 🎨 6. 節點渲染
        # ==========================================================
        def draw_neat_node(pos, label, num, side="left", active_val=0, is_best_action=False):
            fill_color = (255, 255, 255)
            if active_val > 0.1:
                fill_color = (200, 255, 200) # 活躍亮淺綠
            if is_best_action:
                fill_color = (180, 180, 255) # 決策亮淺藍

            pygame.draw.circle(surface, fill_color, pos, 13)
            pygame.draw.circle(surface, (50, 50, 50), pos, 13, 2)
            
            n_txt = num_font.render(str(num), True, (0, 0, 0))
            surface.blit(n_txt, (pos[0]-n_txt.get_width()//2, pos[1]-n_txt.get_height()//2))
            
            if label:
                l_txt = font.render(label, True, (0, 0, 0))
                x_off = -115 if side == "left" else 25
                surface.blit(l_txt, (pos[0] + x_off, pos[1] - 8))

        for i, pos in enumerate(in_pos):
            draw_neat_node(pos, input_labels[i], i, "left", state_vector[i])
        for i, pos in enumerate(hid_pos):
            draw_neat_node(pos, None, i + 100)
        for i, pos in enumerate(act_pos):
            draw_neat_node(pos, action_labels[i], i, "right", 0, is_best_action=(i == best_action_idx))
    def train_model():
        seq_len = 8 # 🎬 關鍵設定：每次讓 LSTM 回想連續的 8 步
        batch_episodes_count = 16 # 每次隨機抽取 16 局遊戲來學習
        
        # 確保記憶庫裡有足夠的「對局」
        if len(memory) < batch_episodes_count:
            return
            
        half_batch = batch_episodes_count // 2
        
        # 1. 抽取整局遊戲 (Episodes)
        if len(success_memory) >= half_batch:
            # ✨ 從排行榜中隨機抽出幾局，並只提取 episode_memory (索引 1)
            elite_samples = [ep for _, ep in random.sample(success_memory, half_batch)]
            
            # 把近期記憶跟菁英記憶混合
            batch_episodes = random.sample(memory, half_batch) + elite_samples
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
                    # 1. 第一大腦 (Brain) 預測下一步的動作 (結合動作遮罩)
                    next_q_actions_brain, _, _ = brain_model(next_state, next_hidden)
                    mask = torch.full_like(next_q_actions_brain, float('-inf'))
                    mask[0, next_allowed] = 0
                    masked_next_q_brain = next_q_actions_brain + mask
                    # 選出下一步「最想做」的動作 ID
                    best_next_action = masked_next_q_brain.max(1)[1].unsqueeze(1)
                    
                    # 2. 取出 N-Step 結尾時真正的隱藏記憶 (對應 S_{t+N})
                    h_out, c_out = step_data[6] 
                    target_hidden = (h_out.to(device), c_out.to(device))
                    
                    # 3. 第二大腦 (Target) 負責評分：算出該動作在未來的 Q 值
                    next_q_actions_target, _, _ = target_model(next_state, target_hidden)
                    
                    # ✨ 修正一：利用第一大腦選出的動作，去拿第二大腦的分數 (這才是正宗 DDQN)
                    next_q = next_q_actions_target.gather(1, best_next_action).squeeze(1)
                    
                    # ✨ 修正二：計算貝爾曼方程式算出目標 target_q
                    # 注意：這裡的 gamma_mult 已經是你前面寫的 GAMMA ** N_STEP 了，非常精準！
                    target_q = reward + gamma_mult * next_q * (1 - done)

                # 4. 累加這一步的誤差 (加上 .detach() 確保梯度不會亂跑)
                loss = nn.SmoothL1Loss()(q_value, target_q.detach())
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
        
        current_session_file = os.path.join(log_dir, f"dqn_training_log_{int(time.time())}.csv")
        
        for ep in range(10000):

            agent = ProductionAI()
            obs_list = env.reset()
            obs = obs_list[0]
            
            episode_memory = []  
            # ✨ 1. N-Step 新增：準備一個長度為 3 的滑動視窗
            n_step_buffer = deque(maxlen=7) 
            
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
            total_reward = 0.0
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
            
            achieved_milestones = set()
            

            while True:
                step_reward = 0.0
                pending_reward = 0.0
                
                
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
                # 🚨 鐵心協定強制介入：5 隻出籠，立刻編隊！
                # ==========================================
                try:
                    group_2_count = obs.observation.control_groups[2][1]
                except:
                    group_2_count = 0
                    
                m_unit = obs.observation.feature_minimap.unit_type
                m_relative = obs.observation.feature_minimap.player_relative
                total_marauders_map = np.sum((m_unit == 51) & (m_relative == 1))
                
                force_action = None
                
                # 條件：地圖上有 5 隻掠奪者，且編隊 2 是空的
                if total_marauders_map >= 5 and group_2_count == 0:
                    force_action = 46
                    agent.is_combat_mode = True # 強制將大腦切換至戰鬥模式

                # 取得當下模式的合法清單
                allowed_indices = get_action_mask(obs, agent)

                # ==========================================
                # 🎯 動作選擇 (乾淨無重疊版)
                # ==========================================
                if force_action is not None:
                    # 強制執行 46 (無視神經網路的隨機)
                    a_id = force_action
                    action_index = VALID_ACTIONS.index(a_id)
                    p_id = random.randint(1, 64)
                    print("⚠️ [系統介入] 偵測到 5 隻掠奪者閒置，強制執行 Action 46 (編隊)！")
                    with torch.no_grad():
                        _, _, next_hidden_state = brain_model(state_t.unsqueeze(0), hidden_state)
                else:
                    # 一般 DQN 決策
                    if random.random() <= epsilon:
                        action_index = random.choice(allowed_indices)
                        p_id = random.randint(1, 64)
                        with torch.no_grad():
                            _, _, next_hidden_state = brain_model(state_t.unsqueeze(0), hidden_state)
                    else:
                        with torch.no_grad():
                            q_actions, q_params, next_hidden_state = brain_model(state_t.unsqueeze(0), hidden_state)
                            mask = torch.full_like(q_actions, float('-inf'))
                            mask[0, allowed_indices] = 0
                            masked_q_actions = q_actions + mask
                            action_index = masked_q_actions.argmax().item()
                            p_id = q_params.argmax().item() + 1
                            
                    a_id = VALID_ACTIONS[action_index]
                    
                # ==========================================
                # ==========================================
                # ==========================================
                # ✨ 核心防彈機制：在一開始就先給定預設發呆動作！
                sc2_action = actions.FUNCTIONS.no_op() 
                is_decision_frame = (agent.locked_action is None) or (agent.lock_timer <= 1)

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
                next_allowed_indices = get_action_mask(next_obs, agent)
                if RENDER_UI and train_step_counter % 2 == 0:  
                    screen.fill((255, 255, 255)) # 建議改成白底 (255, 255, 255) 配合 Dino 風格
                    draw_dual_head_network(screen, brain_model, state, hidden_state)
                    pygame.display.flip()
                    
                # 處理視窗事件防止當機
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        torch.save(brain_model.state_dict(), model_path)
                        return
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
                
                
                

                # 1.科技樹里程碑 (麵包屑導航：數值極小化，只給方向不給刷分空間)
                # 嚴厲邏輯：這些只是過程，不是目標。分數調低防止 AI 卡在「蓋完建築就發呆」。
                #邏輯：當畫面上出現補給站（Depot）、兵營（Barracks）、瓦斯廠（Refinery）]或科技室（Techlab）時，給予 1.0 ~ 3.0 的微量加分。
                if "depot" not in achieved_milestones and current_depots >= 1:
                    step_reward += 1.0; achieved_milestones.add("depot")
                    #print(1)
                if "barracks" not in achieved_milestones and current_barracks >= 1:
                    step_reward += 2.0; achieved_milestones.add("barracks")
                    #print(2)
                if "refinery" not in achieved_milestones and current_refineries >= 1:
                    step_reward += 1.5; achieved_milestones.add("refinery")
                    #print(3)
                if "techlab" not in achieved_milestones and current_techlabs >= 1:
                    step_reward += 3.0; achieved_milestones.add("techlab")
                    #print(4)


                # 2.核心產量獎勵 (掠奪者 Marauder)
                if current_real_count > getattr(agent, 'last_target_count', 0):
                    agent.last_target_count = current_real_count
                    # 設計理念：我們不使用原本的 1000 分，而是改用 10 分。
                    # 這能讓神經網路在更新權重時更穩定，不會因為一次巨大的脈衝訊號導致模型參數「飛掉」。
                    step_reward += 10.0
                    print(f"🎯 掠奪者產出！目前進度: {current_real_count}/5")

                MAX_GAME_LOOP = 13440 # 👉 20 分鐘上限
                current_loop = int(next_obs.observation.game_loop[0])

                # ==========================================
                # 🥊 新增：TKO (技術性擊倒) 判定機制
                # ==========================================
                # PySC2 的 score_cumulative 陣列中，索引 5 是殺敵總值，索引 6 是摧毀建築總值
                killed_units_score = int(next_obs.observation.score_cumulative[5])
                killed_structures_score = int(next_obs.observation.score_cumulative[6])

                # ✨ 如果摧毀了任何敵方建築，或是殺死了夠多敵人 (例如價值 > 500)，就代表對面已經崩潰投降了！
                is_tko_victory = (killed_structures_score > 0) or (killed_units_score > 500)

                if is_tko_victory and "tko_triggered" not in achieved_milestones:
                    achieved_milestones.add("tko_triggered")
                    print(f"🥊 [TKO 擊倒對手！] 摧毀建築得分: {killed_structures_score}, 殺敵得分: {killed_units_score}")

                # 🌟 修正結束條件：加入 is_tko_victory，只要打爆對面就提早收工！
                done = next_obs.last() or current_loop >= MAX_GAME_LOOP or is_tko_victory

                # ==========================================
                # 🏆 實戰里程碑結算 (取代原本的終局結算)
                # ==========================================
                # 只有當「第一次」達到 5 隻時，才給予唯一的一次巨大獎勵！
                if "first_5_marauders" not in achieved_milestones and current_real_count >= 5:
                    achieved_milestones.add("first_5_marauders")
                    
                    # ✨ 核心分數公式：(1 - 消耗時間比例) * 100
                    # 越快完成，(current_loop / MAX_GAME_LOOP) 越小，扣掉的就越少，總分越接近 100 分。
                    # 如果壓線在 20 分鐘才完成，分數就會趨近於 0 分。
                    time_score = 100.0 * (1.0 - (current_loop / MAX_GAME_LOOP))
                    step_reward += time_score
                    print(f"🎉 [任務達成] 湊齊 5 隻掠奪者！花費: {current_loop} 幀，獲得時間分數: +{time_score:.2f}")

                if done:
                    if "first_5_marauders" not in achieved_milestones:
                        # 時間耗盡仍未湊齊 5 隻，給予明確的失敗懲罰
                        step_reward -= 20.0
                        print(f"❌ 任務失敗：20 分鐘耗盡仍未湊齊，最終懲罰 -20.0")

                pending_reward += step_reward
                # ==========================================
                # ⚠️ 第二步：分數壓縮 (Normalization)
                # ==========================================
                if is_decision_frame:
                    # 把這段期間累積的獎勵一次性結算給這個決策
                    scaled_reward = pending_reward / 100.0
                    episode_reward += scaled_reward
                    
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
                    pending_reward = 0.0
                    state = next_state
                    obs = next_obs
                    hidden_state = (next_hidden_state[0].detach(), next_hidden_state[1].detach())
                else:
                    pass

                last_action_id = a_id
                last_action_success = current_action_success



                # ==========================================
                # ⚠️ 第三步：N-Step Bootstrapping 緩衝與寫入
                # ==========================================
                N_STEP = 7
                GAMMA = 0.99

                # 2. 如果視窗滿了，結算過去 N 步的總報酬，並存入大腦
                if len(n_step_buffer) == N_STEP:
                    n_reward = sum([n_step_buffer[i][2] * (GAMMA ** i) for i in range(N_STEP)])
                    
                    n_state = n_step_buffer[0][0]
                    n_action = n_step_buffer[0][1]
                    n_next_state = n_step_buffer[-1][3]
                    n_done = n_step_buffer[-1][4]
                    # 修正後：
                    n_hidden_in = (n_step_buffer[0][5][0].cpu(), n_step_buffer[0][5][1].cpu())
                    n_hidden_out = (n_step_buffer[-1][6][0].cpu(), n_step_buffer[-1][6][1].cpu())
                    n_next_allowed = n_step_buffer[-1][7]

                    # ✨ 打包成新的 N-Step 包裹
                    n_transition = (n_state, n_action, n_reward, n_next_state, n_done, n_hidden_in, n_hidden_out, GAMMA ** N_STEP, n_next_allowed)
                    #agent.store_transition(*n_transition)
                    # 👉 加上這行！把這一刻的記憶確實寫入整局的歷史中
                    episode_memory.append(n_transition)
                    
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
                        n_hidden_in = (n_step_buffer[0][5][0].cpu(), n_step_buffer[0][5][1].cpu())
                        n_hidden_out = (n_step_buffer[-1][6][0].cpu(), n_step_buffer[-1][6][1].cpu())
                        
                        # 👉 新增這行：拿出最後一步的合法名單
                        n_next_allowed = n_step_buffer[-1][7]
                        
                        # 👉 修改這行：把 n_next_allowed 打包進去 (變成 9 個參數，與上方統一)
                        n_transition = (n_state, n_action, n_reward, n_next_state, n_done, n_hidden_in, n_hidden_out, GAMMA ** actual_n, n_next_allowed)
                        
                        episode_memory.append(n_transition)
                        n_step_buffer.popleft()
               
                if done:
                    # 👉 新增這行：把這整局的連續記憶打包存進一般記憶庫
                    memory.append(episode_memory) 

                    # ✨ 核心修改：判斷遊戲是否取得「勝利」或是「TKO 擊倒」
                    if next_obs.reward == 1 or is_tko_victory:
                        
                        # 如果是我們手動判定的 TKO，我們直接補上勝利的 1.0 分！
                        if is_tko_victory:
                            total_reward += 1.0 

                        success_memory.append((total_reward, episode_memory))
                        # 依照總分 (時間獎勵) 從高到低排序，保留最速通關的紀錄
                        success_memory.sort(key=lambda x: x[0], reverse=True)
                        
                        if len(success_memory) > MAX_ELITE_MEMORY:
                            success_memory.pop()
                            
                        print(f"🏆 菁英榜更新！目前收錄 {len(success_memory)} 局，歷史最高分: {success_memory[0][0]:.1f}")
                        
                        # ✨ 修改：不要每局存，排行榜數量是 5 的倍數時才寫入硬碟
                        if len(success_memory) % 5 == 0:
    
                            # 建立一個專屬的背景存檔函數
                            def background_save(data_to_save, path):
                                try:
                                    with open(path, "wb") as f:
                                        pickle.dump(data_to_save, f)
                                    print(f"💾 [背景任務] 菁英榜已成功寫入硬碟！")
                                except Exception as e:
                                    print(f"⚠️ [背景任務] 存檔失敗: {e}")

                            # 複製一份當下的記憶（淺層拷貝），避免背景存檔到一半被主執行緒修改
                            data_copy = copy.copy(success_memory)
                            
                            # 呼叫多執行緒，把存檔工作丟到背景，主程式立刻繼續往下跑！
                            threading.Thread(target=background_save, args=(data_copy, elite_memory_path)).start()

                # ==========================================
                # ⚠️ 第五步：觸發訓練與更新迴圈變數
                # ==========================================
                # 只有在 IS_TRAINING 為 True 時才執行訓練 [cite: 107]
                if IS_TRAINING and len(memory) >= 10 and train_step_counter % 8 == 0:
                    train_model()

                hidden_state = (next_hidden_state[0].detach(), next_hidden_state[1].detach())
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
                    # 如果是 TKO，在任務名稱加上標記，讓你知道它是提早打爆對手收工的
                    final_task_name = "掠奪者任務 (TKO!)" if is_tko_victory else "掠奪者任務"
                    
                    print(f"!!! 即將寫入 CSV 的分數是: {episode_reward} (型別: {type(episode_reward)})")
                    logger.log_episode(ep + 1, final_task_name, epsilon, float(episode_reward), current_loop, current_real_count, marine_count)
                    print(f"✅ 已手動強制寫入 CSV: {episode_reward:.4f}")
                    break # ✨ 記得加上 break 跳出 while 迴圈

                state = next_state
                hidden_state = (next_hidden_state[0].detach(), next_hidden_state[1].detach())
                
            
            # 回合結束後更新 epsilon
            
            # 回合結束後更新 epsilon
            if IS_TRAINING:
                epsilon = max(learn_min, epsilon * epsilon_decay) 
           
                # ✨ 4. 大腦同步機制：每隔 5 局，讓第二大腦吸收第一大腦的最新知識！
                if ep % 5 == 0:
                    target_model.load_state_dict(brain_model.state_dict()) 
                    print(f"🔄 雙大腦同步完成 (Target Model Updated)！")
                
                # 存檔也建議只在訓練模式下執行，避免測試時誤蓋掉好的模型
                torch.save(brain_model.state_dict(), model_path) 
            else:
                epsilon = max(learn_min, epsilon * epsilon_decay) 
            

if __name__ == "__main__":
    from absl import app
    app.run(main)