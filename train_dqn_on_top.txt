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
import pickle
import threading # 加上這行
import copy
import math
import pygame

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
# 🐒 路徑設定git
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(current_dir, "log")

def patched_data_collector_init(self):
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    self.filename = os.path.join(log_dir, f"terran_log_{int(time.time())}.csv")
    with open(self.filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 👉 標題最後加上 "P_ID"
        writer.writerow(["Game_Loop", "Minerals", "Vespene", "Workers", "Ideal", "Barracks", "Action_ID", "P_ID"])

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
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.fc_action = nn.Linear(128, action_size)
        
        # ✨ 擴充第二大腦：從 64 改成 100 個神經元，對應 10x10 網格
        self.fc_param = nn.Linear(128, 100)

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
    

    # 資源正規化（修正分母，讓 AI 對資源變化更敏感）
    # 原本分母 2000/1000 太大，實際遊戲中礦幾乎不超過 400，瓦斯不超過 200
    # 導致 AI 感受到的數值永遠在 0.0~0.1 之間，學習效果很差
    mineral_ratio = min(float(player.minerals) / 400.0, 1.0)
    vespene_ratio = min(float(player.vespene) / 200.0, 1.0)
    action_ratio = min(float(last_action_id) / 50.0, 1.0) # 假設動作 ID 最大約 50
    block_ratio = min(float(current_block) / 100.0, 1.0)   # 區塊範圍是 1~64


    state_list = [
        min(float(player.food_workers) / 22.0, 1.0),                      # 1. 工兵總數
        min(float(player.idle_worker_count) / 10.0, 1.0),                 # 2. 閒置工兵
        mineral_ratio,                                                    # 3. 晶礦比例
        vespene_ratio,                                                    # 4. 瓦斯比例
        mineral_ratio - vespene_ratio,                                    # 5. 資源差距
        min(float(player.food_used) / 200.0, 1.0),                        # 6. 目前人口
        min(float(player.food_cap - player.food_used) / 20.0, 1.0),       # 7. 剩餘人口空間
        min(float(count_unit(19)) / 3.0, 1.0),                            # 8. 補給站數量
        min(float(count_unit(20)) / 2.0, 1.0),                            # 9. 瓦斯廠數量
        min(float(count_unit(21)) / 2.0, 1.0),                            # 10. 軍營數量
        min(float(count_unit(37)) / 1.0, 1.0),                            # 11. 科技室數量
        min(float(is_scv_selected), 1.0),                                 # 12. 選中工兵
        min(float(is_cc_selected), 1.0),                                  # 13. 選中主堡
        min(float(is_barracks_selected), 1.0),                            # 14. 選中兵營
        min(float(count_unit(48)) / 50.0, 1.0),                           # 15. 陸戰隊數量
        min(float(count_unit(51)) / 5.0, 1.0),                            # 16. 掠奪者數量
        min(float(last_action_success), 1.0),                             # 17. 上一動作成功與否
        action_ratio,                                                     # 🌟 18. 上一動作 ID
        block_ratio,                                                      # 🌟 19. 上一執行區塊
        min(float(current_loop) / 13440.0, 1.0)                           # 20. 遊戲時間進度
    ]

    return state_list



# =========================================================
# 🎮 訓練主程式
# =========================================================
def main(argv):
    
    del argv
    
    state_size = 20
    IS_TRAINING = True
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

    RENDER_UI = True # 總開關
    pygame.init()
    screen = pygame.display.set_mode((1200, 700))
    pygame.display.set_caption("DRQN 決策中樞 - 中文化遮罩版")
    
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

    epsilon = 1.00; epsilon_decay = 0.995; gamma = 0.998

    def draw_dual_head_network(surface, model, state_vector, allowed_indices, hidden_state=None):
        import numpy as np
        import torch
        import pygame

        # ==========================================
        # ✨ 畫質革命：SSAA 超採樣反鋸齒設定
        # ==========================================
        SCALE = 2  
        sw, sh = surface.get_size()
        hq_surface = pygame.Surface((sw * SCALE, sh * SCALE))
        
        # 🎨 1. 賽博龐克背景色：極深的深藍/黑色
        hq_surface.fill((10, 12, 18)) 
        
        # 嘗試載入支援中文的字體
        font_candidates = ['microsoftjhenghei', 'microsoftyahei', 'simhei', 'msgothic', 'arialunicode']
        font = None
        for f_name in font_candidates:
            try:
                test_font = pygame.font.SysFont(f_name, 13 * SCALE, bold=True)
                font = test_font
                break
            except:
                continue
        if font is None:
            font = pygame.font.SysFont("arial", 13 * SCALE)
        num_font = pygame.font.SysFont("arial", 10 * SCALE, bold=True) # 數字加粗更醒目

        # 🏷️ 變數名稱標準化
        input_labels = [
            "工兵總數 (22)", "閒置工兵 (10)", "晶礦比例 (400)", "瓦斯比例 (200)", 
            "資源差距 (M-V)", "目前人口 (200)", "剩餘人口 (20)", "補給站數 (3)", 
            "瓦斯廠數 (2)", "軍營數量 (2)", "科技室數 (1)", "選中工兵 (0/1)", 
            "選中主堡 (0/1)", "選中兵營 (0/1)", "陸戰隊數 (50)", "掠奪者數 (5)", 
            "上一動作成功", "上一動作ID(50)", "上一次區塊(64)", "時間進度 (13k)" 
        ]
        
        action_labels = [
            "蓋補給站", "蓋兵營", "蓋瓦斯廠", "製造工兵", "造陸戰隊", 
            "造掠奪者", "兵營升級", "閒置採礦", "指派瓦斯", "發呆休息", "主堡編隊"
        ]

        # 2. 獲取當前決策狀態
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(np.array(state_vector)).unsqueeze(0).to(device)
            if hidden_state is None:
                hidden_state = (torch.zeros(1, 1, 128).to(device), torch.zeros(1, 1, 128).to(device))
            
            q_actions, _, _ = model(state_t, hidden_state)
            q_vals = q_actions.cpu().numpy()[0]
            best_action_idx = int(np.argmax(q_vals))

        # ==========================================
        # 3. 座標佈局 (維持寬螢幕比例不變)
        # ==========================================
        in_pos = [(150 * SCALE, (y * 30 + 50) * SCALE) for y in range(20)]    
        fc1_pos = [(400 * SCALE, (y * 40 + 100) * SCALE) for y in range(12)]  
        lstm_pos = [(750 * SCALE, (y * 40 + 100) * SCALE) for y in range(12)] 
        act_pos = [(1050 * SCALE, (y * 45 + 100) * SCALE) for y in range(11)] 

        fc1_w = model.fc1.weight.data.cpu().numpy()
        lstm_w = model.lstm.weight_ih_l0.data.cpu().numpy()[:128, :]
        act_w = model.fc_action.weight.data.cpu().numpy()

        # ==========================================
        # 5. 繪製連線 (🎨 霓虹螢光配色)
        # ==========================================
        
        # A. 輸入層 -> FC1 特徵層 
        for h_idx in range(12): 
            actual_h = h_idx * 10
            weights = fc1_w[actual_h, :]
            top_2_inputs = np.argsort(np.abs(weights))[-2:]
            
            for i in range(20):
                w = weights[i]
                if i in top_2_inputs and abs(w) > 0.1:
                    # 正權重：霓虹青 (Cyan) / 負權重：賽博粉 (Magenta)
                    color = (0, 200, 255) if w > 0 else (255, 50, 150)
                    active_multiplier = 2.0 if state_vector[i] > 0.1 else 0.5
                    width = max(1, int(abs(w) * 4 * active_multiplier)) * SCALE
                    pygame.draw.line(hq_surface, color, in_pos[i], fc1_pos[h_idx], width)

        # B. FC1 特徵層 -> LSTM 記憶層 
        for j in range(12): 
            actual_j = j * 10
            for i in range(12): 
                actual_i = i * 10
                w = lstm_w[actual_j, actual_i]
                abs_w = abs(w)
                
                if abs_w > 0.05: 
                    if w > 0:
                        # 高權重：亮青色 / 低權重：暗藍綠色
                        color = (0, 200, 255) if abs_w > 0.15 else (0, 80, 120)
                    else:
                        # 高權重：亮粉紅 / 低權重：暗紫紅色
                        color = (255, 50, 150) if abs_w > 0.15 else (120, 20, 70)
                    
                    width = max(1, int(abs_w * 15)) * SCALE
                    pygame.draw.line(hq_surface, color, fc1_pos[i], lstm_pos[j], width)

        # C. LSTM 記憶層 -> 動作層 
        for a_idx in range(11):
            if a_idx not in allowed_indices: continue
            
            weights = act_w[a_idx, :]
            sampled_weights = [weights[h * 10] for h in range(12)]
            top_2_hidden = np.argsort(np.abs(sampled_weights))[-2:]

            for h_idx in range(12):
                actual_h = h_idx * 10
                w = act_w[a_idx, actual_h]
                if h_idx in top_2_hidden and abs(w) > 0.1:
                    color = (0, 200, 255) if w > 0 else (255, 50, 150)
                    is_active_path = (a_idx == best_action_idx)
                    width = max(1, int(abs(w) * (6 if is_active_path else 2))) * SCALE
                    pygame.draw.line(hq_surface, color, lstm_pos[h_idx], act_pos[a_idx], width)

        # ==========================================
        # 🎨 6. 節點渲染 (全息科技風格)
        # ==========================================
        def draw_neat_node(pos, label, num, side="left", active_val=0, is_best=False, is_masked=False):
            if is_masked:
                # 被遮罩：極暗灰
                fill_color, border_color, text_color = (20, 20, 20), (50, 50, 50), (100, 100, 100)
            elif is_best:
                # 最終決策：亮青色發光
                fill_color, border_color, text_color = (0, 80, 150), (0, 255, 255), (255, 255, 255)
            elif active_val > 0.1:
                # 活躍輸入：亮綠色發光
                fill_color, border_color, text_color = (0, 100, 50), (0, 255, 100), (255, 255, 255)
            else:
                # 一般未啟動狀態：深底色搭配暗藍邊框
                fill_color, border_color, text_color = (15, 20, 25), (40, 80, 120), (200, 200, 200)

            radius = 13 * SCALE
            pygame.draw.circle(hq_surface, fill_color, pos, radius)
            pygame.draw.circle(hq_surface, border_color, pos, radius, 2 * SCALE)
            
            n_txt = num_font.render(str(num), True, text_color)
            hq_surface.blit(n_txt, (pos[0]-n_txt.get_width()//2, pos[1]-n_txt.get_height()//2))
            
            if label:
                # 🎨 文字顏色反白，被遮罩的字變暗灰
                l_txt = font.render(label, True, (220, 230, 240) if not is_masked else (100, 100, 100))
                x_off = (-105 * SCALE) if side == "left" else (25 * SCALE)
                y_off = -8 * SCALE
                hq_surface.blit(l_txt, (pos[0] + x_off, pos[1] + y_off))

        for i, pos in enumerate(in_pos): draw_neat_node(pos, input_labels[i], i, "left", state_vector[i])
        for i, pos in enumerate(fc1_pos): draw_neat_node(pos, None, i + 100)
        for i, pos in enumerate(lstm_pos): draw_neat_node(pos, None, i + 200)
        for i, pos in enumerate(act_pos):
            is_masked = (i not in allowed_indices)
            draw_neat_node(pos, action_labels[i], i, "right", 0, is_best=(i == best_action_idx and not is_masked), is_masked=is_masked)

        # 🎬 平滑縮小並貼回真實螢幕
        final_image = pygame.transform.smoothscale(hq_surface, (sw, sh))
        surface.blit(final_image, (0, 0))

    def get_action_mask(target_obs, last_act_id=0): 
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
        
        

        #if float(player.food_workers) >= 22 and 14 in allowed_acts: allowed_acts.remove(14)
        #你直接告訴 AI「工兵最多22個」，但 AI 永遠不會知道「為什麼是22個」、「多生一個工兵會不會更好」。
        #教授說的競爭關係就是要讓 AI 自己學到「工兵夠了就不用再生，要把資源拿去造掠奪者」，而不是你幫它決定。
        
        # 👉 修正 2：兵營只要有 2 座就夠了，禁止再蓋，避免卡建築網格
        #if barracks >= 2 and 2 in allowed_acts: allowed_acts.remove(2)
        #同樣道理，你幫 AI 決定了「兵營最多2個」。但也許 AI 會學到「3個兵營可以同時生產掠奪者，速度更快」，
        #你把這個可能性直接封死了。
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
        
        
            
        if player.idle_worker_count == 0 and 41 in allowed_acts: 
            allowed_acts.remove(41)
        
        if supply_surplus < 1:
            for act in [14, 16]: 
                if act in allowed_acts: allowed_acts.remove(act)
        
        
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
        
        # ==========================================
        # 🛡️ 新增：防刷屏機制 (禁止連續執行純行政指令)
        # ==========================================
        if last_act_id == 45 and 45 in allowed_acts: 
            allowed_acts.remove(45) # 剛剛編過隊了，這回合不准再編
        

        # 🛡️ 正確位置在 return 之前
        if not allowed_acts:
            allowed_acts = [44] 
        return [VALID_ACTIONS.index(act) for act in allowed_acts]
        # 🛡️ 保底機制：如果所有動作都被過濾掉了，強制保留「發呆」
        

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
            h_in, c_in = sequence[0][6] 
            hidden = (h_in.to(device), c_in.to(device))
            
            for step_data in sequence:
                # 📦 完整解包 10 個參數
                state = torch.FloatTensor([step_data[0]]).to(device)
                action = torch.LongTensor([step_data[1]]).to(device)
                param_target = torch.LongTensor([step_data[2]]).to(device)
                reward = torch.FloatTensor([step_data[3]]).to(device)
                next_state = torch.FloatTensor([step_data[4]]).to(device)
                done = step_data[5]
                # step_data[6] 是 hidden_in，由外部維護
                h_out, c_out = step_data[7]
                hidden_out = (h_out.to(device), c_out.to(device))
                gamma_mult = step_data[8]
                next_allowed = step_data[9]

                # 🧠 第一大腦預測分數
                q_actions, q_params, next_hidden = brain_model(state, hidden)
                q_value = q_actions.gather(1, action.unsqueeze(1)).squeeze(1)

                # 🔮 Double DQN 計算未來收益
                with torch.no_grad():
                    next_q_actions, _, _ = brain_model(next_state, hidden_out)
                    next_mask = torch.full_like(next_q_actions, float('-inf'))
                    next_mask[0, next_allowed] = 0
                    best_next_action = (next_q_actions + next_mask).argmax(1).unsqueeze(1)
                    
                    target_q_actions, _, _ = target_model(next_state, hidden_out)
                    max_next_q = target_q_actions.gather(1, best_next_action).squeeze(1)
                    
                    target_q = reward + (gamma_mult * max_next_q * (1 - done))

                # ⚖️ 計算損失並更新
                action_loss = nn.SmoothL1Loss()(q_value, target_q)
                param_loss = torch.tensor(0.0).to(device)
                if reward > 0:
                    param_loss = nn.CrossEntropyLoss()(q_params, param_target)
                
                total_loss += (action_loss + 0.5 * param_loss)
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
        with open(current_session_file, "w", encoding="utf-8") as f:
            f.write("Episode,Task,Epsilon,Total_Reward,End_Loop,Marauders,Marines\n")


        best_record = [13440.0] 

        # 如果你有舊的紀錄檔就讀取，沒有就跳過
        best_time_path = os.path.join(log_dir, "best_time.txt")
        if os.path.exists(best_time_path):
            with open(best_time_path, "r") as f:
                best_record[0] = float(f.read().strip())
            print(f"📖 載入歷史紀錄：目前最短完成時間為 {best_record[0]} 幀")
                # --- 修正 1：初始化歷史最低時間紀錄 ---


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
            final_reward = 0.0
            achieved_milestones = set()
            

            while True:
                
                step_reward = 0.0
            
                
                
                
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
                
                allowed_indices = get_action_mask(obs, last_action_id)
                
                

                
                # 選擇動作 (降維版 + 拘束器發威)
                # 選擇動作 (降維版 + 拘束器發威)
                if random.random() <= epsilon:
                    # ✨ 從「合法」的選項中隨機挑選索引！(不再是全部盲選)
                    action_index = random.choice(allowed_indices)
                    p_id = random.randint(1, 100)
                    
                    # 🌟 補上這行！把 index 轉換成真實的動作 ID
                    a_id = VALID_ACTIONS[action_index] 
                    
                    with torch.no_grad():
                        _, _, next_hidden_state = brain_model(state_t.unsqueeze(0), hidden_state)
                else:
                    with torch.no_grad():
                        # q_actions 輸出 11 個分數
                        
                        q_actions, q_params, next_hidden_state = brain_model(state_t.unsqueeze(0), hidden_state)
                        
                        # ✨ 套用拘束器遮罩
                        mask = torch.full_like(q_actions, float('-inf'))
                        mask[0, allowed_indices] = 0
                        masked_q_actions = q_actions + mask
                        
                        # 從合法的動作中，選出分數最高的那一個索引
                        # 從合法的動作中，選出分數最高的那一個索引
                        action_index = masked_q_actions.argmax().item()
                        
                        # ✨ 核心修正：加上安全鎖，確保大腦算出來的 p_id 落在 1~100 之間！
                        raw_p_id = q_params.argmax().item()
                        p_id = (raw_p_id % 100) + 1 

                        # ==========================================
                        # 🚑 核心急救機制：撞牆強制換位
                        # ==========================================
                        a_id = VALID_ACTIONS[action_index]
                        
                        if a_id == last_action_id and last_action_success == 0.0:
                            # 強制換位也要改成 1~100
                            p_id = random.randint(1, 100)
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
                next_allowed_indices = get_action_mask(next_obs, a_id)

                # ... 在你的訓練迴圈內 ...

                if RENDER_UI and train_step_counter % 2 == 0:
                    # ✨ 核心修正：加入這兩行，讓視窗學會「聽話」
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return # 讓你可以手動關閉視窗結束訓練

                    # 執行原本的繪圖
                    draw_dual_head_network(screen, brain_model, state, allowed_indices, hidden_state)
                    pygame.display.flip()

                # ==========================================
                # ✨ 核心修正：判斷動作是否「真的」被遊戲接受
                # ==========================================
                if sc2_action.function != 0:
                    current_action_success = 1.0
                    
                    # 讀取星海引擎的底層回報
                    # action_result 陣列裡，1 代表成功，大於 1 代表各種報錯 (例如 32 是位置無效)
                    action_results = next_obs.observation.action_result
                    if len(action_results) > 0:
                        for res in action_results:
                            if res != 1:
                                current_action_success = 0.0
                                # print(f"🚫 引擎拒絕動作 (錯誤碼: {res})，觸發換位機制！")
                                break
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
                    real_barracks_count, 
                    a_id,
                    p_id  # 🌟 補上這行！把 AI 決定的網格位置傳給紀錄器
                )

                # ==========================================
                # 🏆 穩定版全局計分系統 (Dense Reward Shaping)
                # ==========================================
                
                # ==========================================
            # 🏆 Reward 計算區塊（教授建議簡化版）
            # 設計理念：
            #   - 所有 reward 最終透過 tanh 壓縮至 [-1, +1]
            #   - 不使用手動除法，讓數學函數自然做 normalization
            #   - 過程獎勵（里程碑）只給方向，終局獎勵才是真正的學習訊號
            # ==========================================

                step_reward = 0.0  # 每一幀開始時，先將本幀獎勵歸零

                # ==========================================
                # 第一步：科技樹里程碑（追蹤用，不給分）
                # 設計理念：
                #   里程碑只用來追蹤建築完成度（x2），
                #   不再給過程分數，避免 AI 刷建築分數。
                #   achieved_milestones 的大小會在終局結算時用到。
                # ==========================================

                if "depot" not in achieved_milestones and current_depots >= 1:
                    achieved_milestones.add("depot")
                    print("補給站里程碑達成")

                if "barracks" not in achieved_milestones and current_barracks >= 1:
                    achieved_milestones.add("barracks")
                    print("兵營里程碑達成")

                if "refinery" not in achieved_milestones and current_refineries >= 1:
                    achieved_milestones.add("refinery")
                    print("瓦斯廠里程碑達成")

                if "techlab" not in achieved_milestones and current_techlabs >= 1:
                    achieved_milestones.add("techlab")
                    print("科技室里程碑達成")

                # ==========================================
                # 第二步：掠奪者產量追蹤（不給過程分）
                # 設計理念：
                #   只追蹤數量變化，實際加分在終局統一結算。
                # ==========================================
                if current_real_count > getattr(agent, 'last_target_count', 0):
                    agent.last_target_count = current_real_count  # 更新追蹤基準
                    #step_reward += 0.15
                    print(f"🎯 掠奪者產出！目前進度: {current_real_count}/5")

                # ==========================================
                # 第三步：終局判定
                # ==========================================
                current_loop = int(next_obs.observation.game_loop[0])
                done = next_obs.last() or current_loop >= 13440 or current_real_count >= 5

                # ==========================================
                # 第四步：終局結算（教授 X-Y 軸公式）
                # Y 軸：造出五隻掠奪者（任務核心目標）
                # X 軸：產量 + 建築完成度 + 時間效率
                # ==========================================



                if done:
                    # Y 軸
                    y = min(current_real_count / 5.0, 1.0)
                    # 0隻=0.0, 1隻=0.2, 2隻=0.4, 3隻=0.6, 4隻=0.8, 5隻=1.0

                    # X 軸三個維度
                    x1 = y                                         # X1. 掠奪者產量（直接等於Y軸）
                    x2 = len(achieved_milestones) / 4.0            # X2. 建築完成度（蓋了幾個里程碑）
                    x3 = 1.0 - (float(current_loop) / 13440.0)     # X3. 時間效率（越快越高）


                    # 成功和失敗用不同公式
                    if current_real_count >= 5:
                        # 成功：產量 + 建築 + 時間效率都算
                        raw_score = 0.7 * x1 + 0.1 * x2 + 0.2 * x3
                    else:
                        # 失敗：只看造了幾隻，直接給負數
                        # 0隻=-0.7, 1隻=-0.5, 2隻=-0.3, 3隻=-0.1, 4隻=+0.1
                        raw_score = x1 - 0.7

                    # 統一用 tanh 壓縮到 [-1, +1]
                    step_reward = raw_score * 3.0

                    # 更新最短時間紀錄（只在成功時）
                    if current_real_count >= 5 and float(current_loop) < best_record[0]:
                        best_record[0] = float(current_loop)
                        with open(best_time_path, "w") as f:
                            f.write(str(best_record[0]))
                        print(f"🔥 新紀錄！最短完成時間: {best_record[0]} 幀")

                    print(f"終局: Y={y:.2f}, X1={x1:.2f}, X2={x2:.2f}, X3={x3:.2f}, raw={raw_score:.4f}")
                

                # ==========================================
                # 第五步：tanh 壓縮（核心）
                # 只有終局幀的 step_reward 不為 0，
                # 過程幀全部是 0，不干擾訓練。
                # ==========================================
                scaled_reward = math.tanh(step_reward)
                episode_reward += scaled_reward
                #episode_reward += scaled_reward  # 累計這局的總分（用於 CSV 記錄與排行榜）
                if done:
                    final_reward = scaled_reward  # 只取終局幀，天然在[-1,+1]，給教授看、寫CSV

                if step_reward != 0:
                    print(f"DEBUG: raw={step_reward:.4f}, tanh後={scaled_reward:.4f}, 累計={episode_reward:.4f}")

                # ==========================================
                # 第六步：取得下一幀狀態，並打包成訓練用的 Transition
                # ==========================================
                next_time_loop = int(next_obs.observation.game_loop[0])
                next_state = get_state_vector(
                    next_obs,
                    getattr(agent, 'active_parameter', 1),
                    18,
                    a_id,
                    current_action_success,
                    next_time_loop
                )

                # 將這一幀的 (狀態, 動作, 獎勵, 下一狀態, 結束旗標, LSTM記憶) 打包存入 N-Step 緩衝區
                # ✨ 核心修正：將 p_id 也打包進去 (記得減 1 轉回 0~63 索引)
                current_transition = (state, action_index, p_id - 1, scaled_reward, next_state, done, hidden_state, next_hidden_state, next_allowed_indices)
                n_step_buffer.append(current_transition)

                # 更新觀測值與狀態，準備進入下一幀
                obs = next_obs
                state = next_state
                hidden_state = (next_hidden_state[0].detach(), next_hidden_state[1].detach())
                last_action_id = a_id
                last_action_success = current_action_success



                # ==========================================
                # ⚠️ 第三步：N-Step Bootstrapping 緩衝與寫入
                # ==========================================
                N_STEP = 7
                GAMMA = 0.99

                # 2. 如果視窗滿了，結算過去 N 步的總報酬，並存入大腦
                # 2. 如果視窗滿了，結算過去 N 步的總報酬，並存入大腦
                if len(n_step_buffer) == N_STEP:
                    # 👉 修正 1：reward 被擠到了 index 3
                    n_reward = sum([n_step_buffer[i][3] * (GAMMA ** i) for i in range(N_STEP)])
                    
                    n_state = n_step_buffer[0][0]
                    n_action = n_step_buffer[0][1]
                    n_p_id = n_step_buffer[0][2]          # ✨ 新增：拿出當時的 p_id
                    n_next_state = n_step_buffer[-1][4]   # 👉 修正：後面的 index 全部 +1
                    n_done = n_step_buffer[-1][5]         
                    
                    # 👉 修正：hidden_state 現在在 6 跟 7
                    n_hidden_in = (n_step_buffer[0][6][0].cpu(), n_step_buffer[0][6][1].cpu())
                    n_hidden_out = (n_step_buffer[-1][7][0].cpu(), n_step_buffer[-1][7][1].cpu())
                    n_next_allowed = n_step_buffer[-1][8] # 👉 修正：到了 8

                    # ✨ 修正 2：打包成新的包裹時，把 n_p_id 放進第三個位置 (index 2)
                    n_transition = (n_state, n_action, n_p_id, n_reward, n_next_state, n_done, n_hidden_in, n_hidden_out, GAMMA ** N_STEP, n_next_allowed)
                    
                    episode_memory.append(n_transition)
                    n_step_buffer.popleft()

                # 3. 遊戲結束時，強制清空並結算緩衝區裡剩下的尾巴
                # 3. 遊戲結束時，強制清空並結算緩衝區裡剩下的尾巴
                if done:
                    while len(n_step_buffer) > 0:
                        actual_n = len(n_step_buffer)
                        # 👉 同樣把 [2] 改成 [3]
                        n_reward = sum([n_step_buffer[i][3] * (GAMMA ** i) for i in range(actual_n)])
                        
                        n_state = n_step_buffer[0][0]
                        n_action = n_step_buffer[0][1]
                        n_p_id = n_step_buffer[0][2]          # ✨ 新增
                        n_next_state = n_step_buffer[-1][4]   # 👉 修正
                        n_done = n_step_buffer[-1][5]
                        
                        n_hidden_in = (n_step_buffer[0][6][0].cpu(), n_step_buffer[0][6][1].cpu())
                        n_hidden_out = (n_step_buffer[-1][7][0].cpu(), n_step_buffer[-1][7][1].cpu())
                        n_next_allowed = n_step_buffer[-1][8]
                        
                        # ✨ 同步把 n_p_id 放進來
                        n_transition = (n_state, n_action, n_p_id, n_reward, n_next_state, n_done, n_hidden_in, n_hidden_out, GAMMA ** actual_n, n_next_allowed)
                        
                        episode_memory.append(n_transition)
                        n_step_buffer.popleft()
               
                if done:
                    # 👉 新增這行：把這整局的連續記憶打包存進一般記憶庫
                    memory.append(episode_memory) 

                    if current_real_count >= 5:
                        success_memory.append((current_loop, episode_memory))
                        success_memory.sort(key=lambda x: x[0], reverse=False)
                        
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
                    # 只壓縮用於顯示的分數，不影響任何訓練邏輯
                    

                    print(f"!!! 即將寫入 CSV 的分數是: {final_reward} (型別: {type(final_reward)})")
                    #logger.log_episode(ep + 1, "掠奪者任務", epsilon, float(episode_reward), current_loop, current_real_count, marine_count)
                    #log_file = f"d:/RL_Projects/dudu DRQN/log/dqn_training_log_{int(time.time())}.csv"
                    with open(current_session_file, "a", encoding="utf-8") as f:
                        line = f"{ep+1},掠奪者任務,{epsilon:.3f},{final_reward:.4f},{current_loop},{current_real_count},{marine_count}\n"
                        f.write(line)
                    print(f"✅ 已手動強制寫入 CSV: {final_reward:.4f}")
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