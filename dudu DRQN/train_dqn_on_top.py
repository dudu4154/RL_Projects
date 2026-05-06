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

# 匯入底層腳本
# --- train_dqn_on_top.py 的最上方 ---
import production_ai
from production_ai import ProductionAI
import logging
from absl import logging as absl_logging

# 1. 嘗試讀取系統環境變數 SC2PATH
sc2_path = os.environ.get("SC2PATH")

# 2. 如果系統裡沒設定，提供幾個常見的預設路徑讓它自己去試
if not sc2_path:
    common_paths = [
        r"C:\Program Files (x86)\StarCraft II",
        r"D:\StarCraft II",
        r"E:\StarCraft II",
        "/Applications/StarCraft II" # Mac 預設路徑
    ]
    for path in common_paths:
        if os.path.exists(path):
            sc2_path = path
            os.environ["SC2PATH"] = path
            break

if not sc2_path:
    print("⚠️ 警告：找不到 StarCraft II 安裝路徑！請設定環境變數 SC2PATH。")

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
    screen = pygame.display.set_mode((1600, 900))
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

        SCALE = 1.4   # ⭐ 不要用2，會卡
        sw, sh = surface.get_size()
        hq_surface = pygame.Surface((int(sw * SCALE), int(sh * SCALE)))

        hq_surface.fill((10, 12, 18))

        font = pygame.font.SysFont("microsoftjhenghei", int(12 * SCALE))
        num_font = pygame.font.SysFont("arial", int(10 * SCALE), bold=True)

        # ===== 自動維度 =====
        STATE_DIM = len(state_vector)

        # ===== 20 個特徵的真實名稱 =====
        input_labels = [
            "工兵總數",          # 0: food_workers
            "閒置工兵",          # 1: idle_worker_count
            "晶礦比例",          # 2: minerals
            "瓦斯比例",          # 3: vespene
            "資源差距",          # 4: mineral - vespene
            "目前人口",          # 5: food_used
            "剩餘人口空間",       # 6: food_cap - food_used
            "補給站數量",         # 7: count_unit(19)
            "瓦斯廠數量",         # 8: count_unit(20)
            "軍營數量",          # 9: count_unit(21)
            "科技室數量",         # 10: count_unit(37)
            "選中工兵",          # 11: is_scv_selected
            "選中主堡",          # 12: is_cc_selected
            "選中兵營",          # 13: is_barracks_selected
            "陸戰隊數",          # 14: count_unit(48)
            "掠奪者數",          # 15: count_unit(51)
            "上步成功",          # 16: last_action_success
            "上一動作ID",        # 17: last_action_id
            "上一座標",          # 18: current_block
            "遊戲時間"           # 19: current_loop
        ]

        action_labels = [
            "蓋補給站","蓋兵營","蓋瓦斯","造工兵","造陸戰隊",
            "造掠奪者","升級","採礦","採氣","待機","編隊"
        ]

        # ===== 推論 =====
        device = next(model.parameters()).device
        with torch.no_grad():
            s = torch.FloatTensor(np.array(state_vector)).unsqueeze(0).to(device)
            if hidden_state is None:
                hidden_state = (torch.zeros(1,1,128).to(device), torch.zeros(1,1,128).to(device))
            q,_,_ = model(s, hidden_state)
            q_vals = q.cpu().numpy()[0]
            best = int(np.argmax(q_vals))

        # ===== 自動排版（重點）=====
        layer_x = np.linspace(80*SCALE, sw*SCALE-80*SCALE, 4)

        in_pos   = [(layer_x[0], (i/(STATE_DIM-1+1e-5)*sh*SCALE*0.8 + 50*SCALE)) for i in range(STATE_DIM)]
        fc1_pos  = [(layer_x[1], (i/11*sh*SCALE*0.8 + 50*SCALE)) for i in range(12)]
        lstm_pos = [(layer_x[2], (i/11*sh*SCALE*0.8 + 50*SCALE)) for i in range(12)]
        act_pos  = [(layer_x[3], (i/10*sh*SCALE*0.8 + 50*SCALE)) for i in range(11)]

        fc1_w = model.fc1.weight.data.cpu().numpy()
        lstm_w = model.lstm.weight_ih_l0.data.cpu().numpy()[:128, :]
        act_w = model.fc_action.weight.data.cpu().numpy()

        # ===== 畫線 =====
        for h in range(12):
            w = fc1_w[h*10]
            top = np.argsort(np.abs(w))[-2:]
            for i in range(STATE_DIM):
                if i in top:
                    col = (0,200,255) if w[i]>0 else (255,50,150)
                    pygame.draw.line(hq_surface, col, in_pos[i], fc1_pos[h], 2)

        for j in range(12):
            for i in range(12):
                w = lstm_w[j*10][i*10]
                if abs(w)>0.05:
                    col = (0,200,255) if w>0 else (255,50,150)
                    pygame.draw.line(hq_surface, col, fc1_pos[i], lstm_pos[j], 1)

        for a in range(11):
            if a not in allowed_indices: continue
            w = act_w[a]
            top = np.argsort(np.abs([w[i*10] for i in range(12)]))[-2:]
            for i in range(12):
                if i in top:
                    col = (0,200,255) if w[i*10]>0 else (255,50,150)
                    width = 3 if a==best else 1
                    pygame.draw.line(hq_surface, col, lstm_pos[i], act_pos[a], width)

        # ===== 畫節點 =====
        def node(pos, txt, active=False, best=False, masked=False):
            if masked:
                fill=(30,30,30); border=(70,70,70)
            elif best:
                fill=(0,120,200); border=(0,255,255)
            elif active:
                fill=(0,100,50); border=(0,255,100)
            else:
                fill=(15,20,25); border=(50,100,150)

            pygame.draw.circle(hq_surface, fill, pos, int(10*SCALE))
            pygame.draw.circle(hq_surface, border, pos, int(10*SCALE), 2)

            t = num_font.render(str(txt), True, (255,255,255))
            hq_surface.blit(t, (pos[0]-t.get_width()//2, pos[1]-t.get_height()//2))

        # ===== 畫節點與標籤 =====
        for i, p in enumerate(in_pos): 
            node(p, i, state_vector[i] > 0.1)
            # 在輸入節點「左側」畫上特徵名稱
            label_txt = font.render(input_labels[i], True, (180, 200, 220))
            hq_surface.blit(label_txt, (p[0] - label_txt.get_width() - 15 * SCALE, p[1] - label_txt.get_height() // 2))

        for i, p in enumerate(fc1_pos): 
            node(p, i)
            
        for i, p in enumerate(lstm_pos): 
            node(p, i)

        for i, p in enumerate(act_pos):
            node(p, i, best=(i == best and i in allowed_indices), masked=(i not in allowed_indices))
            # 在輸出節點「右側」畫上動作名稱，並根據狀態改變顏色
            text_color = (255, 205, 90) if (i == best and i in allowed_indices) else (180, 200, 220)
            if i not in allowed_indices: text_color = (80, 80, 80) # 灰暗色代表被 Mask 擋住不能用
            
            label_txt = font.render(action_labels[i], True, text_color)
            hq_surface.blit(label_txt, (p[0] + 15 * SCALE, p[1] - label_txt.get_height() // 2))

        final = pygame.transform.smoothscale(hq_surface,(sw,sh))
        surface.blit(final,(0,0))
            
    # =========================================================
    # HUD Dashboard：不用嵌 SC2 視窗，左上用 feature_screen 畫戰場
    # =========================================================

    ACTION_NAMES = {
        1: "Build Depot",
        2: "Build Barracks",
        11: "Build Refinery",
        14: "Train SCV",
        16: "Train Marine",
        18: "Train Marauder",
        34: "Build TechLab",
        41: "Gather Minerals",
        42: "Assign Gas",
        44: "Idle",
        45: "Bind CC"
    }

    HUD_HISTORY = {
        "minerals": deque(maxlen=120),
        "vespene": deque(maxlen=120),
    }

    def ui_colors():
        return {
            "bg": (6, 18, 24),
            "panel": (8, 26, 34),
            "panel2": (12, 34, 44),
            "border": (70, 220, 210),
            "title": (150, 255, 235),
            "text": (190, 255, 245),
            "good": (120, 255, 130),
            "warn": (255, 205, 90),
            "bad": (255, 100, 100),
            "cyan": (80, 220, 255),
            "gray": (85, 105, 115),
            "darkgray": (45, 55, 60),
        }

    def get_fonts():
        return {
            "title": pygame.font.SysFont("microsoftjhenghei", 22, bold=True),
            "header": pygame.font.SysFont("microsoftjhenghei", 18, bold=True),
            "text": pygame.font.SysFont("microsoftjhenghei", 16),
            "small": pygame.font.SysFont("microsoftjhenghei", 14),
            "timer": pygame.font.SysFont("consolas", 42, bold=True),
        }

    def draw_glow_rect(surface, rect, fill, border, border_width=2):
        pygame.draw.rect(surface, fill, rect, border_radius=8)
        pygame.draw.rect(surface, border, rect, border_width, border_radius=8)

    def draw_panel(surface, rect, title, fonts, colors):
        draw_glow_rect(surface, rect, colors["panel"], colors["border"], 2)
        txt = fonts["title"].render(title, True, colors["title"])
        surface.blit(txt, (rect[0] + 12, rect[1] + 10))

    def count_units_from_screen(obs, unit_id, pixels_per_unit=1.0):
        unit_type = obs.observation.feature_screen.unit_type
        player_rel = obs.observation.feature_screen.player_relative
        pixels = np.sum((unit_type == unit_id) & (player_rel == 1))
        if pixels_per_unit <= 1:
            return int(pixels)
        return int(np.round(float(pixels) / float(pixels_per_unit)))

    def get_dashboard_snapshot(obs):
        player = obs.observation.player
        current_loop = int(obs.observation.game_loop[0])

        return {
            "minerals": int(player.minerals),
            "vespene": int(player.vespene),
            "workers": int(player.food_workers),
            "food_used": int(player.food_used),
            "food_cap": int(player.food_cap),
            "idle_workers": int(player.idle_worker_count),
            "depots": count_units_from_screen(obs, 19, 69.0),
            "refineries": count_units_from_screen(obs, 20, 97.0),
            "barracks": count_units_from_screen(obs, 21, 137.0),
            "techlabs": count_units_from_screen(obs, 37, 85.0),
            "marines": count_units_from_screen(obs, 48, 10.0),
            "marauders": count_units_from_screen(obs, 51, 10.0),
            "loop": current_loop,
            "remain_loop": max(0, 13440 - current_loop),
        }

    def format_loop_to_mmss(loop_value):
        total_seconds = int((loop_value / 13440.0) * 600)
        return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"

    def draw_feature_game_panel(surface, rect, obs, fonts, colors):
        draw_panel(surface, rect, "AGENT CONTROL - Feature Screen", fonts, colors)

        x, y, w, h = rect
        inner = pygame.Rect(x + 12, y + 45, w - 24, h - 57)
        pygame.draw.rect(surface, (15, 30, 38), inner, border_radius=6)

        unit_type = obs.observation.feature_screen.unit_type
        player_rel = obs.observation.feature_screen.player_relative

        sh, sw = unit_type.shape
        sx = inner.width / float(sw)
        sy = inner.height / float(sh)

        # 礦
        mineral_pts = np.argwhere(unit_type == 341)
        step = max(1, len(mineral_pts) // 120 + 1)
        for yy, xx in mineral_pts[::step]:
            px = int(inner.x + xx * sx)
            py = int(inner.y + yy * sy)
            pygame.draw.circle(surface, (70, 140, 255), (px, py), 3)

        # 我方
        own_pts = np.argwhere(player_rel == 1)
        step = max(1, len(own_pts) // 250 + 1)
        for yy, xx in own_pts[::step]:
            px = int(inner.x + xx * sx)
            py = int(inner.y + yy * sy)
            pygame.draw.circle(surface, colors["good"], (px, py), 3)

        # 敵方
        enemy_pts = np.argwhere(player_rel == 4)
        step = max(1, len(enemy_pts) // 150 + 1)
        for yy, xx in enemy_pts[::step]:
            px = int(inner.x + xx * sx)
            py = int(inner.y + yy * sy)
            pygame.draw.circle(surface, colors["bad"], (px, py), 3)

    def q_values_to_confidence(q_values, allowed_indices):
        q = np.array(q_values, dtype=np.float32)

        # ✨ 直接對「所有」原始 Q 值進行 Softmax 轉換
        # 移除原本的 -1e9 遮罩設定，讓被 LOCK 的動作也能參與百分比計算
        exp_q = np.exp(q - np.max(q))
        
        return exp_q / max(np.sum(exp_q), 1e-6)

    def draw_bar(surface, x, y, w, h, value, fg, bg):
        pygame.draw.rect(surface, bg, (x, y, w, h), border_radius=4)
        fill_w = int(max(0, min(1, value)) * w)
        pygame.draw.rect(surface, fg, (x, y, fill_w, h), border_radius=4)

    def draw_action_table(surface, rect, q_values, allowed_indices, valid_actions, fonts, colors):
        draw_panel(surface, rect, "Current Action Intent & Masks", fonts, colors)

        x, y, w, h = rect
        conf = q_values_to_confidence(q_values, allowed_indices)
        best_idx = int(np.argmax(q_values)) if len(q_values) > 0 else -1

        surface.blit(fonts["header"].render("Action", True, colors["text"]), (x + 15, y + 50))
        surface.blit(fonts["header"].render("Confidence", True, colors["text"]), (x + 210, y + 50))
        surface.blit(fonts["header"].render("Mask", True, colors["text"]), (x + 360, y + 50))

        row_y = y + 82
        for i, act_id in enumerate(valid_actions):
            yy = row_y + i * 28
            is_open = i in allowed_indices
            is_best = i == best_idx and is_open

            color = colors["warn"] if is_best else (colors["good"] if is_open else colors["gray"])
            name = ACTION_NAMES.get(act_id, f"Action {act_id}")

            if is_best:
                pygame.draw.rect(surface, (28, 50, 60), (x + 8, yy - 2, w - 16, 24), border_radius=4)

            surface.blit(fonts["text"].render(name, True, color), (x + 15, yy))
            draw_bar(surface, x + 210, yy + 5, 120, 14, float(conf[i]), colors["good"], (20, 40, 46))
            surface.blit(fonts["small"].render(f"{int(conf[i] * 100)}%", True, color), (x + 335, yy))
            surface.blit(fonts["text"].render("OPEN" if is_open else "LOCK", True, color), (x + 380, yy))

    def draw_resource_panel(surface, rect, obs, fonts, colors):
        draw_panel(surface, rect, "Current Resources", fonts, colors)
        x, y, w, h = rect
        s = get_dashboard_snapshot(obs)

        items = [
            ("Minerals", s["minerals"], colors["cyan"]),
            ("Vespene", s["vespene"], colors["good"]),
            ("SCV", s["workers"], colors["text"]),
            ("Supply", f'{s["food_used"]}/{s["food_cap"]}', colors["warn"]),
            ("Barracks", s["barracks"], colors["text"]),
            ("TechLab", s["techlabs"], colors["text"]),
            ("Marauders", s["marauders"], colors["good"]),
        ]

        for i, (label, value, color) in enumerate(items):
            yy = y + 45 + i * 25
            surface.blit(fonts["text"].render(label, True, colors["text"]), (x + 15, yy))
            txt = fonts["text"].render(str(value), True, color)
            surface.blit(txt, (x + w - 20 - txt.get_width(), yy))

    def draw_timer_panel(surface, rect, obs, fonts, colors):
        draw_panel(surface, rect, "Mission Timer", fonts, colors)
        x, y, w, h = rect
        s = get_dashboard_snapshot(obs)

        surface.blit(fonts["timer"].render(format_loop_to_mmss(s["remain_loop"]), True, colors["cyan"]), (x + 20, y + 50))

        status = f'{s["marauders"]}/5 Marauders Target'
        surface.blit(fonts["header"].render(status, True, colors["good"]), (x + 20, y + 110))

    def draw_timeline(surface, rect, obs, fonts, colors):
        draw_panel(surface, rect, "Production Milestone Timeline", fonts, colors)
        x, y, w, h = rect
        s = get_dashboard_snapshot(obs)

        milestones = [
            ("Start", True),
            ("Depot", s["depots"] >= 1),
            ("Barracks", s["barracks"] >= 1),
            ("Refinery", s["refineries"] >= 1),
            ("TechLab", s["techlabs"] >= 1),
            ("Target(5)", s["marauders"] >= 5),
        ]

        line_y = y + 70
        start_x = x + 35
        end_x = x + w - 35
        pygame.draw.line(surface, colors["gray"], (start_x, line_y), (end_x, line_y), 3)

        for i, (label, done) in enumerate(milestones):
            px = start_x + int(i / (len(milestones) - 1) * (end_x - start_x))
            c = colors["good"] if done else colors["darkgray"]
            pygame.draw.circle(surface, c, (px, line_y), 9)
            pygame.draw.circle(surface, colors["border"], (px, line_y), 9, 2)
            txt = fonts["small"].render(label, True, c)
            surface.blit(txt, (px - txt.get_width() // 2, line_y + 18))
        progress = min(1.0, s["loop"] / 13440)

        bar_x = rect[0] + 30
        bar_y = rect[1] + rect[3] - 22
        bar_w = rect[2] - 60
        bar_h = 10

        pygame.draw.rect(surface, (40, 60, 70), (bar_x, bar_y, bar_w, bar_h), border_radius=5)
        pygame.draw.rect(surface, colors["cyan"], (bar_x, bar_y, int(bar_w * progress), bar_h), border_radius=5)    




    def draw_sparkline(surface, rect, values, color, fonts, colors, title):
        draw_glow_rect(surface, rect, colors["panel2"], colors["border"], 1)
        x, y, w, h = rect

        surface.blit(
            fonts["small"].render(title, True, colors["title"]),
            (x + 8, y + 6)
        )

        if len(values) < 2:
            return

        arr = np.array(values, dtype=np.float32)
        mn, mx = float(np.min(arr)), float(np.max(arr))

        if abs(mx - mn) < 1e-6:
            mx = mn + 1

        pts = []
        for i, v in enumerate(arr):
            px = x + 8 + int(i / max(1, len(arr) - 1) * (w - 16))
            py = y + h - 8 - int((v - mn) / (mx - mn) * (h - 35))
            pts.append((px, py))

        pygame.draw.lines(surface, color, False, pts, 2)

        latest = fonts["small"].render(str(int(arr[-1])), True, color)
        surface.blit(latest, (x + w - latest.get_width() - 8, y + 6))

    def draw_small_info(surface, rect, title, value, fonts, colors, history_values=None, line_color=None):
        draw_panel(surface, rect, title, fonts, colors)

        # 標題
        

        # 數字
        val_txt = fonts["header"].render(str(value), True, colors["text"])
        surface.blit(val_txt, (rect[0] + 12, rect[1] + 30))

        # 小折線圖
        if history_values is not None and len(history_values) >= 2:
            arr = np.array(history_values, dtype=np.float32)
            mn, mx = float(np.min(arr)), float(np.max(arr))
            if abs(mx - mn) < 1e-6:
                mx = mn + 1

            graph_x = rect[0] + int(rect[2] * 0.48)
            graph_y = rect[1] + 22
            graph_w = int(rect[2] * 0.46)
            graph_h = rect[3] - 30

            pts = []
            for i, v in enumerate(arr):
                px = graph_x + int(i / max(1, len(arr) - 1) * graph_w)
                py = graph_y + graph_h - int((v - mn) / (mx - mn) * graph_h)
                pts.append((px, py))

            pygame.draw.lines(surface, line_color or colors["cyan"], False, pts, 2)

    def draw_current(surface, rect, q_values, allowed_indices, valid_actions, fonts, colors):
        draw_panel(surface, rect, "Current Action Intent & Masks", fonts, colors)

        x, y, w, h = rect
        conf = q_values_to_confidence(q_values, allowed_indices)

        # 🔧 調整欄位 X 座標，徹底解決文字重疊問題
        col_action = x + 15
        col_conf   = x + 175  # 往左移一點
        col_mask   = x + 410  # 大幅往右移，讓出空間給 Q值 文字
        col_tech   = x + 510

        header_y = y + 30
        row_y = y + 65
        row_h = 28

        # 表頭
        surface.blit(fonts["header"].render("Action", True, colors["text"]), (col_action, header_y))
        surface.blit(fonts["header"].render("Probability & Q-Value", True, colors["cyan"]), (col_conf, header_y))
        surface.blit(fonts["header"].render("Mask Status", True, colors["text"]), (col_mask, header_y))
        surface.blit(fonts["header"].render("Tech-Tree", True, colors["text"]), (col_tech, header_y))

        # ✨ 全新排序邏輯：優先排 "OPEN" (可以做的)，然後才按機率高低排序
        sort_keys = []
        for i in range(len(valid_actions)):
            is_open = i in allowed_indices
            # 將 (是否合法, 機率, 原始索引) 打包
            sort_keys.append((is_open, conf[i], i))
            
        # 進行排序：優先比對 is_open (True 優先於 False)，再比對 conf (機率高優先)
        sort_keys.sort(key=lambda item: (item[0], item[1]), reverse=True)
        
        # 取出前 8 名排序好的動作索引
        order = [item[2] for item in sort_keys][:8]

        for rank, i in enumerate(order):
            act_id = valid_actions[i]
            name = ACTION_NAMES.get(act_id, f"Action {act_id}")

            is_open = i in allowed_indices
            pct = int(conf[i] * 100)
            q_val = q_values[i]

            yy = row_y + rank * row_h

            # 交錯底色
            bg = (16, 42, 48) if rank % 2 == 0 else (20, 50, 56)
            pygame.draw.rect(surface, bg, (x + 10, yy - 4, w - 20, row_h))

            text_color = colors["good"] if is_open else colors["gray"]

            # Action
            surface.blit(fonts["text"].render(name, True, text_color), (col_action, yy))

            # Confidence bar
            bar_x = col_conf
            bar_y = yy + 4
            bar_w = 60  # 🔧 將進度條長度縮減，配合文字長度
            bar_h = 14

            pygame.draw.rect(surface, (10, 28, 32), (bar_x, bar_y, bar_w, bar_h))
            # 即使被 LOCK，只要有算出版面機率，依然畫出進度條(灰色)
            pygame.draw.rect(surface, text_color, (bar_x, bar_y, int(bar_w * pct / 100), bar_h))

            # 機率與 Q 值文字
            display_text = f"{pct:>2}% (Q: {q_val:+.4f})"
            q_txt_surface = fonts["small"].render(display_text, True, text_color)
            surface.blit(q_txt_surface, (bar_x + bar_w + 8, yy))

            # Mask Status
            mask_text = "OPEN" if is_open else "LOCK"
            mask_color = colors["good"] if is_open else colors["bad"]
            surface.blit(fonts["small"].render(mask_text, True, mask_color), (col_mask, yy))

            # Tech Tree
            tech_text = "OK" if is_open else "BLOCK"
            tech_color = colors["good"] if is_open else colors["bad"]
            surface.blit(fonts["small"].render(tech_text, True, tech_color), (col_tech, yy))

    def draw_full_hud(surface, obs, brain_model, state_vector, allowed_indices, hidden_state, valid_actions):
        colors = ui_colors()
        fonts = get_fonts()
        surface.fill(colors["bg"])

        s = get_dashboard_snapshot(obs)
        HUD_HISTORY["minerals"].append(s["minerals"])
        HUD_HISTORY["vespene"].append(s["vespene"])

        device = next(brain_model.parameters()).device
        with torch.no_grad():
            st = torch.FloatTensor(np.array(state_vector)).unsqueeze(0).to(device)
            q_actions, _, _ = brain_model(st, hidden_state)
            q_values = q_actions.cpu().numpy()[0]

        LEFT = 20
        TOP = 20
        GAP = 10

        LEFT_W = 880
        RIGHT_W = 620
        TOP_H = 480

        # ===== 上排 =====
        game_rect = (LEFT, TOP, LEFT_W, TOP_H)
        network_rect = (LEFT + LEFT_W + GAP, TOP, RIGHT_W, TOP_H)

        # ===== 中排 =====
        mid_y = TOP + TOP_H + GAP
        mid_h = 160

        # ⭐ 一定要有這行（你現在缺這個）
        timer_w = 180
        timer_rect = (LEFT, mid_y, timer_w, mid_h)

        # 小資訊
        info_w = (LEFT_W - timer_w - GAP*2) // 2
        info_h = (mid_h - GAP) // 2

        x1 = LEFT + timer_w + GAP
        x2 = x1 + info_w + GAP

        info1_rect = (x1, mid_y, info_w, info_h)
        info2_rect = (x2, mid_y, info_w, info_h)
        info3_rect = (x1, mid_y + info_h + GAP, info_w, info_h)
        info4_rect = (x2, mid_y + info_h + GAP, info_w, info_h)

        # ===== 下排 =====
        timeline_y = mid_y + mid_h + GAP
        timeline_h = 140

        timeline_rect = (LEFT, timeline_y, LEFT_W, timeline_h)

        # ===== Current =====
        current_rect = (
            LEFT + LEFT_W + GAP,
            mid_y,
            RIGHT_W,
            timeline_y + timeline_h - mid_y
        )    

        # 1️⃣ 先畫外框
        draw_glow_rect(surface, network_rect, colors["panel"], colors["border"], 2)

        # 2️⃣ 畫標題
        surface.blit(fonts["header"].render("DRQN Network", True, colors["title"]),
                    (network_rect[0] + 12, network_rect[1] + 10))

        # 3️⃣ 畫內容
        sub_rect = pygame.Rect(
            network_rect[0] + 10,
            network_rect[1] + 40,
            network_rect[2] - 20,
            network_rect[3] - 50
        )

        if sub_rect.width > 0 and sub_rect.height > 0:
            sub_surface = surface.subsurface(sub_rect)

            draw_dual_head_network(
                sub_surface,
                brain_model,
                state_vector,
                allowed_indices,
                hidden_state
            )

        
            
        # 左上
        draw_feature_game_panel(surface, game_rect, obs, fonts, colors)
       
        # 中間資訊
        s = get_dashboard_snapshot(obs)

        draw_panel(surface, timer_rect, "Timer", fonts, colors)
        surface.blit(
            fonts["header"].render(format_loop_to_mmss(s["remain_loop"]), True, colors["cyan"]),
            (timer_rect[0] + 10, timer_rect[1] + 60)
        )

        draw_small_info(surface, info1_rect, "Marauders", s["marauders"], fonts, colors)

        # ⭐ 把 Minerals 放到 info2
        draw_small_info(
            surface,
            info2_rect,
            "Minerals",
            s["minerals"],
            fonts,
            colors,
            HUD_HISTORY["minerals"],
            colors["cyan"]
        )

        # ⭐ 把 Supply 放到 info3
        draw_small_info(surface, info3_rect, "Supply", f"{s['food_used']}/{s['food_cap']}", fonts, colors)

        draw_small_info(
            surface,
            info4_rect,
            "Gas",
            s["vespene"],
            fonts,
            colors,
            HUD_HISTORY["vespene"],
            colors["good"]
        )


        # 右下決策
        draw_current(surface, current_rect, q_values, allowed_indices, valid_actions, fonts, colors)


        # 底部 timeline
        draw_timeline(surface, timeline_rect, obs, fonts, colors)

        pygame.display.flip()

    def get_action_mask(target_obs, last_act_id=0): 
        """ 根據當前畫面狀態，回傳合法的 Action 索引列表 """
        player = target_obs.observation.player
        s_unit = target_obs.observation.feature_screen.unit_type
        s_player = target_obs.observation.feature_screen.player_relative
        workers = int(player.food_workers)
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
        allowed_acts.remove(16)
        if depots == 0 and 2 in allowed_acts: allowed_acts.remove(2)
            
        
                
        if techlabs == 0 and 18 in allowed_acts: allowed_acts.remove(18)
        if refineries >= 1 and 11 in allowed_acts: allowed_acts.remove(11)
        has_geyser = np.any(s_unit == 342) or np.any(s_unit == 341)
        if not has_geyser and 11 in allowed_acts: allowed_acts.remove(11)
        if (refineries == 0 or getattr(agent, 'gas_workers_assigned', 0) >= 2) and 42 in allowed_acts: 
            allowed_acts.remove(42)
        if supply_surplus >= 20 and 1 in allowed_acts: allowed_acts.remove(1)
        if barracks >= 3 and 2 in allowed_acts: allowed_acts.remove(2)
        if refineries > 0 and 11 in allowed_acts: 
            allowed_acts.remove(11)
        if player.idle_worker_count == 0 and 41 in allowed_acts: 
            allowed_acts.remove(41)
        if workers > 18 and 14 in allowed_acts: 
            allowed_acts.remove(14)
        if supply_surplus < 1:
            for act in [14, 16]: 
                if act in allowed_acts: allowed_acts.remove(act)
        if supply_surplus < 2 and 18 in allowed_acts: allowed_acts.remove(18)  # ← 加這行
        
        if minerals < 50:
            for act in [14, 34]: 
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
                # 📦 完整解包現在變成 11 個參數了
                state = torch.FloatTensor([step_data[0]]).to(device)
                action = torch.LongTensor([step_data[1]]).to(device)
                param_target = torch.LongTensor([step_data[2]]).to(device)
                reward = torch.FloatTensor([step_data[3]]).to(device)
                next_state = torch.FloatTensor([step_data[4]]).to(device)
                done = step_data[5]
                h_out, c_out = step_data[7]
                hidden_out = (h_out.to(device), c_out.to(device))
                gamma_mult = step_data[8]
                next_allowed = step_data[9]
                
                # ✨ 核心修復 3A：拿出真正的單步成功鐵證
                step_success_flag = step_data[10]

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
                
                # ✨ 核心修復 3B：只有當「這一步的滑鼠點擊真正被引擎接受 (success == 1.0)」時，
                # 我們才允許神經網路去記住這個 p_id！不再被未來總分 (reward) 給騙了！
                if step_success_flag == 1.0:
                    param_loss = nn.CrossEntropyLoss()(q_params, param_target)
                
                total_loss += (action_loss + 0.5 * param_loss)
                hidden = next_hidden
                
        # 4. 根據這批連續記憶的總誤差，更新神經網路
        if valid_sequences > 0 and isinstance(total_loss, torch.Tensor):
            # 取平均避免梯度爆炸
            mean_loss = total_loss / (valid_sequences * seq_len) 
            mean_loss.backward()
            optimizer.step()

    # ✨ 全新路線：解除星海視窗鎖定，改為可用滑鼠自由縮放的正常視窗
   # ✨ 全新路線：保持視窗模式，並由程式自動精準校正大小
    def resize_sc2_to_grid():
        import ctypes
        import time

        # 定義 Windows 矩形結構，用來計算邊框厚度
        class RECT(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long),
                        ("top", ctypes.c_long),
                        ("right", ctypes.c_long),
                        ("bottom", ctypes.c_long)]

        print("⏳ 等待星海視窗出現，準備將畫面精準縮放...")
        for _ in range(30): 
            time.sleep(1)
            
            # 支援中英文雙語系視窗尋找
            hwnd = ctypes.windll.user32.FindWindowW(None, "《星海爭霸II》")
            if not hwnd:
                hwnd = ctypes.windll.user32.FindWindowW(None, "StarCraft II")
                
            if hwnd:
                # 1. 取得視窗目前的樣式 (保留原生的標題列)
                style = ctypes.windll.user32.GetWindowLongW(hwnd, -16)
                
                # 2. 設定我們想要的「內部遊戲畫面」大小 (剛好等於你的格子)
                target_w = 856
                target_h = 423
                
                # 3. 讓 Windows 幫我們計算：如果內部要 856x423，那加上標題列跟邊框後，外部視窗總共要多大？
                rect = RECT(0, 0, target_w, target_h)
                ctypes.windll.user32.AdjustWindowRect(ctypes.byref(rect), style, False)
                
                final_w = rect.right - rect.left
                final_h = rect.bottom - rect.top
                
                # 4. 強制改變視窗大小 (保留原本的位置，只改大小)
                # SWP_NOMOVE (0x0002) 代表不移動位置；SWP_NOZORDER (0x0004) 代表不改變圖層
                ctypes.windll.user32.SetWindowPos(hwnd, 0, 0, 0, final_w, final_h, 0x0002 | 0x0004)
                
                print(f"✅ 星海畫面已精準設定為 {target_w}x{target_h}！你可以抓著標題列自由拖曳了！")
                break

    # 🚀 在啟動星海前，派執行緒去背景等待校正
    threading.Thread(target=resize_sc2_to_grid, daemon=True).start()

    # 原本啟動環境的程式碼保持不動
    with sc2_env.SC2Env(
        map_name="Simple96",
        players=[sc2_env.Agent(sc2_env.Race.terran),sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy)],
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
            failed_actions = 0
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
                        
                        
                # ==========================================
                # ==========================================
                # ✨ 核心防彈機制：在一開始就先給定預設發呆動作！
                sc2_action = actions.FUNCTIONS.no_op() 

                # 這裡是你原本的呼叫邏輯 (無論你有沒有包在 if 裡面都沒關係了)
                try:
                    sc2_action = agent.get_action(obs, a_id, parameter=p_id)
                except Exception as e:
                    print(f"⚠️ 執行動作 {a_id} 時發生錯誤: {e}，自動轉為發呆 (no_op)")
                    sc2_action = actions.FUNCTIONS.no_op()

                # 執行動作並進入下一幀
                obs_list = env.step([sc2_action, actions.FUNCTIONS.no_op()])
                next_obs = obs_list[0]
                next_allowed_indices = get_action_mask(next_obs, a_id)

                # ... 在你的訓練迴圈內 ...

                if RENDER_UI:
                    # 每一幀都要處理事件，防止系統判定視窗無回應
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return
                            
                    # 只有繪圖才降頻
                    if train_step_counter % 5 == 0:
                        draw_full_hud(
                            screen, obs, brain_model, state,
                            allowed_indices, hidden_state, VALID_ACTIONS
                        )
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
                    if agent.locked_action is not None:
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
                current_real_count = min(current_real_count, 5)

                
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

                step_reward = 0.0
                
                # 1. 里程碑追蹤 (不直接給分，用於終局結算)
                if "depot" not in achieved_milestones and current_depots >= 1:
                    achieved_milestones.add("depot")
                if "barracks" not in achieved_milestones and current_barracks >= 1:
                    achieved_milestones.add("barracks")
                if "refinery" not in achieved_milestones and current_refineries >= 1:
                    achieved_milestones.add("refinery")
                if "techlab" not in achieved_milestones and current_techlabs >= 1:
                    achieved_milestones.add("techlab")

                # 2. 掠奪者產量追蹤 (不直接給分，用於終局結算)
                if current_real_count > getattr(agent, 'last_target_count', 0):
                    agent.last_target_count = current_real_count
                    print(f"🎯 掠奪者產出！目前進度: {current_real_count}/5")

                # 3. 無效動作懲罰 (Process Penalty)
                # 為了讓訓練穩定，過程中的錯誤還是給予微小的懲罰，促使它尋找合法座標
                if current_action_success == 0.0:
                    step_reward -= 0.01
                    # 追蹤整局的失敗次數，用於終局結算
                    failed_actions += 1

                # 4. 終局判定與四大維度結算
                current_loop = int(next_obs.observation.game_loop[0])
                MAX_LOOP = 13440.0
                done = next_obs.last() or current_loop >= MAX_LOOP or current_real_count >= 5

                if done:
                    # 維度 1：時間分數 (權重 0.6) - 越快完成分數越高
                    # 如果沒造出 5 隻，時間分數為 0
                    time_score = 0.0
                    if current_real_count >= 5:
                        time_score = 1.0 - (current_loop / MAX_LOOP)
                        time_score = max(0.0, time_score) # 確保不為負
                    
                    # 維度 2：里程碑分數 (權重 0.2) - 蓋了幾種關鍵建築
                    milestone_score = len(achieved_milestones) / 4.0
                    
                    # 維度 3：掠奪者數量分數 (權重 0.1)
                    marauder_score = min(current_real_count / 5.0, 1.0)
                    
                    # 維度 4：操作效率分數 (權重 0.1) - 容忍一定比例的錯誤，過多則扣分
                    # 假設整局平均操作 500 次，失敗 50 次以內算滿分 1.0
                    efficiency_score = max(0.0, 1.0 - (failed_actions / 100.0))
                    
                    # 💥 最終總分計算 (0.6, 0.2, 0.1, 0.1)
                    raw_final_score = (
                        (0.6 * time_score) + 
                        (0.2 * milestone_score) + 
                        (0.1 * marauder_score) + 
                        (0.1 * efficiency_score)
                    )
                    
                    # 將 0.0 ~ 1.0 的分數，映射到 -1.0 ~ 1.0 的區間
                    # 如果連一隻都沒生出來，分數一定會是負的
                    if current_real_count >= 5:
                        # 成功時，分數保底 0.5 以上，速度越快越高分
                        step_reward += 0.5 + (raw_final_score * 0.5) 
                    else:
                        # 失敗時，根據蓋房子的進度給予 -1.0 到 0.0 之間的懲罰
                        step_reward += (raw_final_score * 2.0) - 1.0
                        
                    print(f"🏁 終局結算: 時間={time_score:.2f}, 建築={milestone_score:.2f}, 產量={marauder_score:.2f}, 效率={efficiency_score:.2f} => 總分={step_reward:.4f}")

                # 我們不再每一幀都用 tanh 壓縮，而是保持 step_reward 的原值
                # 只有在終局時，step_reward 才會有較大的數值
                scaled_reward = np.clip(step_reward, -1.0, 1.0) 
                episode_reward += scaled_reward
                #episode_reward += scaled_reward  # 累計這局的總分（用於 CSV 記錄與排行榜）
                if done:
                    final_reward = scaled_reward  # 只取終局幀，天然在[-1,+1]，給教授看、寫CSV

                if step_reward != 0:
                    print(f"DEBUG: raw={step_reward:.4f}, tanh後={scaled_reward:.4f}, 累計={episode_reward:.4f}")

                # ==========================================
                # 第六步：取得下一幀狀態，並打包成訓練用的 Transition
                # ==========================================
                # ==========================================
                # 第六步：取得下一幀狀態，並打包成訓練用的 Transition
                # ==========================================
                # 將這一幀的狀態打包
                next_time_loop = int(next_obs.observation.game_loop[0])
                next_state = get_state_vector(
                    next_obs, p_id, 18, a_id, current_action_success, next_time_loop)

                # ✨ 核心修復 1：在包裹中多塞一個 current_action_success 當作第 10 個參數
                current_transition = (state, action_index, p_id - 1, scaled_reward, next_state, done, hidden_state, next_hidden_state, next_allowed_indices, current_action_success)
                
                # 依然維持只有成功的動作才放入 N-Step 緩衝
                if current_action_success == 1.0:
                    n_step_buffer.append(current_transition)
                else:
                    failed_transition = (state, action_index, p_id - 1, scaled_reward, next_state, done, hidden_state, next_hidden_state, 0.0, next_allowed_indices, current_action_success)
                    episode_memory.append(failed_transition)
                    # (下方清空 n_step_buffer 的邏輯維持不變... 但打包 n_transition 時要記得補上這個欄位！)
                    
                    # 並且清空當前的緩衝區，因為這條「順暢的連續技」已經被失敗打斷了！
                    # 如果不清空，前面成功的步驟就會誤以為這個失敗是它造成的
                    while len(n_step_buffer) > 0:
                        fail_n = len(n_step_buffer)
                        n_reward = sum([n_step_buffer[i][3] * (GAMMA ** i) for i in range(fail_n)])
                        n_state = n_step_buffer[0][0]
                        n_action = n_step_buffer[0][1]
                        n_p_id = n_step_buffer[0][2]          
                        n_next_state = n_step_buffer[-1][4]   
                        n_done = n_step_buffer[-1][5]
                        n_hidden_in = (n_step_buffer[0][6][0].cpu(), n_step_buffer[0][6][1].cpu())
                        n_hidden_out = (n_step_buffer[-1][7][0].cpu(), n_step_buffer[-1][7][1].cpu())
                        # ... 前面維持不變
                        n_next_allowed = n_step_buffer[-1][8]
                        n_success_flag = n_step_buffer[0][9] # ✨ 拿出這一步當時到底有沒有成功
                        
                        # ✨ 核心修復 2：把 n_success_flag 放進包裹的最後面
                        n_transition = (n_state, n_action, n_p_id, n_reward, n_next_state, n_done, n_hidden_in, n_hidden_out, GAMMA ** fail_n, n_next_allowed, n_success_flag)
                        
                        episode_memory.append(n_transition)
                        n_step_buffer.popleft()

                # 更新觀測值與狀態，準備進入下一幀
                '''obs = next_obs
                state = next_state
                hidden_state = (next_hidden_state[0].detach(), next_hidden_state[1].detach())
                last_action_id = a_id
                last_action_success = current_action_success'''



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
                    # ... 前面維持不變
                    n_next_allowed = n_step_buffer[-1][8]
                    n_success_flag = n_step_buffer[0][9] # ✨ 拿出這一步當時到底有沒有成功
                    
                    # ✨ 核心修復 2：把 n_success_flag 放進包裹的最後面
                    n_transition = (n_state, n_action, n_p_id, n_reward, n_next_state, n_done, n_hidden_in, n_hidden_out, GAMMA ** N_STEP, n_next_allowed, n_success_flag)
                    
                    episode_memory.append(n_transition)
                    n_step_buffer.popleft()

                # 3. 遊戲結束時，強制清空並結算緩衝區裡剩下的尾巴
                # 3. 遊戲結束時，強制清空並結算緩衝區裡剩下的尾巴
                # 3. 遊戲結束時，強制清空並結算緩衝區裡剩下的尾巴
                if done:
                    while len(n_step_buffer) > 0:
                        end_n = len(n_step_buffer)  # ✅ 統一叫 end_n
                        n_reward = sum([n_step_buffer[i][3] * (GAMMA ** i) for i in range(end_n)]) # ✅ 這裡也是 end_n
                        
                        n_state = n_step_buffer[0][0]
                        n_action = n_step_buffer[0][1]
                        n_p_id = n_step_buffer[0][2]          
                        n_next_state = n_step_buffer[-1][4]   
                        n_done = n_step_buffer[-1][5]
                        
                        n_hidden_in = (n_step_buffer[0][6][0].cpu(), n_step_buffer[0][6][1].cpu())
                        n_hidden_out = (n_step_buffer[-1][7][0].cpu(), n_step_buffer[-1][7][1].cpu())
                        n_next_allowed = n_step_buffer[-1][8]
                        n_success_flag = n_step_buffer[0][9] 
                        
                        n_transition = (n_state, n_action, n_p_id, n_reward, n_next_state, n_done, n_hidden_in, n_hidden_out, GAMMA ** end_n, n_next_allowed, n_success_flag)
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
                            
                        print(f"🏆 菁英榜更新！目前收錄 {len(success_memory)} 局，歷史最短時間: {success_memory[0][0]:.0f} 幀")
                        
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
                # ✨ 核心修復：直接將真實決策的 p_id 餵給神經網路當作記憶！
                current_block = p_id

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