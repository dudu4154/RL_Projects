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
        # 共同特徵層
        self.common = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        # 動作頭：輸出 43 個動作的 Q 值
        self.action_head = nn.Linear(64, action_size)
        
        # 參數頭：輸出 64 個網格位置的 Q 值
        self.param_head = nn.Linear(64, 64) 

    def forward(self, x):
        x = self.common(x)
        # 同時回傳動作與參數的預測值
        return self.action_head(x), self.param_head(x)
    
    
    
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
    action_size = 48
    batch_size = 64  # ✨ 新增：每次訓練抓取的樣本數
    train_step_counter = 0
    brain_model = QNetwork(state_size, action_size)
    optimizer = optim.Adam(brain_model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()
    memory = deque(maxlen=100000)
    
    gamma = 0.99
    logger = TrainingLogger() # 使用 TrainingLogger 紀錄產量
    learn_min = 0.01
    last_action_id = 0   
    current_block = 1    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_info = REWARD_CONFIG[18] 
    CURRENT_TRAIN_TASK = 18
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True) # 確保資料夾一定存在
    model_path = os.path.join(log_dir, "dqn_model.pth")
    if os.path.exists(model_path):
        brain_model = QNetwork(state_size, action_size).to(device)
        print("✅ 載入成功！接續之前的記憶繼續訓練...")

    epsilon = 1.00; epsilon_decay = 0.995; gamma = 0.99 
    def train_model():
        if len(memory) < batch_size:
            return
        
        # 從記憶庫隨機抽樣
        mini_batch = random.sample(memory, batch_size)
        
        # 準備訓練資料 (轉換為 Tensor)
        states = torch.FloatTensor(np.array([m[0] for m in mini_batch])).to(device)
        actions_t = torch.LongTensor(np.array([m[1] for m in mini_batch])).to(device)
        rewards = torch.FloatTensor(np.array([m[2] for m in mini_batch])).to(device)
        next_states = torch.FloatTensor(np.array([m[3] for m in mini_batch])).to(device)
        dones = torch.FloatTensor(np.array([m[4] for m in mini_batch])).to(device)

        # 計算當前的 Q 值
        current_q_actions, _ = brain_model(states)
        q_value = current_q_actions.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # 計算目標 Q 值 (Bellman Equation)
        with torch.no_grad():
            next_q_actions, _ = brain_model(next_states)
            max_next_q = next_q_actions.max(1)[0]
            target_q = rewards + (1 - dones) * gamma * max_next_q

        # 計算損失並更新權重
        loss = criterion(q_value, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with sc2_env.SC2Env(
        map_name="Simple96",
        players=[sc2_env.Agent(sc2_env.Race.terran),sc2_env.Agent(sc2_env.Race.terran)], #sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy)
        agent_interface_format=sc2_env.AgentInterfaceFormat(
            feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64), use_raw_units=False),
        step_mul=16, realtime=False
        
    ) as env:
        
        for ep in range(3000):
            agent = ProductionAI()
            obs_list = env.reset()
            obs = obs_list[0]
            total_reward = 0
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
                # ------------------------------------------------------------------

                # 初始允許的動作
                allowed_actions = [1, 2, 11, 14, 16, 18, 34, 41, 42, 44, 45]
                # B. 補給站鎖定：如果剩餘人口大於 16 (非常夠用)，拔掉 Action 1
                supply_surplus = player.food_cap - player.food_used
                if supply_surplus >= 16 and 1 in allowed_actions:
                    allowed_actions.remove(1)

                # C. 資源不足鎖定：沒錢就不要想著蓋兵營 (需要 150 礦)
                if float(player.minerals) < 150 and 2 in allowed_actions:
                    allowed_actions.remove(2)
                    
                # D. 兵營數量鎖定：如果兵營已經 >= 3 座，絕對不准再蓋兵營
                if current_barracks >= 3 and 2 in allowed_actions:
                    allowed_actions.remove(2)

                if current_refineries >= 2:
                    #if 42 in allowed_actions: allowed_actions.remove(42)
                    if 11 in allowed_actions: allowed_actions.remove(11)

                # C. 資源不足鎖定：沒錢就不要想著蓋兵營 (需要 150 礦)
                if player.minerals < 150 and 2 in allowed_actions:
                    allowed_actions.remove(2)

                if current_barracks == 0:
                    for act in [16, 18, 34]:
                        if act in allowed_actions: allowed_actions.remove(act)
                
                # 2. 如果沒有科技實驗室，絕對不准「造掠奪者」
                if current_techlabs == 0:
                    if 18 in allowed_actions: allowed_actions.remove(18)
                    
                # 3. 如果連瓦斯廠都沒有，絕對不准派工兵去「採瓦斯」
                if current_refineries == 0:
                    if 42 in allowed_actions: allowed_actions.remove(42)
                
                # F. ✨ 工兵數量鎖定：一個基地最多只要 22 隻工兵，超過絕對不准再造！
                if float(player.food_workers) >= 22 and 14 in allowed_actions:
                    allowed_actions.remove(14)
                # ==========================================

                # 選擇動作 (Epsilon-Greedy)
                if random.random() <= epsilon:
                    a_id = random.choice(allowed_actions)
                    p_id = random.randint(1, 64)
                else:
                    with torch.no_grad():
                        q_actions, q_params = brain_model(state_t.unsqueeze(0))
                        
                        # 使用動態更新後的 allowed_actions 來生成遮罩
                        mask = torch.full_like(q_actions, float('-inf'))
                        mask[0, allowed_actions] = 0 
                        
                        masked_q_actions = q_actions + mask
                        a_id = torch.argmax(masked_q_actions).item() 
                        p_id = torch.argmax(q_params).item() + 1

                # 2. 選擇動作 (Epsilon-Greedy)
                if random.random() <= epsilon:
                    # 隨機探索：直接從清單抽樣
                    a_id = random.choice(allowed_actions) 
                    p_id = random.randint(1, 64)
                else:
                    # 模型決策：實施動作遮罩 (Action Masking)
                    with torch.no_grad():
                        q_actions, q_params = brain_model(state_t.unsqueeze(0))
                        mask = torch.full_like(q_actions, float('-inf'))
                        mask[0, allowed_actions] = 0 
                        masked_q_actions = q_actions + mask
                        a_id = torch.argmax(masked_q_actions).item() 
                        p_id = torch.argmax(q_params).item() + 1
                # ==========================================
                # (D) 執行動作與環境互動
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

                # 5. ⚔️ 漸進式產量獎勵
                # (請確保迴圈外有宣告 last_marine_count = 0, last_target_count = 0)
                if marine_count > getattr(agent, 'last_marine_count', 0):
                    step_reward += 50.0  # 造出陸戰隊給小獎
                    agent.last_marine_count = marine_count
                    
                if current_real_count > getattr(agent, 'last_target_count', 0):
                    step_reward += 500.0 # 造出掠奪者給大獎
                    agent.last_target_count = current_real_count
                    print(f"🎯 產出掠奪者！目前數量: {current_real_count}")

                # 6. 🏁 終局結算：規則永遠不變
                current_loop = int(next_obs.observation.game_loop[0])
                done = next_obs.last() or current_loop >= 13440 or current_real_count >= 5
                
                if done:
                    if current_real_count >= 5:
                        step_reward += 3000.0 # 完美達成目標
                        print(f"✅ 任務成功！產出 5 隻掠奪者 (耗時: {current_loop} 幀)")
                        if current_loop <= 6720:
                            step_reward += 2000.0 # 5分鐘內極速加碼
                    else:
                        step_reward -= 1000.0 # 沒達成目標統一扣分
                        print(f"❌ 任務失敗：時間到未達成目標，結算懲罰")

                # ✅ 將分數縮小 1000 倍存入大腦，防止梯度爆炸
                scaled_reward = step_reward / 10000.0
                # ==========================================
                # 取得下一步的時間
                next_time_loop = int(next_obs.observation.game_loop[0])
                
                # (2) 存入 AI 的記憶庫 (補上 next_time_loop)
                next_state = get_state_vector(next_obs, getattr(agent, 'active_parameter', 1), 18, a_id, current_action_success, next_time_loop)
                # 這樣才是把除過 10000 的 scaled_reward 丟給大腦！
                memory.append((state, a_id, scaled_reward, next_state, done))
                
                if len(memory) > batch_size:
                    train_model()

                obs = next_obs
                total_reward += step_reward
                last_action_id = a_id
                last_action_success = current_action_success # ✨ 將成功狀態繼承給下一步
                current_block = getattr(agent, 'active_parameter', 1)

                
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
            epsilon = max(learn_min, epsilon * epsilon_decay)
            torch.save(brain_model.state_dict(), model_path)

if __name__ == "__main__":
    from absl import app
    app.run(main)