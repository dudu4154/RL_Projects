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
    # [Tier 0] 基礎經濟
    14: {"name": "SCV", "id": 45, "first": 50.0, "repeat": 10.0, "pixel": 15.0, "req": ['depot', 'refinery']},
    
    # [Tier 1] 兵營系列
    16: {"name": "陸戰隊", "id": 48, "first": 100.0, "repeat": 30.0, "pixel": 10.0, "req": ['depot', 'barracks']},
    17: {"name": "死神",       "id": 49,  "first": 200.0,  "repeat": 50.0,  "pixel": 15.0,"req": ['depot', 'barracks']},
    18: {"name": "掠奪者", "id": 51, "first": 500.0, "repeat": 150.0, "pixel": 12.0, "req": ['depot', 'barracks', 'refinery', 'techlab']}, # ✨ 下調 pixel 從 22.0 -> 12.0
    19: {"name": "幽靈特務",   "id": 50,  "first": 800.0,  "repeat": 200.0, "pixel": 15.0,"req": ['depot', 'barracks', 'refinery', 'techlab']}, # 需幽靈學院
    
    # [Tier 2] 軍工廠 (Factory)
    20: {"name": "惡狼",       "id": 53,  "first": 400.0,  "repeat": 100.0, "pixel": 25.0},
    22: {"name": "寡婦詭雷",   "id": 498, "first": 400.0,  "repeat": 100.0, "pixel": 12.0},
    24: {"name": "颶風飛彈車", "id": 692, "first": 600.0,  "repeat": 150.0, "pixel": 40.0},
    # [Tier 2] 軍工廠系列
    23: {"name": "攻城坦克", "id": 33, "first": 1000.0, "repeat": 250.0, "pixel": 150.0,"req": ['depot', 'barracks', 'refinery', 'factory', 'techlab']},
    21: {"name": "戰狼",       "id": 484, "first": 800.0,  "repeat": 200.0, "pixel": 30.0,"req": ['depot', 'barracks', 'refinery', 'factory', 'techlab']},  # 需兵工廠
    25: {"name": "雷神",       "id": 52,  "first": 1500.0, "repeat": 400.0, "pixel": 200.0,"req": ['depot', 'barracks', 'refinery', 'factory', 'techlab']}, # 需科技室+兵工廠
    
    # [Tier 3] 星際港 (Starport)
    26: {"name": "維京戰機",   "id": 34,  "first": 800.0,  "repeat": 200.0, "pixel": 45.0, "req": ['depot', 'barracks', 'refinery', 'factory', 'starport', 'techlab', 'fusion_core']},
    27: {"name": "醫療艇",     "id": 54,  "first": 800.0,  "repeat": 200.0, "pixel": 50.0, "req": ['depot', 'barracks', 'refinery', 'factory', 'starport', 'techlab', 'fusion_core']},
    28: {"name": "解放者",     "id": 689, "first": 1000.0, "repeat": 250.0, "pixel": 60.0, "req": ['depot', 'barracks', 'refinery', 'factory', 'starport', 'techlab', 'fusion_core']},
    29: {"name": "渡鴉",       "id": 56,  "first": 1200.0, "repeat": 300.0, "pixel": 40.0, "req": ['depot', 'barracks', 'refinery', 'factory', 'starport', 'techlab', 'fusion_core']},  # 需科技室
    31: {"name": "女妖轟炸機", "id": 55,  "first": 1200.0, "repeat": 300.0, "pixel": 55.0, "req": ['depot', 'barracks', 'refinery', 'factory', 'starport', 'techlab', 'fusion_core']},  # 需科技室
    30: {"name": "戰巡艦", "id": 57, "first": 2500.0, "repeat": 600.0, "pixel": 250.0, "req": ['depot', 'barracks', 'refinery', 'factory', 'starport', 'techlab', 'fusion_core']},
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
class TrainingLogger:
    def __init__(self):
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        self.filename = os.path.join(log_dir, f"dqn_training_log_{int(time.time())}.csv")
        # ✨ 加入 encoding='utf-8-sig'
        with open(self.filename, mode='w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Task", "Epsilon", "Total_Reward", "Barracks", "TechLabs", "Marauders", "End_Loop", "Reason", "Is_Bottom_Right"])

    def log_episode(self, ep, task_name, eps, reward, b_cnt, t_cnt, m_cnt, loop, reason, location):
        if hasattr(reward, "item"): 
            reward = reward.item()
        
        # ✨ 加入 encoding='utf-8-sig'
        with open(self.filename, mode='a', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow([ep, task_name, f"{eps:.3f}", int(reward), b_cnt, t_cnt, m_cnt, loop, reason, location])

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
    
    
    
def get_state_vector(obs, current_block, target_project_id, last_action_id):
    player = obs.observation.player
    m_unit = obs.observation.feature_minimap.unit_type
    m_relative = obs.observation.feature_minimap.player_relative
    available = obs.observation.available_actions

    # 相機座標
    camera_layer = obs.observation.feature_minimap.camera
    y_cam, x_cam = camera_layer.nonzero()
    cam_x, cam_y = (x_cam.mean() / 64.0, y_cam.mean() / 64.0) if x_cam.any() else (0.5, 0.5)

    # 選擇狀態
    is_scv_selected = 1.0 if any(u.unit_type == production_ai.SCV_ID for u in obs.observation.multi_select) else 0.0
    is_cc_selected = 1.0 if len(obs.observation.single_select) > 0 and obs.observation.single_select[0].unit_type == production_ai.COMMAND_CENTER_ID else 0.0
    
    # 用「動作可用性」偵測科技室 (這不會報 AttributeError)
    can_train_marauder = 1.0 if actions.FUNCTIONS.Train_Marauder_quick.id in available else 0.0

    state_list = [
        player.food_workers / 16,
        player.minerals / 1000,
        player.vespene / 500,
        player.food_used / 50,
        player.food_cap / 50,
        np.sum((m_unit == production_ai.BARRACKS_ID) & (m_relative == 1)),
        np.sum((m_unit == production_ai.SUPPLY_DEPOT_ID) & (m_relative == 1)),
        np.sum((m_unit == production_ai.REFINERY_ID) & (m_relative == 1)),
        np.sum((m_unit == production_ai.BARRACKS_TECHLAB_ID) & (m_relative == 1)),
        np.sum((m_unit == 27) & (m_relative == 1)), 
        np.sum((m_unit == 28) & (m_relative == 1)), 
        np.sum((m_unit == 30) & (m_relative == 1)), 
        current_block / 64.0,
        is_scv_selected, 
        is_cc_selected,
        cam_x, 
        cam_y,
        can_train_marauder,
        target_project_id / 40.0,
        last_action_id / 48.0
    ]
    return state_list

# =========================================================
# 🎮 訓練主程式
# =========================================================
def main(argv):
    del argv
    state_size = 20 
    action_size = 48
    train_step_counter = 0
    brain_model = QNetwork(state_size, action_size)
    optimizer = optim.Adam(brain_model.parameters(), lr=0.0005) 
    criterion = nn.MSELoss()
    memory = deque(maxlen=100000) 
    logger = TrainingLogger() # 使用 TrainingLogger 紀錄產量
    learn_min = 0.01
    last_action_id = 0   
    current_block = 1    
    
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True) # 確保資料夾一定存在
    model_path = os.path.join(log_dir, "dqn_model.pth")
    if os.path.exists(model_path):
        brain_model.load_state_dict(torch.load(model_path))
        print("✅ 載入成功！接續之前的記憶繼續訓練...")

    epsilon = 1.00; epsilon_decay = 0.999; gamma = 0.99 

    with sc2_env.SC2Env(
        map_name="Simple96",
        players=[sc2_env.Agent(sc2_env.Race.terran), sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy)],
        agent_interface_format=sc2_env.AgentInterfaceFormat(
            feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64), use_raw_units=False),
        step_mul=16, realtime=False
    ) as env:
        for ep in range(3000):
            CURRENT_TRAIN_TASK = 18
            task_info = REWARD_CONFIG[CURRENT_TRAIN_TASK]
            print(f"🚀 Episode {ep+1} | 訓練任務：{task_info['name']}")
            last_action_id = 0
            hands = ProductionAI() 
            obs_list = env.reset() 
            obs = obs_list[0]
            next_obs = obs  # ✨ 關鍵修正：初始化 next_obs 避免 UnboundLocalError
            last_action_id = 0   # ✨ 初始化上一步動作
            current_block = 1    # ✨ 初始化當前區塊
            hands = ProductionAI()
            # 初始化追蹤變數
            
            rewarded_depots = 0     # 【新增】紀錄已給分過的補給站數量
            last_d_pixels = 0
            scv_reward_count = 0
            has_rewarded_barracks = False 
            has_rewarded_techlab = False  
            has_rewarded_home = False # 【新增】一次性回家獎勵旗標
            has_rewarded_control_group = False
            
            # 預設動作與參數
            a_id = 40; p_id = 1 
            # 初始化回合變數
            last_target_count = 0 
            total_reward = 0.0
            last_vespene = 0
            off_screen_steps = 0
            milestones = {
                'depot': False, 'barracks': False, 'techlab': False, 
                'refinery': False, 'factory': False, 'starport': False, 
                'fusion_core': False, 'first_unit': False, 'cc_destroyed': False
            }

            while True:
                actual_id = hands.locked_action if hands.locked_action is not None else a_id
                obs_data = next_obs.observation 
                next_m_unit = obs_data.feature_minimap[features.MINIMAP_FEATURES.unit_type.index]
                next_m_relative = obs_data.feature_minimap[features.MINIMAP_FEATURES.player_relative.index]
                next_s_unit = obs_data.feature_screen[features.SCREEN_FEATURES.unit_type.index]
                next_s_relative = obs_data.feature_screen[features.SCREEN_FEATURES.player_relative.index]
                next_state = get_state_vector(next_obs, current_block, CURRENT_TRAIN_TASK, actual_id)
                # --- 1. 取得當前狀態與選擇動作 ---
                current_block = getattr(hands, 'active_parameter', 1)
                player = obs_data.player
                state = get_state_vector(obs, current_block, CURRENT_TRAIN_TASK, last_action_id)
                state_t = torch.FloatTensor(np.array(state))
                # Epsilon-Greedy 選擇 (a_id 決定做什麼，p_id 決定在哪做)
                if random.random() <= epsilon:
                    a_id = random.choice([1,2,11,14,16,18,34,41,42,47])#random.randint(1, 45) 
                    p_id = random.randint(1, 64)
                else:
                    with torch.no_grad():
                        q_actions, q_params = brain_model(state_t.unsqueeze(0))
                        a_id = torch.argmax(q_actions).item()
                        p_id = torch.argmax(q_params).item() + 1

                # --- 2. 執行單一動作 ---
                sc2_action = hands.get_action(obs, a_id, parameter=p_id)
                obs_list = env.step([sc2_action, actions.FUNCTIONS.no_op()])
                next_obs = obs_list[0]
                
                # 確定這一步實際執行的動作 (考慮鎖定機制)
                actual_id = hands.locked_action if hands.locked_action is not None else a_id
                # --- 5. 狀態更新與存入記憶 ---
                updated_block = getattr(hands, 'active_parameter', 1)
                
                # ✨ 修正：補上第四個參數 actual_action_id
                actual_action_id = hands.locked_action if hands.locked_action is not None else a_id
                next_state = get_state_vector(next_obs, updated_block, CURRENT_TRAIN_TASK, actual_action_id)
                # --- 【關鍵修正：變數定義必須移到最前面】 ---
                
                # --- 3. 數據計算與紀錄 (現在變數已定義，不會報錯了) ---
                curr_b_count = np.sum((next_m_unit == production_ai.BARRACKS_ID) & (next_m_relative == 1))
                
                # 呼叫紀錄器將每一步寫入 terran_log
                hands.collector.log_step(
                    obs_data.game_loop,        # Time
                    obs_data.player.minerals, 
                    obs_data.player.vespene,
                    obs_data.player.food_workers, 
                    16, curr_b_count, actual_id
                )
                # --- 4. 獎勵判定與印出資訊 ---
                
                step_reward = -0.01 

                if train_step_counter % 10 == 0:
                    print(f"Episode {ep+1} | 執行動作: {a_id} | 參數: {p_id} | 礦石: {obs_data.player.minerals}")
                # 【修正】計算掠奪者數量 (原本代碼漏掉這段，會導致 NameError)
            
                # 偵測選取狀態
                is_scv_selected = False
                is_cc_selected = False
                if len(obs_data.single_select) > 0:
                    u_type = obs_data.single_select[0].unit_type
                    if u_type == production_ai.SCV_ID: is_scv_selected = True
                    if u_type == production_ai.COMMAND_CENTER_ID: is_cc_selected = True
                elif len(obs_data.multi_select) > 0:
                    if any(u.unit_type == production_ai.SCV_ID for u in obs_data.multi_select):
                        is_scv_selected = True

                
                
                # 現在可以安全計算 vespene_diff 了
                vespene_diff = player.vespene - last_vespene
                if vespene_diff > 0:
                    step_reward += vespene_diff * 0.5 
                last_vespene = player.vespene
                
                # 2. 補給站限建令 (修正 NameError: a_id)
                if a_id == 1: 
                    if player.food_cap - player.food_used > 20:
                        step_reward -= 10.0
                        print(f"🏚️ 補給過剩（{player.food_cap - player.food_used}），禁止洗分，扣 5 分")
                # --- [新增：空閒工兵懲罰] ---
                # 從 player 數據中直接獲取空閒工兵數量
                idle_workers = player.idle_worker_count
                
                if idle_workers > 0:
                    # 每一隻空閒工兵每步扣 0.01 分 (與時間懲罰相同強度)
                    idle_penalty = idle_workers * 0.01
                    step_reward -= idle_penalty
                    
                    # 為了方便觀察，可以每 50 步印一次警告
                    if train_step_counter % 50 == 0:
                        print(f"👷 警告：有 {idle_workers} 隻工兵正在發呆！扣除 {idle_penalty:.2f} 分")
                    # --- [正向里程碑獎勵系統 - 任務關聯版] ---
                
                task_cfg = REWARD_CONFIG.get(CURRENT_TRAIN_TASK)
                allowed_tech = task_cfg.get("req", []) # 取得當前任務允許的科技
                # --- [修正版：主堡存活判定] ---
                # 4. 主堡保護 (遊戲 500 幀後生效)
                m_unit = obs_data.feature_minimap.unit_type
                m_relative = obs_data.feature_minimap.player_relative
                cc_exists = np.any((m_unit == production_ai.COMMAND_CENTER_ID) & (m_relative == 1))
                if obs_data.game_loop[0] > 500 and not cc_exists and not milestones['cc_destroyed']:
                    step_reward -= 500
                    milestones['cc_destroyed'] = True
                    print("💥 【警告】主堡被拆除！嚴重懲罰 -500")
                # A. 補給站里程碑 (只有在 req 包含 depot 時才給分)
                if 'depot' in allowed_tech:
                    if obs_data.player.food_cap > 15 and not milestones['depot']:
                        step_reward += 50.0; milestones['depot'] = True
                        print(f"🏠 【人口突破】補給站完工，獎勵 +50")

                # B. 科技建築里程碑 (加入 key in allowed_tech 判定)
                tech_buildings = [
                    (production_ai.BARRACKS_ID, 'barracks', 100.0, "兵營"),
                    (production_ai.REFINERY_ID, 'refinery', 50.0, "瓦斯廠"),
                    (production_ai.BARRACKS_TECHLAB_ID, 'techlab', 300.0, "科技室"),
                    (27, 'factory', 300.0, "軍工廠"),
                    (28, 'starport', 400.0, "星際港"),
                    (29, 'armory', 250.0, "兵工廠"),
                    (26, 'ghost_academy', 250.0, "幽靈學院"),
                    (30, 'fusion_core', 500.0, "核融合核心")
                ]
                
                for u_id, key, val, msg in tech_buildings:
                    # ✨ 【核心修正】只有當該建築屬於當前任務的必要科技時，才進行加分判定
                    if key in allowed_tech:
                        if (np.sum((next_s_unit == u_id) & (next_s_relative == 1)) > 0 or \
                            np.sum((next_m_unit == u_id) & (next_m_relative == 1)) > 0) and not milestones[key]:
                            step_reward += val; milestones[key] = True
                            print(f"🏗️ 【任務科技】{msg}完工，獎勵 +{val}")

                # C. 任務目標單位給分
                # --- [C. 任務目標單位：無限加分邏輯] ---
                current_real_count = 0
                if task_cfg:
                    # 1. 偵測目前螢幕上的單位像素
                    u_pixels = np.sum((next_s_unit == task_cfg["id"]) & (next_s_relative == 1))
                    # 2. 換算成數量 (例如戰巡艦 250 像素算 1 隻)
                    current_real_count = int(np.round(float(u_pixels) / task_cfg["pixel"]))
                    
                    if current_real_count > 0:
                        # 🏅 首隻獎勵：這輩子拿一次
                        if not milestones['first_unit']:
                            step_reward += task_cfg["first"]
                            milestones['first_unit'] = True
                            print(f"🥇 首隻 {task_cfg['name']} 誕生！獎勵 +{task_cfg['first']}")
                        
                        # 📈 重複產出獎勵：只要數量增加，就無限給分 (例如每一隻陸戰隊 +30)
                        if current_real_count > last_target_count:
                            gain = (current_real_count - last_target_count) * task_cfg["repeat"]
                            step_reward += gain
                            last_target_count = current_real_count # 更新歷史最高數量
                            print(f"📈 產量增加！目前共 {current_real_count} 隻，追加獎勵 +{gain}")
                
                is_base_on_screen = np.any((next_s_unit == 18) | (next_s_unit == 21))
                

                # --- [D. 狀態與回合結束判定] ---
                # ✨ 確保 done 條件只有「遊戲結束」或「時間到」，不再限制數量
                u_pixels = np.sum((obs_data.feature_screen[features.SCREEN_FEATURES.unit_type.index] == task_info["id"]) & 
                                  (obs_data.feature_screen[features.SCREEN_FEATURES.player_relative.index] == 1))
                current_real_count = int(np.round(float(u_pixels) / task_info["pixel"]))

                # --- 5. 狀態更新與存入記憶 ---
                # --- 5. 狀態更新與存入記憶 ---
                updated_block = getattr(hands, 'active_parameter', 1)
                
                # ✨ 核心修正：補上第 4 個參數 actual_action_id
                # 這樣 AI 才會知道「剛才做了這個動作後，世界變成了什麼樣子」
                actual_action_id = hands.locked_action if hands.locked_action is not None else a_id
                next_state = get_state_vector(next_obs, updated_block, CURRENT_TRAIN_TASK, actual_action_id)
                done = bool(next_obs.last() or obs_data.game_loop[0] >= 13440 or current_real_count >= 5)
                # 如果這一步因為鎖定機制執行了 Action 1，即便隨機抽到 40，也要記為 1
                
                memory.append((state, int(actual_action_id), int(p_id), float(step_reward), next_state, bool(done)))
                obs = next_obs
                # --- 6. 模型訓練 (批次學習) ---
                
                # --- 6. 模型訓練 (批次學習) ---
                train_step_counter += 1
                if len(memory) > 1000 and train_step_counter % 8 == 0:
                    batch = random.sample(memory, 64)
                    
                    # 準備批次數據
                    states, actions_id, params_id, rewards, next_states, dones = zip(*batch)
                    
                    states_t = torch.FloatTensor(np.array(states))
                    next_states_t = torch.FloatTensor(np.array(next_states))
                    actions_t = torch.LongTensor(actions_id)
                    params_t = torch.LongTensor(params_id) - 1 # 轉回 0-15 索引
                    rewards_t = torch.FloatTensor(rewards)
                    dones_t = torch.FloatTensor(dones)

                    # 計算當前 Q 值
                    current_q_actions, current_q_params = brain_model(states_t)
                    q_a = current_q_actions.gather(1, actions_t.unsqueeze(1)).squeeze(1)
                    q_p = current_q_params.gather(1, params_t.unsqueeze(1)).squeeze(1)

                    # 計算目標 Q 值 (Double DQN 簡化版)
                    with torch.no_grad():
                        next_q_actions, next_q_params = brain_model(next_states_t)
                        max_next_q_a = next_q_actions.max(1)[0]
                        max_next_q_p = next_q_params.max(1)[0]
                        target_a = rewards_t + (1 - dones_t) * gamma * max_next_q_a
                        target_p = rewards_t + (1 - dones_t) * gamma * max_next_q_p

                    # 算損失並更新
                    loss = criterion(q_a, target_a) + criterion(q_p, target_p)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                    last_action_id = actual_id  # ✨ 更新記憶：這一步的動作變成下一步的「上一步」
                    current_block = updated_block # 更新目前區塊
                    obs = next_obs
                
                # ✨ 統一合併成一個 done 判定
                # ✨ 統一合併成一個 done 判定 (動態統計版)
                # ✨ 修正後的結算邏輯：使用 milestones 紀錄
                # ✨ 修正後的結算邏輯
                if done:
                    task_name = task_cfg['name'] if task_cfg else "未知任務"
                    final_b_count = 0 
                    final_t_count = 0
                    
                    task_cfg = REWARD_CONFIG.get(CURRENT_TRAIN_TASK)
                    allowed_tech = task_cfg.get("req", []) if task_cfg else []
                    
                    stats_map = [
                        (production_ai.BARRACKS_ID, "兵營", "barracks"),
                        (production_ai.BARRACKS_TECHLAB_ID, "科技室", "techlab"),
                        (production_ai.SUPPLY_DEPOT_ID, "補給站", "depot"),
                        (production_ai.REFINERY_ID, "瓦斯廠", "refinery"), # 👈 檢查這裡
                        (27, "軍工廠", "factory"),
                        (28, "星際港", "starport"),
                        (30, "核融合核心", "fusion_core")
                    ]
                    tech_status_str = ""
                    for u_id, name, key in stats_map:
                        if key in allowed_tech:
                            exists = milestones.get(key, False) 
                            status_icon = "✅" if exists else "❌"
                            tech_status_str += f"{status_icon} {name} | "
                            
                            # ✨ 這裡會正確更新變數
                            if key == "barracks" and exists: final_b_count = 1
                            if key == "techlab" and exists: final_t_count = 1

                    
                    
                    logger.log_episode(
                        ep + 1, 
                        task_info['name'], 
                        epsilon, 
                        total_reward, 
                        final_b_count, 
                        final_t_count, 
                        last_target_count,  # 👈 改成這個，就不怕結算時鏡頭沒對準了
                        obs_data.game_loop[0], 
                        "Done", 
                        False
                    )
                    
                    print(f" 🎯 這一局最高產量: {last_target_count} 隻")
                    print(f" 💰 累計總獎勵: {int(total_reward)}")
                    break
                
            
            # 回合結束後更新 epsilon
            epsilon = max(learn_min, epsilon * epsilon_decay)
            torch.save(brain_model.state_dict(), model_path)

if __name__ == "__main__":
    from absl import app
    app.run(main)