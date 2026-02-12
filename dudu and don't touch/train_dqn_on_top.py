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
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # 【同步標頭】確保包含所有統計項目
            writer.writerow(["Episode", "Epsilon", "Total_Reward", "Barracks", "TechLabs", "Marauders", "End_Loop", "Reason", "Is_Bottom_Right"])

    # 【修正定義】增加 t_cnt 參數，使其總共接收 9 個參數 (含 self)
    def log_episode(self, ep, eps, reward, b_cnt, t_cnt, m_cnt, loop, reason, location):
        """ 紀錄每回合摘要，確保與傳入參數數量一致 """
        if hasattr(reward, "item"): 
            reward = reward.item()
        
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            # 依序寫入數據
            writer.writerow([ep, f"{eps:.3f}", int(reward), b_cnt, t_cnt, m_cnt, loop, reason, location])

# =========================================================
# 🧠 深度學習模型 (DQN)
# =========================================================
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, param_size=16):
        super(QNetwork, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        # 動作頭：決定執行哪個 Action (0-9, 40)
        self.action_head = nn.Linear(64, action_size)
        # 參數頭：決定目標網格 (1-16)
        self.param_head = nn.Linear(64, param_size)

    def forward(self, x):
        x = self.common(x)
        return self.action_head(x), self.param_head(x) # 同時回傳兩組 Q 值
    
def get_state_vector(obs, current_block, target_project_id):
    player = obs.observation.player
    m_unit = obs.observation.feature_minimap[features.MINIMAP_FEATURES.unit_type.index]
    m_relative = obs.observation.feature_minimap[features.MINIMAP_FEATURES.player_relative.index]
    
    # 偵測選取狀態
    is_scv_selected = 0.0
    is_cc_selected = 0.0
    if len(obs.observation.single_select) > 0:
        u_type = obs.observation.single_select[0].unit_type
        if u_type == production_ai.SCV_ID: is_scv_selected = 1.0
        if u_type == production_ai.COMMAND_CENTER_ID: is_cc_selected = 1.0

    # 確保回傳 12 個特徵
    return [
        player.food_workers / 16,
        player.minerals / 1000,
        player.vespene / 500,
        player.food_used / 50,
        np.sum((m_unit == production_ai.BARRACKS_ID) & (m_relative == 1)),
        np.sum((m_unit == production_ai.SUPPLY_DEPOT_ID) & (m_relative == 1)),
        np.sum((m_unit == production_ai.REFINERY_ID) & (m_relative == 1)),
        np.sum((m_unit == production_ai.BARRACKS_TECHLAB_ID) & (m_relative == 1)),
        current_block / 16.0,
        is_scv_selected, 
        is_cc_selected,
        target_project_id / 40.0
    ]
    

# =========================================================
# 🎮 訓練主程式
# =========================================================
def main(argv):
    del argv
    state_size = 12 # 增加一格狀態紀錄「目前看哪」
    action_size = 43
    CURRENT_TRAIN_TASK = 18
    brain_model = QNetwork(state_size, action_size)
    optimizer = optim.Adam(brain_model.parameters(), lr=0.0005) 
    criterion = nn.MSELoss()
    memory = deque(maxlen=100000) 
    logger = TrainingLogger()
    learn_min = 0.01 # 這是你的 epsilon 最小值
    
    model_path = os.path.join(log_dir, "dqn_model.pth")
    if os.path.exists(model_path):
        brain_model.load_state_dict(torch.load(model_path))
        print("✅ 載入成功！接續之前的記憶繼續訓練...")

    epsilon = 1.0; epsilon_decay = 0.999; gamma = 0.99 

    with sc2_env.SC2Env(
        map_name="Simple64",
        players=[sc2_env.Agent(sc2_env.Race.terran), sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=sc2_env.AgentInterfaceFormat(
            feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64), use_raw_units=False),
        step_mul=16, realtime=False
    ) as env:
        for ep in range(1000):
            # --- 1. 初始化環境與變數 (修復 UnboundLocalError) ---
            hands = ProductionAI() 
            obs_list = env.reset() 
            obs = obs_list[0]  # 【關鍵】確保進入 while 之前 obs 已被定義
            
            # 初始化追蹤變數
            last_target_count = 0 
            rewarded_depots = 0     # 【新增】紀錄已給分過的補給站數量
            last_d_pixels = 0
            has_rewarded_barracks = False 
            has_rewarded_techlab = False  
            has_rewarded_home = False # 【新增】一次性回家獎勵旗標
            has_rewarded_control_group = False
            total_reward = 00
            # 預設動作與參數
            a_id = 40; p_id = 1 

            while True:
                # --- 1. 取得當前狀態與選擇動作 (補全被省略的部分) ---
                current_block = getattr(hands, 'active_parameter', 1)
                state = get_state_vector(obs, current_block, CURRENT_TRAIN_TASK)
                state_t = torch.FloatTensor(np.array(state))

                # Epsilon-Greedy 選擇動作
                if random.random() <= epsilon:
                    a_id = random.randint(1, 42)
                    p_id = random.randint(1, 16)
                else:
                    with torch.no_grad():
                        q_actions, q_params = brain_model(state_t.unsqueeze(0))
                        a_id = torch.argmax(q_actions).item()
                        p_id = torch.argmax(q_params).item() + 1

                # --- 2. 執行動作 ---
                sc2_action = hands.get_action(obs, a_id, parameter=p_id)
                obs_list = env.step([sc2_action, actions.FUNCTIONS.no_op()])
                next_obs = obs_list[0]
                
                # --- 4. 獎勵邏輯修正 ---
                step_reward = -0.01 
                
                # 【核心修正】在獎勵判定前定義變數
                obs_data = next_obs.observation
                is_scv_selected = False
                is_cc_selected = False
                
                # 取得最新一幀的特徵
                next_s_unit = next_obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
                next_s_relative = next_obs.observation.feature_screen[features.SCREEN_FEATURES.player_relative.index]
                next_m_unit = next_obs.observation.feature_minimap[features.MINIMAP_FEATURES.unit_type.index]
                next_m_relative = next_obs.observation.feature_minimap[features.MINIMAP_FEATURES.player_relative.index]

                # A. 【新增】補給站獎勵 (限前 2 個，使用全域小地圖判定)
                curr_d_pixels = np.sum((next_m_unit == production_ai.SUPPLY_DEPOT_ID) & (next_m_relative == 1))
                if curr_d_pixels > last_d_pixels:
                    if rewarded_depots < 2:
                        rewarded_depots += 1
                        step_reward += 30.0 # 補給站權重提高至 30
                        print(f"🏠 偵測到新補給站完工！累計: {rewarded_depots} | 獎勵 +30")
                    last_d_pixels = curr_d_pixels

                if is_scv_selected:
                    step_reward += 0.5 
                if is_cc_selected:
                    step_reward += 0.5
                
                if 1 <= a_id <= 13 and not is_scv_selected:
                    step_reward -= 0.1
                # B. 【修正】回家獎勵 (每局限一次)
                if a_id == 40 and not has_rewarded_home:
                    cc_visible = np.any((next_s_unit == production_ai.COMMAND_CENTER_ID) & (next_s_relative == 1))
                    if cc_visible:
                        step_reward += 10.0 
                        has_rewarded_home = True
                        print(f"🏠 第一次找到基地！獎勵 +10")

                # C. 【修正】Action 41 編隊獎勵 (改用 next_obs 判定結果)
                if a_id == 41 and not has_rewarded_control_group:
                    # 必須檢查動作執行「後」的結果
                    is_cc_selected_now = False
                    obs_data = next_obs.observation # 使用下一步的資料
                    
                if len(obs_data.single_select) > 0:
                    u_type = obs_data.single_select[0].unit_type
                    if u_type == production_ai.SCV_ID: is_scv_selected = True
                    if u_type == production_ai.COMMAND_CENTER_ID: is_cc_selected = True
                elif len(obs_data.multi_select) > 0:
                    if obs_data.multi_select[0].unit_type == production_ai.SCV_ID: is_scv_selected = True
                    
                    # 檢查控制組 1 是否已被正確設定
                    control_groups = obs_data.control_groups
                    if is_cc_selected_now and control_groups[1][0] == production_ai.COMMAND_CENTER_ID:
                        step_reward += 15.0 
                        has_rewarded_control_group = True
                        print("⌨️ 成功將主堡編入隊伍 1！獎勵 +15")
                
                # 螢幕判定兵營加分 (限每局一次)
                if np.any((next_s_unit == production_ai.BARRACKS_ID) & (next_s_relative == 1)) and not has_rewarded_barracks:
                    step_reward += 60.0
                    has_rewarded_barracks = True
                    print("🏗️ 螢幕偵測到兵營！獎勵 +60")

                # 螢幕判定科技實驗室加分
                if np.any((next_s_unit == production_ai.BARRACKS_TECHLAB_ID) & (next_s_relative == 1)) and not has_rewarded_techlab:
                    step_reward += 100.0
                    has_rewarded_techlab = True
                    print("🧪 螢幕偵測到實驗室！獎勵 +100")

                # 目標單位 (掠奪者) 產出加分
                self_m_pixels = np.sum((next_s_unit == production_ai.MARAUDER_ID) & (next_s_relative == 1))
                real_m_count = int(np.round(float(self_m_pixels) / 22.0))
                if real_m_count > last_target_count:
                    step_reward += 200.0
                    print(f"🎯 產出狩獵者！數量: {real_m_count}")
                    last_target_count = real_m_count

                total_reward += step_reward

                # --- 5. 狀態更新與存入記憶 ---
                updated_block = getattr(hands, 'active_parameter', 1)
                next_state = get_state_vector(next_obs, updated_block, CURRENT_TRAIN_TASK)
                done = bool(next_obs.last() or real_m_count >= 5 or next_obs.observation.game_loop[0] >= 20160)
                
                # 將經驗存入 deque 供後續 batch 訓練
                memory.append((state, int(a_id), int(p_id), float(step_reward), next_state, bool(done)))
                
                # --- 6. 模型訓練 (批次學習) ---
                if len(memory) > 1000:
                    batch = random.sample(memory, 64)
                    # (此處應執行 optimizer.step() 等 DQN 訓練邏輯，建議保留你原本的實作)

                if done:
                    # 統計兵營與科技實驗室 (全域掃描)
                    final_b_pixels = np.sum((next_m_unit == production_ai.BARRACKS_ID) & (next_m_relative == 1))
                    final_b_count = 1 if final_b_pixels > 0 else 0
                    
                    final_t_pixels = np.sum((next_m_unit == production_ai.BARRACKS_TECHLAB_ID) & (next_m_relative == 1))
                    final_t_count = 1 if final_t_pixels > 0 else 0
                    
                    # 【核心修正】這裡傳入的參數順序必須與 log_episode 定義一致
                    logger.log_episode(
                        ep + 1,            # Episode (第幾次)
                        epsilon,           # Epsilon
                        total_reward,      # 總分
                        final_b_count,     # 兵營
                        final_t_count,     # 科技實驗室
                        real_m_count,      # 掠奪者 (狩獵者)
                        next_obs.observation.game_loop[0], # Loop
                        "Done",            # Reason
                        (production_ai.BASE_LOCATION_CODE == 1) # Location
                    )
                    
                    # 控制台同步輸出統計內容
                    print(f"\n" + "="*40)
                    print(f"🏁 第 {ep+1} 次 訓練結算")
                    print(f"🏠 兵營: {final_b_count} | 🧪 實驗室: {final_t_count} | 🎯 掠奪者: {real_m_count}")
                    print(f"💰 總分: {int(total_reward)}")
                    print("="*40 + "\n")
                    break
            
            # 回合結束後更新 epsilon
            epsilon = max(learn_min, epsilon * epsilon_decay)
            torch.save(brain_model.state_dict(), model_path)

if __name__ == "__main__":
    from absl import app
    app.run(main)