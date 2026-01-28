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

# =========================================================
# 🐒 猴子補丁與路徑設定
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(current_dir, "log")

def patched_data_collector_init(self):
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    self.filename = os.path.join(log_dir, f"terran_log_{int(time.time())}.csv")
    with open(self.filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Game_Loop", "Minerals", "Vespene", "Workers", "Ideal", "Action_ID"])

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
            writer.writerow(["Episode", "Total_Reward", "Marauders", "End_Loop", "Reason", "Is_Bottom_Right"])

    def log_episode(self, ep, reward, m_cnt, loop, reason, location):
        """ 紀錄每回合摘要，將獎勵轉為整數 """
        # 確保獎勵是純數字並轉換為整數
        if hasattr(reward, "item"): 
            reward = reward.item()
        int_reward = int(reward) 
        
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            # 使用與傳入參數一致的變數名稱
            writer.writerow([ep, int_reward, m_cnt, loop, reason, location])
            
# =========================================================
# 🧠 深度學習模型 (DQN)
# =========================================================
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_size)
        )
    def forward(self, x): return self.fc(x)

# =========================================================
# 🎮 訓練主程式
# =========================================================
def main(argv):
    del argv
    state_size = 10; action_size = 10
    
    brain_model = QNetwork(state_size, action_size)
    optimizer = optim.Adam(brain_model.parameters(), lr=0.0005) 
    criterion = nn.MSELoss()
    memory = deque(maxlen=10000) 
    logger = TrainingLogger()
    
    model_path = os.path.join(log_dir, "dqn_model.pth")
    if os.path.exists(model_path):
        brain_model.load_state_dict(torch.load(model_path))
        print("✅ 載入成功！接續之前的記憶繼續訓練...")

    epsilon = 1.0; epsilon_decay = 0.995; gamma = 0.99 

    with sc2_env.SC2Env(
        map_name="Simple64",
        players=[sc2_env.Agent(sc2_env.Race.terran), sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=sc2_env.AgentInterfaceFormat(
            feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64), use_raw_units=False),
        step_mul=32, realtime=False
    ) as env:
        for ep in range(200):
            hands = ProductionAI() 
            print(f"\n🚀 === 啟動第 {ep+1} 回合 (Epsilon: {epsilon:.3f}) ===")
            obs_list = env.reset()
            last_m=0; last_b=0; last_r=0; last_t=0
            total_reward = 0
            
            while True:
                obs = obs_list[0]
                curr_loop = int(obs.observation.game_loop)
                
                # 1. 提取狀態
                unit_type = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
                b_cnt = np.sum(unit_type == 21); r_cnt = np.sum(unit_type == 20)
                t_cnt = np.sum(unit_type == 37); m_cnt = int(np.sum(unit_type == 51) / 20)
                
                state = [
                    obs.observation.player.minerals / 1000, obs.observation.player.vespene / 500,
                    obs.observation.player.food_used / 50, b_cnt, r_cnt, t_cnt, m_cnt / 10,
                    0, 0, 1.0
                ]
                # 優化：先轉 numpy 陣列再轉 Tensor
                state_t = torch.FloatTensor(np.array(state))

                # 2. 選擇動作
                if random.random() <= epsilon:
                    a_id = random.randint(0, 9)
                else:
                    with torch.no_grad(): a_id = torch.argmax(brain_model(state_t.unsqueeze(0))).item()

                # 3. 執行動作
                sc2_action = hands.get_action(obs, a_id)
                obs_list = env.step([sc2_action, actions.FUNCTIONS.no_op()])
                
                # 4. 獎勵邏輯 (維持原有架構)
                step_reward = -0.01 
                if r_cnt > last_r and r_cnt <= 2: step_reward += 15.0; last_r = r_cnt
                if b_cnt > last_b and b_cnt <= 2: step_reward += 20.0; last_b = b_cnt
                if t_cnt > last_t: step_reward += 40.0; last_t = t_cnt
                if m_cnt > last_m:
                    step_reward += 150.0; last_m = m_cnt
                    if m_cnt >= 5: step_reward += 500.0
                total_reward += step_reward

                # 5. 提取下一個狀態並存入記憶
                next_obs = obs_list[0]
                next_unit = next_obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
                next_state = [
                    next_obs.observation.player.minerals / 1000, next_obs.observation.player.vespene / 500,
                    next_obs.observation.player.food_used / 50, np.sum(next_unit==21), 
                    np.sum(next_unit==20), np.sum(next_unit==37), int(np.sum(next_unit==51)/20) / 10,
                    0, 0, 1.0
                ]
                
                done = bool(next_obs.last() or m_cnt >= 5 or curr_loop >= 13440)
                # 存入記憶時強制轉換類型
                memory.append((state, int(a_id), float(step_reward), next_state, bool(done)))

                # --- 🧠 模型學習部分的修正 ---
                if len(memory) > 128:
                    batch = random.sample(memory, 64)
                    b_states, b_actions, b_rewards, b_next_states, b_dones = zip(*batch)
                    
                    # 使用 torch.as_tensor 或先轉為 float 類型的 numpy 陣列
                    b_states_t = torch.as_tensor(np.array(b_states), dtype=torch.float32)
                    b_next_states_t = torch.as_tensor(np.array(b_next_states), dtype=torch.float32)
                    b_actions_t = torch.as_tensor(b_actions, dtype=torch.long)
                    b_rewards_t = torch.as_tensor(b_rewards, dtype=torch.float32)
                    
                    # 這裡最關鍵：先轉成 float 的 numpy 陣列，再轉 Tensor
                    b_dones_t = torch.as_tensor(np.array(b_dones, dtype=np.float32))

                    with torch.no_grad():
                        # 使用 .max(1)[0] 確保 Q 值維度正確
                        next_q = brain_model(b_next_states_t).max(1)[0]
                        targets = b_rewards_t + (1 - b_dones_t) * gamma * next_q
                    
                    current_q = brain_model(b_states_t).gather(1, b_actions_t.unsqueeze(1)).squeeze(1)
                    loss = criterion(current_q, targets)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()

                if done:
                    # 讀取出生點狀態
                    loc_text = (production_ai.BASE_LOCATION_CODE == 1)
                    reason = "Target_Reached" if m_cnt >= 5 else "Timeout"
                    
                    # 傳入 6 個參數給紀錄器
                    logger.log_episode(ep+1, total_reward, m_cnt, curr_loop, reason, loc_text)
                    
                    # 終端機顯示同樣轉為整數
                    print(f"回合結束 | 出生點右下: {loc_text} ({production_ai.BASE_LOCATION_CODE}) | "
                        f"產量: {int(m_cnt)} | 總分: {int(total_reward)}")
                    break
            
            epsilon = max(0.99, epsilon * epsilon_decay)
            torch.save(brain_model.state_dict(), model_path)

if __name__ == "__main__":
    from absl import app
    app.run(main)