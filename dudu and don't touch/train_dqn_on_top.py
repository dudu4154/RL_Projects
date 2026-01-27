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
# 🐒 猴子補丁：確保路徑正確與解決右下角建築空間 Bug
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(current_dir, "log")

def patched_data_collector_init(self):
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    self.filename = os.path.join(log_dir, f"terran_log_{int(time.time())}.csv")
    with open(self.filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Game_Loop", "Minerals", "Vespene", "Workers", "Ideal", "Action_ID"])

def patched_calc_barracks_pos(self, obs):
    player_relative = obs.observation.feature_minimap[features.MINIMAP_FEATURES.player_relative.index]
    _, x_mini = (player_relative == 1).nonzero()
    is_on_right_side = (x_mini.mean() if x_mini.any() else 0) > 32
    target_x = self.cc_x_screen - 25 if is_on_right_side else self.cc_x_screen + 25
    return (np.clip(target_x, 10, 70), np.clip(self.cc_y_screen - 15, 10, 70))

production_ai.DataCollector.__init__ = patched_data_collector_init
production_ai.ProductionAI._calc_barracks_pos = patched_calc_barracks_pos

# =========================================================
# 📊 訓練紀錄器
# =========================================================
class TrainingLogger:
    def __init__(self):
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        self.filename = os.path.join(log_dir, f"dqn_training_log_{int(time.time())}.csv")
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Total_Reward", "Marauders", "End_Loop", "Reason"])

    def log_episode(self, ep, reward, marauders, loop, reason):
        # ⭐ 修正點：確保 reward 是純數字，防止 numpy 報錯
        if hasattr(reward, "item"): 
            reward = reward.item() # 如果是 numpy 格式，轉換成純數字
        
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([ep, round(float(reward), 2), marauders, loop, reason])

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
    optimizer = optim.Adam(brain_model.parameters(), lr=0.0005) # 穩定學習率
    criterion = nn.MSELoss()
    memory = deque(maxlen=10000) # ⭐ 大容量長期記憶區
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
                curr_loop = obs.observation.game_loop
                
                # 1. 提取狀態 (轉為 list 確保存儲格式統一)
                unit_type = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
                b_cnt = np.sum(unit_type == 21); r_cnt = np.sum(unit_type == 20)
                t_cnt = np.sum(unit_type == 37); m_cnt = int(np.sum(unit_type == 51) / 20)
                
                state = [
                    obs.observation.player.minerals / 1000, obs.observation.player.vespene / 500,
                    obs.observation.player.food_used / 50, b_cnt, r_cnt, t_cnt, m_cnt / 10,
                    0, 0, 1.0
                ]
                state_t = torch.FloatTensor(state)

                # 2. 選擇動作
                if random.random() <= epsilon:
                    a_id = random.randint(0, 9)
                else:
                    with torch.no_grad(): a_id = torch.argmax(brain_model(state_t.unsqueeze(0))).item()

                # 3. 執行動作 (沙包對手模式)
                sc2_action = hands.get_action(obs, a_id)
                obs_list = env.step([sc2_action, actions.FUNCTIONS.no_op()])
                
                # 4. 計算強化版獎勵 (防止洗分 + 全過獎金)
                step_reward = -0.01 
                if r_cnt > last_r and r_cnt <= 2: step_reward += 15.0; last_r = r_cnt
                if b_cnt > last_b and b_cnt <= 2: step_reward += 20.0; last_b = b_cnt
                if t_cnt > last_t: step_reward += 40.0; last_t = t_cnt
                if m_cnt > last_m:
                    step_reward += 150.0; last_m = m_cnt
                    if m_cnt >= 5: # 第 5 隻大紅包
                        step_reward += (500.0 + (13440 - curr_loop) / 100)
                total_reward += step_reward

                # 5. 提取下一個狀態並存入記憶 (統一轉為 list 避免 shape 錯誤)
                next_obs = obs_list[0]
                next_unit = next_obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
                next_state = [
                    next_obs.observation.player.minerals / 1000, next_obs.observation.player.vespene / 500,
                    next_obs.observation.player.food_used / 50, np.sum(next_unit==21), 
                    np.sum(next_unit==20), np.sum(next_unit==37), int(np.sum(next_unit==51)/20) / 10,
                    0, 0, 1.0
                ]
                
                done = next_obs.last() or m_cnt >= 5 or curr_loop >= 13440
                # ⭐ 關鍵修正：存入 list 而非 tensor，解決 inhomogeneous shape 問題
                memory.append((state, a_id, step_reward, next_state, done))

                # 🧠 高效批量學習
                if len(memory) > 128:
                    batch = random.sample(memory, 64)
                    b_states, b_actions, b_rewards, b_next_states, b_dones = zip(*batch)
                    
                    # 使用 np.array() 中轉，這是最穩定的格式轉換方法
                    b_states_t = torch.tensor(np.array(b_states), dtype=torch.float)
                    b_next_states_t = torch.tensor(np.array(b_next_states), dtype=torch.float)
                    b_actions_t = torch.tensor(np.array(b_actions), dtype=torch.long)
                    b_rewards_t = torch.tensor(np.array(b_rewards), dtype=torch.float)
                    b_dones_t = torch.tensor(np.array(b_dones), dtype=torch.bool)

                    with torch.no_grad():
                        next_q = brain_model(b_next_states_t).max(1)[0]
                        targets = b_rewards_t + (~b_dones_t).float() * gamma * next_q
                    
                    current_q = brain_model(b_states_t).gather(1, b_actions_t.unsqueeze(1)).squeeze(1)
                    loss = criterion(current_q, targets)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()

                if done:
                    reason = "Target_Reached" if m_cnt >= 5 else "Timeout"
                    logger.log_episode(ep+1, total_reward, m_cnt, curr_loop, reason)
                    # 修正後的 print 語句
                    print(f"回合結束 | 產量: {int(m_cnt)} | 總分: {float(total_reward):.2f}")
                    break
            
            # 回合間更新
            epsilon = max(0.15, epsilon * epsilon_decay)
            torch.save(brain_model.state_dict(), model_path)

if __name__ == "__main__":
    from absl import app
    app.run(main)