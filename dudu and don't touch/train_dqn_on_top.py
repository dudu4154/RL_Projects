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
            writer.writerow(["Episode", "Epsilon", "Total_Reward", "Marauders", "End_Loop", "Reason", "Is_Bottom_Right"])

    def log_episode(self, ep, eps, reward, m_cnt, loop, reason, location):
        """ 紀錄每回合摘要，加入 eps 參數 """
        if hasattr(reward, "item"): 
            reward = reward.item()
        int_reward = int(reward) 
        
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            # 寫入數據時對應標題順序
            writer.writerow([ep, f"{eps:.3f}", int_reward, m_cnt, loop, reason, location])

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
    
    def get_state_vector(obs, current_block):
        player = obs.observation.player
        m_unit = obs.observation.feature_minimap[features.MINIMAP_FEATURES.unit_type.index]
        s_unit = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
        
        # 統一計算邏輯
        return [
            player.food_workers / 16,                               # 1. 工兵
            player.minerals / 1000,                                 # 2. 礦石
            player.vespene / 500,                                  # 3. 瓦斯
            player.food_used / 50,                                 # 4. 人口
            np.sum(m_unit == production_ai.BARRACKS_ID),            # 5. 全地圖兵營 (小地圖)
            np.sum(m_unit == production_ai.REFINERY_ID),            # 6. 全地圖瓦斯廠 (小地圖)
            np.sum(m_unit == production_ai.BARRACKS_TECHLAB_ID),    # 7. 全地圖實驗室 (小地圖)
            int(np.sum(s_unit == 51) / 20) / 10,                   # 8. 掠奪者 (畫面)
            current_block / 16.0,                                   # 9. 視角位置
            float(np.sum(s_unit == 21) > 0),                        # 10. 畫面是否有兵營 (轉為 0.0/1.0)
            1.0                                                     # 11. 常數
        ]

# =========================================================
# 🎮 訓練主程式
# =========================================================
def main(argv):
    del argv
    state_size = 11 # 增加一格狀態紀錄「目前看哪」
    action_size = 41
    
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
        for ep in range(100):
            hands = ProductionAI() 
            print(f"\n🚀 === 啟動第 {ep+1} 回合 (Epsilon: {epsilon:.3f}) ===")
            obs_list = env.reset()
            last_m=0; last_b=0; last_r=0; last_t=0
            total_reward = 0
            
            while True:
                obs = obs_list[0]
                player = obs.observation.player # 新增：提取 player 資訊
                current_workers = player.food_workers # 
                minimap_unit_type = obs.observation.feature_minimap[features.MINIMAP_FEATURES.unit_type.index]
                
                # --- 【修正 1】 提取必要變數 ---
                unit_type = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
                r_cnt = np.sum(unit_type == 20)  
                b_cnt = np.sum(unit_type == 21)  
                t_cnt = np.sum(unit_type == 37)  
                m_cnt = int(np.sum(unit_type == 51) / 20) 
                curr_loop = obs.observation.game_loop[0]
                # 在小地圖中，建築物也會以對應的 ID 顯示
                global_b_cnt = np.sum(minimap_unit_type == production_ai.BARRACKS_ID)
                global_r_cnt = np.sum(minimap_unit_type == production_ai.REFINERY_ID)
                global_t_cnt = np.sum(minimap_unit_type == production_ai.BARRACKS_TECHLAB_ID)
                
                # 偵測當前畫面 (Screen) 是否有建築，這能幫助 AI 學習「移動視角」的必要性
                screen_unit = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
                screen_b_cnt = np.sum(screen_unit == production_ai.BARRACKS_ID)

                # 提取狀態 (確保 current_workers 與 player 已定義)
                current_block = getattr(hands, 'active_parameter', 1)
                state = [
                    current_workers / 16,
                    player.minerals / 1000,
                    player.vespene / 500,
                    player.food_used / 50,
                    global_b_cnt,             # 全地圖兵營總數
                    global_r_cnt,             # 全地圖瓦斯廠總數
                    global_t_cnt,             # 全地圖實驗室總數
                    m_cnt / 10.0,             
                    current_block / 16.0,     # 目前看哪裡
                    screen_b_cnt > 0,         # 目前畫面看得到兵營嗎？ (布林值轉 0/1)
                    1.0
                ]
                state_t = torch.FloatTensor(np.array(state))

                # 2. 同時選擇動作與參數 (Epsilon-Greedy)
                if random.random() <= epsilon:
                    a_id = random.randint(1, 40)# 從可用動作中選
                    p_id = random.randint(1, 16) # 隨機選一個網格
                else:
                    with torch.no_grad():
                        q_actions, q_params = brain_model(state_t.unsqueeze(0))
                        a_id = torch.argmax(q_actions).item()
                        p_id = torch.argmax(q_params).item() + 1

                # 3. 執行動作：傳入參數
                sc2_action = hands.get_action(obs, a_id, parameter=p_id)
                obs_list = env.step([sc2_action, actions.FUNCTIONS.no_op()])
                
                # 4. 獎勵邏輯 (維持原有架構)
                # --- 獎勵邏輯 (強化掠奪者權重) ---
                step_reward = -0.01 
                if m_cnt > last_m:
                    step_reward += 200.0  
                    last_m = m_cnt
                    if m_cnt >= 5: 
                        step_reward += 1000.0

                total_reward += step_reward


                # 5. 提取下一個狀態並存入記憶
                next_obs = obs_list[0]
                next_unit = next_obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
                # --- 修正後的 next_state (確保與 state 的 11 維度完全對齊) ---
                next_player = next_obs.observation.player
                next_unit = next_obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]

                next_state = [
                    next_player.food_workers / 16,               # 1. 工兵 (原代碼漏掉這項導致維度變 10)
                    next_player.minerals / 1000,                # 2. 礦石
                    next_player.vespene / 500,                 # 3. 瓦斯
                    next_player.food_used / 50,                 # 4. 人口
                    np.sum(next_unit == 21),                    # 5. 兵營
                    np.sum(next_unit == 20),                    # 6. 瓦斯廠
                    np.sum(next_unit == 37),                    # 7. 科技實驗室
                    int(np.sum(next_unit == 51) / 20) / 10,     # 8. 掠奪者
                    current_block / 16.0,                       # 9. 目前視角
                    0,                                          # 10. 預留
                    1.0                                         # 11. 常數
                ]
                
                done = bool(next_obs.last() or m_cnt >= 5 or curr_loop >= 13440)
                memory.append((state, int(a_id), int(p_id), float(step_reward), next_state, bool(done)))

                # --- 🧠 模型學習部分的修正 ---
                # --- 學習部分的雙頭 Loss ---
                if len(memory) > 256:
                    batch = random.sample(memory, 64)
                    # 加入 b_params
                    b_states, b_actions, b_params, b_rewards, b_next_states, b_dones = zip(*batch)
                    
                    b_states_t = torch.as_tensor(np.array(b_states), dtype=torch.float32)
                    b_next_states_t = torch.as_tensor(np.array(b_next_states), dtype=torch.float32)
                    b_actions_t = torch.as_tensor(b_actions, dtype=torch.long)
                    b_params_t = torch.as_tensor(b_params, dtype=torch.long) - 1 # 轉回 0-15 索引
                    b_rewards_t = torch.as_tensor(b_rewards, dtype=torch.float32)
                    b_dones_t = torch.as_tensor(np.array(b_dones, dtype=np.float32))

                    # 同時計算當前動作與參數的 Q 值
                    curr_q_a, curr_q_p = brain_model(b_states_t)
                    # 同時計算下一個狀態的動作與參數 Q 值
                    next_q_a, next_q_p = brain_model(b_next_states_t)
                    
                    # 動作 Loss 計算
                    targets_a = b_rewards_t + (1 - b_dones_t) * gamma * next_q_a.max(1)[0].detach()
                    loss_a = criterion(curr_q_a.gather(1, b_actions_t.unsqueeze(1)).squeeze(1), targets_a)
                    
                    # 參數 Loss 計算 (讓網格選擇也跟著獎勵學習)
                    targets_p = b_rewards_t + (1 - b_dones_t) * gamma * next_q_p.max(1)[0].detach()
                    loss_p = criterion(curr_q_p.gather(1, b_params_t.unsqueeze(1)).squeeze(1), targets_p)
                    
                    # 合併 Loss 並更新模型
                    total_loss = loss_a + loss_p
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                if done:
                    loc_text = (production_ai.BASE_LOCATION_CODE == 1)
                    reason = "Target_Reached" if m_cnt >= 5 else "Timeout"
                    
                    logger.log_episode(ep+1, epsilon, total_reward, m_cnt, curr_loop, reason, loc_text)
                    
                    # 【修正】將 worker_cnt 改為 current_workers
                    print(f"回合結束 | 掠奪者數量: {m_cnt} | 工兵數量: {current_workers} | 總分: {int(total_reward)}")
                    break
            
            # 回合結束後更新 epsilon
            epsilon = max(learn_min, epsilon * epsilon_decay)
            torch.save(brain_model.state_dict(), model_path)

if __name__ == "__main__":
    from absl import app
    app.run(main)