import sys
import random
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app, flags
import os

# 嘗試告訴 PySC2 去 D 槽找遊戲
os.environ["SC2PATH"] = "D:/StarCraft II"
from chx_dqn_agent import ACTION_MAP, N_ACTIONS, get_state, check_valid_actions

# =================================================================
# 設定參數
# =================================================================
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 8000 # 🚀 延長探索時間，讓 AI 有更多機會嘗試不同動作
TARGET_UPDATE = 10
LR = 1e-4
MEMORY_SIZE = 10000
MAP_NAME = "Simple96"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.head = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.head(x)

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, next_state, reward, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, next_state, reward, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# =================================================================
# 主訓練迴圈
# =================================================================
def main(unused_argv):
    writer = SummaryWriter('runs/SC2_DQN_Optimized')
    
    try:
        env = sc2_env.SC2Env(
            map_name=MAP_NAME,
            players=[
                sc2_env.Agent(sc2_env.Race.terran),
                sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)
            ],
            step_mul=24,             # 🚀 提升遊戲運行速度
            visualize=False,
            game_steps_per_episode=4000, # 🚀 限制每場時長，專注練習開局
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),
                use_feature_units=True
            )
        )
    except Exception as e:
        print(f"❌ 啟動環境失敗: {e}")
        return

    n_observations = 30
    policy_net = DQN(n_observations, N_ACTIONS).to(DEVICE)
    target_net = DQN(n_observations, N_ACTIONS).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)
    steps_done = 0

    try:
        for i_episode in range(1000):
            print(f"🎮 Episode {i_episode + 1} 開始...")
            obs = env.reset()[0]
            state = torch.tensor(get_state(obs), dtype=torch.float32).to(DEVICE).unsqueeze(0)
            
            milestones = {'barracks': False, 'factory': False, 'starport': False}
            total_custom_reward = 0
            
            while True:
                valid_mask = check_valid_actions(obs, ACTION_MAP)
                sample = random.random()
                eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
                steps_done += 1

                if sample > eps_threshold:
                    with torch.no_grad():
                        q_values = policy_net(state)
                        mask_tensor = torch.tensor(valid_mask, device=DEVICE).unsqueeze(0)
                        q_values[mask_tensor == 0] = -1e9
                        action_index = q_values.max(1)[1].item()
                else:
                    possible_actions = np.where(valid_mask == 1)[0]
                    action_index = np.random.choice(possible_actions) if len(possible_actions) > 0 else 13

                func = ACTION_MAP[action_index]
                if func.id in obs.observation.available_actions:
                    args = [[np.random.randint(0, 84), np.random.randint(0, 84)]] if len(func.args) > 0 else []
                    try:
                        next_obs = env.step(actions=[func(*args)])[0]
                    except:
                        next_obs = env.step(actions=[actions.FUNCTIONS.no_op()])[0]
                else:
                    next_obs = env.step(actions=[actions.FUNCTIONS.no_op()])[0]

                # =========================================================
                # ✨ 獎勵與邏輯計算 (Reward Shaping)
                # =========================================================
                reward = next_obs.reward 
                custom_reward = -0.001   # 🚀 時間懲罰：逼 AI 別原地發呆
                
                player = next_obs.observation.player
                unit_types = next_obs.observation.feature_screen.unit_type

                if not milestones['barracks'] and (unit_types == units.Terran.Barracks).any():
                    custom_reward += 0.2
                    milestones['barracks'] = True
                if not milestones['factory'] and (unit_types == units.Terran.Factory).any():
                    custom_reward += 0.3
                    milestones['factory'] = True
                
                if player.food_used >= player.food_cap and player.food_cap < 200:
                    custom_reward -= 0.01 
                
                if player.minerals > 1500:
                    custom_reward -= 0.005 
                
                kill_value = next_obs.observation.score_cumulative.killed_value_units / 5000.0
                
                total_step_reward = reward + custom_reward + kill_value
                total_custom_reward += total_step_reward

                next_state = torch.tensor(get_state(next_obs), dtype=torch.float32).to(DEVICE).unsqueeze(0)
                done = next_obs.last()

                # =========================================================
                # 🚩 Episode 結束時的懲罰與監控 (if done 區塊)
                # =========================================================
                if done:
                    # 嚴重消極懲罰：如果整場都沒產兵，重扣 1 分
                    if player.food_army == 0:
                        total_step_reward -= 1.0
                        total_custom_reward -= 1.0 
                    
                    # 統計並紀錄數據至 TensorBoard
                    barracks_count = (unit_types == units.Terran.Barracks).sum().item()
                    writer.add_scalar('Metrics/Barracks_Count', barracks_count, i_episode)
                    writer.add_scalar('Metrics/Army_Population', player.food_army, i_episode)
                    writer.add_scalar('Training/Total_Reward', total_custom_reward, i_episode)

                    print(f"🏁 Episode {i_episode + 1} 結束, 總得分: {total_custom_reward:.2f}, 兵營數: {barracks_count}, 兵力: {player.food_army}")
                    
                    memory.push(state, action_index, next_state, total_step_reward, done)
                    break

                memory.push(state, action_index, next_state, total_step_reward, done)
                state = next_state

                # 訓練網路
                if len(memory) > BATCH_SIZE:
                    transitions = memory.sample(BATCH_SIZE)
                    batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*transitions)
                    
                    b_state = torch.cat(batch_state)
                    b_action = torch.tensor(batch_action).unsqueeze(1).to(DEVICE)
                    b_reward = torch.tensor(batch_reward).unsqueeze(1).to(DEVICE).float()
                    b_next_state = torch.cat(batch_next_state)
                    b_done = torch.tensor(batch_done, dtype=torch.float32).unsqueeze(1).to(DEVICE)

                    q_eval = policy_net(b_state).gather(1, b_action)
                    q_next = target_net(b_next_state).detach().max(1)[0].unsqueeze(1)
                    q_target = b_reward + GAMMA * q_next * (1 - b_done).float()

                    loss = F.mse_loss(q_eval, q_target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if steps_done % 100 == 0:
                        writer.add_scalar('Training/Loss', loss.item(), steps_done)

            if i_episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

    except KeyboardInterrupt:
        print("停止訓練。")
    finally:
        env.close()
        writer.close()

if __name__ == "__main__":
    app.run(main)