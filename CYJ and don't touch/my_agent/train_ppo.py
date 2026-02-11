import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import datetime
import torch.nn.functional as F

from absl import app
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from torch.utils.tensorboard import SummaryWriter

# 匯入我們剛剛寫好的大腦
from ppo_model import ActorCritic 

# === 參數設定 ===
LR = 0.0003              # 學習率
GAMMA = 0.99             # 折扣因子 (看重未來的程度)
UPDATE_STEPS = 10        # 每幾局更新一次網路 (或是累積多少步)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定義 6 個簡單動作
ACTION_DO_NOTHING = 0
ACTION_BUILD_PROBE = 1
ACTION_BUILD_PYLON = 2
ACTION_BUILD_GATEWAY = 3
ACTION_BUILD_ASSIMILATOR = 4
ACTION_TRAIN_ZEALOT = 5

class ProtossRLAgent:
    def __init__(self):
        self.model = ActorCritic(num_inputs=27, num_actions=6).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        
        # 記憶體：用來存一整局的資料 (State, Action, LogProb, Reward, Value)
        self.memory_states = []
        self.memory_actions = []
        self.memory_logprobs = []
        self.memory_rewards = []
        self.memory_values = []
        
        # 輔助變數
        self.nexus_id = 59
        self.pylon_id = 60
        self.probe_id = 84
        self.gateway_id = 62

    def preprocess(self, obs):
        screen = obs.observation.feature_screen
        # 增加 batch 維度並轉送到 GPU
        return torch.from_numpy(screen).float().unsqueeze(0).to(DEVICE)

    def step(self, obs):
        state_tensor = self.preprocess(obs)
        
        # 1. 讓大腦決定動作
        action_probs, state_value = self.model(state_tensor)
        
        # 2. 根據機率隨機採樣 (Exploration)
        dist = torch.distributions.Categorical(action_probs)
        action_index = dist.sample()
        log_prob = dist.log_prob(action_index)
        
        # 3. 暫存資料 (等待這一局結束後計算 Reward)
        self.memory_states.append(state_tensor)
        self.memory_actions.append(action_index)
        self.memory_logprobs.append(log_prob)
        self.memory_values.append(state_value)
        
        # 4. 執行動作
        return self.execute_action(action_index.item(), obs)

    def execute_action(self, action_cmd, obs):
        """將 AI 指令翻譯成 PySC2 動作"""
        minerals = obs.observation.player.minerals
        
        # 簡單的規則：如果有錢且動作可用，就做；否則發呆或選取單位
        # 這裡為了簡化，位置都先寫死或隨機，之後你可以優化找位置的邏輯
        
        if action_cmd == ACTION_BUILD_PROBE:
            if minerals >= 50 and actions.FUNCTIONS.Train_Probe_quick.id in obs.observation.available_actions:
                return actions.FUNCTIONS.Train_Probe_quick("now")
            return self.select_unit(obs, self.nexus_id)

        elif action_cmd == ACTION_BUILD_PYLON:
            if minerals >= 100 and actions.FUNCTIONS.Build_Pylon_screen.id in obs.observation.available_actions:
                target = (random.randint(20, 60), random.randint(20, 60)) # 隨機亂蓋
                return actions.FUNCTIONS.Build_Pylon_screen("now", target)
            return self.select_unit(obs, self.probe_id)

        elif action_cmd == ACTION_BUILD_GATEWAY:
            if minerals >= 150 and actions.FUNCTIONS.Build_Gateway_screen.id in obs.observation.available_actions:
                target = (random.randint(20, 60), random.randint(20, 60))
                return actions.FUNCTIONS.Build_Gateway_screen("now", target)
            return self.select_unit(obs, self.probe_id)
            
        elif action_cmd == ACTION_TRAIN_ZEALOT:
            if minerals >= 100 and actions.FUNCTIONS.Train_Zealot_quick.id in obs.observation.available_actions:
                return actions.FUNCTIONS.Train_Zealot_quick("now")
            return self.select_unit(obs, self.gateway_id)

        return actions.FUNCTIONS.no_op()

    def select_unit(self, obs, unit_id):
        unit_type = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
        y, x = (unit_type == unit_id).nonzero()
        if x.any():
            # 選第一個找到的單位
            target = (int(x[0]), int(y[0]))
            return actions.FUNCTIONS.select_point("select", target)
        return actions.FUNCTIONS.no_op()

    def clear_memory(self):
        self.memory_states = []
        self.memory_actions = []
        self.memory_logprobs = []
        self.memory_rewards = []
        self.memory_values = []

# === 這是你原本的 TerranBot (腳本)，用來當對手 ===
class TerranScriptedBot:
    def __init__(self):
        self.step_cnt = 0
    def step(self, obs):
        self.step_cnt += 1
        # 讓它發呆就好，或者你可以把之前的 TerranBot 程式碼貼回來這裡
        # 為了測試方便，我們先讓它做一個無害的對手
        return actions.FUNCTIONS.no_op()

# === PPO 核心更新算法 ===
def update_ppo(agent):
    rewards = []
    discounted_reward = 0
    
    # 1. 計算折扣回報 (Discounted Rewards)
    # 從最後一步往回推
    for reward in reversed(agent.memory_rewards):
        discounted_reward = reward + GAMMA * discounted_reward
        rewards.insert(0, discounted_reward)
    
    # 正規化獎勵 (讓訓練更穩定)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
    
    # 轉換 list 為 tensor
    old_states = torch.cat(agent.memory_states).detach()
    old_actions = torch.stack(agent.memory_actions).detach()
    old_logprobs = torch.stack(agent.memory_logprobs).detach()
    old_values = torch.cat(agent.memory_values).detach()

    # 2. 計算 Loss (這裡用簡化版的 Actor-Critic Loss，方便理解)
    # Advantage = 實際回報 - Critic預測的分數
    advantages = rewards - old_values.squeeze()
    
    # 計算 Actor Loss (Policy Gradient)
    loss_actor = -(old_logprobs * advantages.detach()).mean()
    
    # 計算 Critic Loss (MSE)
    loss_critic = F.mse_loss(old_values.squeeze(), rewards)
    
    # 總 Loss
    total_loss = loss_actor + 0.5 * loss_critic
    
    # 3. 反向傳播更新
    agent.optimizer.zero_grad()
    total_loss.backward()
    agent.optimizer.step()
    
    return total_loss.item()

def main(argv):
    # Log 設定
    base_dir = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M_PPO")
    log_dir = os.path.join(base_dir, "logs", current_time)
    writer = SummaryWriter(log_dir)
    print(f"PPO 訓練開始! Log 路徑: {log_dir}")

    bot_rl = ProtossRLAgent()
    bot_script = TerranScriptedBot()

    try:
        with sc2_env.SC2Env(
            map_name="Simple64",
            players=[sc2_env.Agent(sc2_env.Race.protoss), sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64),
                use_raw_units=True,
            ),
            step_mul=8,      # 加快節奏
            game_steps_per_episode=1000, 
            visualize=True   # 想看畫面設 True，想快一點設 False
        ) as env:
            
            # 訓練迴圈：跑 100 個 Episode (局)
            for episode in range(100):
                obs_list = env.reset()
                bot_rl.clear_memory() # 每局開始清空記憶
                
                total_reward = 0
                step_count = 0
                
                while True:
                    step_count += 1
                    
                    # 獲取動作
                    action_rl = bot_rl.step(obs_list[0])
                    action_script = bot_script.step(obs_list[1])
                    
                    # 執行
                    obs_list = env.step([action_rl, action_script])
                    
                    # 收集 Reward (注意：這是上一步動作造成的後果)
                    # PySC2 的 reward 通常是勝負(1/-1)或挖礦分數，這裡我們用內建 reward
                    r_rl = obs_list[0].reward
                    bot_rl.memory_rewards.append(r_rl)
                    
                    total_reward += r_rl
                    
                    # 判斷遊戲是否結束
                    if obs_list[0].last():
                        break

                # === 這一局結束，進行學習 (Update) ===
                loss = update_ppo(bot_rl)
                
                print(f"Episode {episode+1} | Reward: {total_reward} | Loss: {loss:.4f}")
                
                # 寫入 TensorBoard
                writer.add_scalar('Train/Reward', total_reward, episode)
                writer.add_scalar('Train/Loss', loss, episode)
                writer.flush()

    except KeyboardInterrupt:
        print("訓練中斷")
    finally:
        writer.close()
        print("訓練結束，Log 已存檔")

if __name__ == "__main__":
    app.run(main)