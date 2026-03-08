# ppo_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        
        # === 視覺處理層 (CNN) ===
        # 處理畫面特徵 (Screen Features)
        # 輸入維度: num_inputs (例如 feature_screen 的層數), 84x84
        self.conv1 = nn.Conv2d(num_inputs, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        
        # 計算 CNN 輸出後的維度 (84 -> 42 -> 21 -> 11)
        # 32 channel * 11 * 11
        self.flatten_size = 32 * 11 * 11
        
        # 全連接層 (整合資訊)
        self.fc1 = nn.Linear(self.flatten_size, 256)
        
        # === 雙頭輸出 ===
        # 1. Actor (演員): 決定動作的機率 (Policy)
        self.actor = nn.Linear(256, num_actions)
        
        # 2. Critic (評論家): 預測目前盤面好不好 (Value)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        # x 必須是 (Batch, Channel, Height, Width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.reshape(-1, self.flatten_size) # 攤平
        x = F.relu(self.fc1(x))
        
        # 輸出動作機率分佈 (Softmax) 與 價值 (Value)
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        
        return action_probs, state_value