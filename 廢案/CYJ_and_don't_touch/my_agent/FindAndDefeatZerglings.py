import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import datetime

# ==========================================
# 1. FullyConv 神經網路架構
# ==========================================
class FullyConvSC2Net(nn.Module):
    def __init__(self, screen_channels=6, minimap_channels=4, num_actions=6, resolution=84):
        super(FullyConvSC2Net, self).__init__()
        self.resolution = resolution
        
        # --- 卷積層：Screen 特徵提取 (輸入: ch0~ch5) ---
        self.screen_conv1 = nn.Conv2d(screen_channels, 16, kernel_size=5, stride=1, padding=2)
        self.screen_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # --- 卷積層：Minimap 特徵提取 (輸入: ch6~ch9) ---
        self.minimap_conv1 = nn.Conv2d(minimap_channels, 16, kernel_size=5, stride=1, padding=2)
        self.minimap_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # --- 空間特徵融合 (State Representation) ---
        self.state_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc_state = nn.Linear(64 * resolution * resolution, 256)
        
        # --- 輸出分支 (Policy & Value) ---
        # 1. 離散動作 (Action Type: 0~5)
        self.policy_action = nn.Linear(256, num_actions)
        # 2. 空間動作 (Screen [X, Y])
        self.policy_screen = nn.Conv2d(64, 1, kernel_size=1) 
        # 3. 空間動作 (Minimap [X, Y])
        self.policy_minimap = nn.Conv2d(64, 1, kernel_size=1)
        # 4. 價值評估 (Value)
        self.value_head = nn.Linear(256, 1)

    def forward(self, screen, minimap, last_action=None):
        # 處理 Screen 特徵
        s_out = F.relu(self.screen_conv1(screen))
        s_out = F.relu(self.screen_conv2(s_out))
        
        # 處理 Minimap 特徵
        m_out = F.relu(self.minimap_conv1(minimap))
        m_out = F.relu(self.minimap_conv2(m_out))
        
        # 將畫面與小地圖特徵在通道維度 (dim=1) 拼接
        sm_concat = torch.cat([s_out, m_out], dim=1)
        spatial_features = F.relu(self.state_conv(sm_concat))
        
        # 攤平進入全連接層以評估大局
        flatten_features = spatial_features.view(spatial_features.size(0), -1)
        state_repr = F.relu(self.fc_state(flatten_features))
        
        # --- 價值輸出 (Value) ---
        value = self.value_head(state_repr)
        
        # --- 動作輸出 (Action Logits) ---
        action_logits = self.policy_action(state_repr)
        
        # [核心實作] 無效動作遮罩 (Invalid Action Masking)
        # 規則：如果上一步驟的動作是 1 (Select_Army)，則將這一步驟的 1 機率降為 0 (logit = -1e9)
        if last_action is not None:
            mask = (last_action == 1).unsqueeze(1)
            action_logits = action_logits.masked_fill(mask, -1e9)
            
        action_probs = F.softmax(action_logits, dim=-1)
        
        # --- 空間座標輸出 (轉換為攤平的 84x84 機率分佈) ---
        screen_logits = self.policy_screen(spatial_features).view(-1, self.resolution * self.resolution)
        minimap_logits = self.policy_minimap(spatial_features).view(-1, self.resolution * self.resolution)
        
        screen_probs = F.softmax(screen_logits, dim=-1)
        minimap_probs = F.softmax(minimap_logits, dim=-1)
        
        return action_probs, screen_probs, minimap_probs, value

# ==========================================
# 2. PPO 代理人與損失計算邏輯
# ==========================================
class PPOAgent:
    def __init__(self, net, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, c1=0.5, c2=0.01):
        self.net = net
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.c1 = c1 # Value Loss 的權重
        self.c2 = c2 # Entropy 的權重 (數字越大越鼓勵 AI 亂逛探索)
        self.initial_lr = lr

    # 學習率線性退火 (Learning Rate Linear Annealing)
    def update_learning_rate(self, current_step, total_steps):
        fraction = max(0.0, 1.0 - (current_step / total_steps))
        lr = self.initial_lr * fraction
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    # 計算廣義優勢估計 (GAE)
    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        values = values + [next_value]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        return advantages

    def ppo_update(self, states, actions, old_probs, returns, advantages):
        screen_states, minimap_states, last_actions = states
        action_idx, screen_idx, minimap_idx = actions
        
        # 標準化優勢函數，使訓練更穩定
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 重新計算當前策略的機率與價值
        new_action_probs, new_screen_probs, new_minimap_probs, values = self.net(screen_states, minimap_states, last_actions)
        
        # 取得被選中動作的當前機率
        prob_a = new_action_probs.gather(1, action_idx.unsqueeze(1)).squeeze(1)
        ratio = prob_a / (old_probs + 1e-8)
        
        # PPO Clipped Objective (截斷目標)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value Loss (均方誤差)
        value_loss = F.mse_loss(values.squeeze(1), returns)
        
        # Entropy Loss (資訊熵，促使機率分佈不至於太早收斂到單一動作)
        entropy = -(new_action_probs * torch.log(new_action_probs + 1e-8)).sum(dim=-1).mean()
        
        # 總損失計算
        loss = policy_loss + self.c1 * value_loss - self.c2 * entropy
        
        # 反向傳播
        self.optimizer.zero_grad()
        loss.backward()
        
        # [核心實作] 防止梯度爆炸 (Gradient Clipping)
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
        
        self.optimizer.step()
        
        return policy_loss.item(), value_loss.item(), entropy.item(), loss.item()

# ==========================================
# 3. 訓練迴圈與記錄機制 (CSV & PTH)
# ==========================================
def train_sc2_agent():
    # --- 超參數設定 ---
    total_episodes = 10000
    step_mul = 8       # PySC2 的跳幀設定 (若在此呼叫 env 需留意設定)
    resolution = 84    # 畫面解析度 84x84
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 實例化網路與代理人
    net = FullyConvSC2Net(resolution=resolution).to(device)
    agent = PPOAgent(net)
    
    # --- 設定專屬的存檔路徑 ---
    save_dir = r"C:\RL_Projects\CYJ_and_don't_touch\my_agent\models\FindAndDefeatZerglings"
    
    # 防呆機制：確保資料夾存在，若無則自動建立多層目錄
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"📁 建立全新儲存資料夾: {save_dir}")

    csv_filename = os.path.join(save_dir, f"sc2_training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv")
    log_data = []

    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚀 開始訓練 SC2 AI 模型...")
    print(f"🎯 使用設備: {device} | 📌 Step Multiplier: {step_mul}")
    print(f"💾 檔案儲存目標: {save_dir}")
    print("-" * 60)

    # ================= 訓練主迴圈 =================
    for episode in range(1, total_episodes + 1):
        
        episode_reward = 0
        step_count = 0
        last_action_val = 0 # 初始設為 0 (no_op)
        
        # 收集當前 Episode 軌跡用的 Buffer
        states_s, states_m, last_acts, actions, old_probs, rewards, values, dones = [], [], [], [], [], [], [], []

        done = False
        
        # [替換點]: obs = env.reset() 
        while not done and step_count < 256: # 假設一次軌跡最多走 256 步
            step_count += 1
            
            # [替換點]: 從真實 env 的 obs 解析出特徵。這裡以模擬的常態分佈張量代替
            screen_tensor = torch.randn(1, 6, resolution, resolution).to(device)
            minimap_tensor = torch.randn(1, 4, resolution, resolution).to(device)
            last_act_tensor = torch.tensor([last_action_val], dtype=torch.long).to(device)
            
            # 網路預測 (無梯度，僅為收集資料)
            with torch.no_grad():
                a_probs, s_probs, m_probs, val = net(screen_tensor, minimap_tensor, last_act_tensor)
            
            # 依機率取樣動作
            dist = torch.distributions.Categorical(a_probs)
            action = dist.sample()
            
            # 更新 last_action_val 供下一個 step 進行 Action Masking 檢查
            last_action_val = action.item()
            
            # [替換點]: 模擬環境回饋 (請換成真實的 env.step(action) 解析)
            # 結合 ICM (好奇心) + 血量差分 + 擊殺事件 的自訂 reward
            step_reward = np.random.randn() 
            done = step_count >= 256   
            
            episode_reward += step_reward
            
            # 將資料存入 Buffer
            states_s.append(screen_tensor)
            states_m.append(minimap_tensor)
            last_acts.append(last_act_tensor)
            actions.append(action.item())
            old_probs.append(a_probs[0, action].item())
            rewards.append(step_reward)
            values.append(val.item())
            dones.append(done)

        # ====== 單局結束，準備進行 PPO 更新 ======
        # 1. 學習率線性退火
        current_lr = agent.update_learning_rate(episode, total_episodes)
        
        # 2. 計算下一狀態的價值預估 (如果結束則為 0)
        # [替換點]: next_val = net(next_screen, next_minimap...)[3].item() if not done else 0
        next_val = 0 
        
        # 3. 計算 GAE 與 Returns
        advantages = agent.compute_gae(rewards, values, dones, next_val)
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        # 4. 將 List 轉為 GPU 上的 Tensor 批次
        t_states_s = torch.cat(states_s)
        t_states_m = torch.cat(states_m)
        t_last_acts = torch.stack(last_acts).squeeze(-1) # 形狀: (Batch,)
        t_action_idx = torch.tensor(actions, dtype=torch.long).to(device)
        t_old_probs = torch.tensor(old_probs, dtype=torch.float32).to(device)
        t_returns = torch.tensor(returns, dtype=torch.float32).to(device)
        t_advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        
        # 5. 執行 PPO 更新運算
        p_loss, v_loss, ent, tot_loss = agent.ppo_update(
            states=(t_states_s, t_states_m, t_last_acts),
            actions=(t_action_idx, None, None), # 暫時簡化，不更新空間座標動作的 Loss
            old_probs=t_old_probs,
            returns=t_returns,
            advantages=t_advantages
        )
        
        # --- 列印觀察狀況 (多多 print 以便觀察) ---
        print(f"🔹 Episode {episode:04d} 結束 | 步數: {step_count:03d} | 總獎勵: {episode_reward:+.2f}")
        print(f"   ↳ 📈 Loss: {tot_loss:+.4f} | Policy: {p_loss:+.4f} | Value: {v_loss:+.4f} | Entropy: {ent:.4f} | LR: {current_lr:.6f}")
        
        # --- 將數據存入 CSV 紀錄檔 ---
        log_data.append({
            "Episode": episode,
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Reward": round(episode_reward, 3),
            "PolicyLoss": round(p_loss, 4),
            "ValueLoss": round(v_loss, 4),
            "Loss": round(tot_loss, 4),
            "Entropy": round(ent, 4)
        })
        
        if episode % 10 == 0:
            df = pd.DataFrame(log_data)
            df.to_csv(csv_filename, index=False)
            print(f"   [💾 自動儲存] 已將訓練紀錄同步至 CSV")
            
        # --- 定期儲存模型 (.pth) ---
        if episode % 100 == 0:
            model_path = os.path.join(save_dir, f"sc2_agent_ep{episode}.pth")
            torch.save(net.state_dict(), model_path)
            print(f"   [🧠 模型備份] 已保存模型權重至: sc2_agent_ep{episode}.pth")
        
        if episode % 10 == 0:
             print("-" * 60) # 分隔線讓終端機更好閱讀

if __name__ == "__main__":
    train_sc2_agent()