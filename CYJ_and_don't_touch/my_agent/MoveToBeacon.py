import sys
import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from absl import app
from pysc2.env import sc2_env
from pysc2.lib import actions, features

# 1. 環境與設備設定
os.environ["SC2PATH"] = r"D:\Game\StarCraft II"
torch.set_num_threads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 常數與路徑
MAP_NAME = "MoveToBeacon"
SCREEN_SIZE = 64
SAVE_DIR = "models/navigation"
MODEL_PATH = os.path.join(SAVE_DIR, "fullyconv_final_v4.pth")
# CSV 檔名加上後綴
LOG_PATH = os.path.join(SAVE_DIR, "training_log_MoveToBeacon.csv")
os.makedirs(SAVE_DIR, exist_ok=True)

# 動作 ID 對應 (只保留選兵與移動)
ACTION_MAP = {0: 7, 1: 331} 

# =========================================================
# 🧠 FullyConv 神經網路架構
# =========================================================
class FullyConvPPO(nn.Module):
    def __init__(self, state_channels):
        super(FullyConvPPO, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(state_channels, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.actor_spatial = nn.Conv2d(32, 1, kernel_size=1)
        self.actor_type = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * SCREEN_SIZE * SCREEN_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        self.critic = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * SCREEN_SIZE * SCREEN_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        # 正交初始化有助於穩定初期梯度
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))

    def forward(self, x):
        feat = self.conv(x)
        spatial_logits = self.actor_spatial(feat).view(-1, SCREEN_SIZE * SCREEN_SIZE)
        type_logits = self.actor_type(feat)
        value = self.critic(feat)
        return type_logits, spatial_logits, value

def preprocess(obs):
    rel = obs.observation.feature_screen[features.SCREEN_FEATURES.player_relative.index]
    # 通道 0: 我方, 通道 1: 目標
    data = np.stack([rel == 1, rel == 3], axis=0).astype(float)
    return torch.FloatTensor(data).unsqueeze(0).to(device)

# =========================================================
# 🎮 主訓練程式
# =========================================================
def main(unused_argv):
    # 降低點擊頻率：step_mul=14
    env = sc2_env.SC2Env(
        map_name=MAP_NAME,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=SCREEN_SIZE, minimap=SCREEN_SIZE),
            use_raw_units=True),
        step_mul=14, 
        visualize=True
    )
    
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'w', newline='') as f:
            csv.writer(f).writerow(['Episode', 'Reward', 'Loss', 'Entropy', 'LR'])

    try:
        print(f"🔥 動力引擎已就緒: {device}")
        model = FullyConvPPO(state_channels=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=3e-4)
        
        start_ep, best_r = 0, -float('inf')
        if os.path.exists(MODEL_PATH):
            ckpt = torch.load(MODEL_PATH, weights_only=False, map_location=device)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['opt'])
            start_ep, best_r = ckpt['ep'] + 1, ckpt['best_r']

        for ep in range(start_ep, 10000):
            obs = env.reset()[0]
            total_reward, step_count = 0, 0
            states, type_acts, spat_acts, log_ps, rewards, values, masks = [], [], [], [], [], [], []
            
            while True:
                step_count += 1
                state = preprocess(obs)
                avail_ids = obs.observation.available_actions
                is_selected = len(obs.observation.single_select) > 0 or len(obs.observation.multi_select) > 0
                
                # --- 智慧動作遮罩 ---
                t_logits, s_logits, val = model(state)
                mask = torch.ones_like(t_logits) * -1e8
                if 7 in avail_ids: mask[0, 0] = 0 # 允許選兵
                if 331 in avail_ids and is_selected: mask[0, 1] = 0 # 允許移動
                if torch.max(mask) < -1e7: mask[0, 0] = 0 # 安全退路
                
                t_dist = Categorical(torch.softmax(t_logits + mask, dim=-1))
                a_type = t_dist.sample()
                s_dist = Categorical(torch.softmax(s_logits, dim=-1))
                a_spatial = s_dist.sample()
                
                # --- 座標系修正 ---
                idx = a_spatial.item()
                y, x = idx // SCREEN_SIZE, idx % SCREEN_SIZE
                
                sc2_act = actions.FUNCTIONS.no_op()
                if a_type == 0: 
                    sc2_act = actions.FUNCTIONS.select_army("select")
                elif a_type == 1:
                    sc2_act = actions.FUNCTIONS.Move_screen("now", (x, y))

                # --- 偵錯文字 ---
                if step_count % 50 == 0:
                    pt = torch.softmax(t_logits + mask, dim=-1)[0]
                    print(f"[{ep}-{step_count}] 動作:{a_type.item()} 座標:({x},{y}) 選兵:{pt[0]:.2f} 移動:{pt[1]:.2f}")

                # 獎勵塑造
                rel_layer = obs.observation.feature_screen[features.SCREEN_FEATURES.player_relative.index]
                uy, ux = np.where(rel_layer == 1); by, bx = np.where(rel_layer == 3)
                shaping = 0
                if len(ux) > 0 and len(bx) > 0:
                    d = np.sqrt((ux[0] - bx[0])**2 + (uy[0] - by[0])**2)
                    shaping = np.exp(-d / 10.0) * 1.0
                
                states.append(state); type_acts.append(a_type); spat_acts.append(a_spatial)
                log_ps.append(t_dist.log_prob(a_type) + s_dist.log_prob(a_spatial))
                values.append(val)
                masks.append(torch.FloatTensor([1.0]).to(device))
                
                obs = env.step([sc2_act])[0]
                step_r = obs.reward + shaping - 0.005 
                total_reward += step_r
                rewards.append(torch.FloatTensor([step_r]).to(device))
                
                # 縮短步數：400 步
                if obs.last() or step_count > 400: break

            # =========================================================
            # 🚀 [維度修復版] PPO 更新邏輯 (GAE)
            # =========================================================
            if len(states) > 1:
                _, _, last_val = model(preprocess(obs))
                
                # 確保所有張量都是 1D，防止廣播錯誤
                v_tensor = torch.cat(values).view(-1).detach()
                mask_tensor = torch.cat(masks).view(-1).detach()
                reward_tensor = torch.cat(rewards).view(-1).detach()
                
                gae, adv = 0, []
                for i in reversed(range(len(rewards))):
                    next_val = v_tensor[i+1] if i + 1 < len(v_tensor) else last_val.item()
                    delta = reward_tensor[i] + 0.99 * next_val * mask_tensor[i] - v_tensor[i]
                    gae = delta + 0.99 * 0.95 * mask_tensor[i] * gae
                    adv.insert(0, gae)
                
                b_adv = torch.FloatTensor(adv).to(device)
                b_ret = b_adv + v_tensor
                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-5)
                
                b_s = torch.cat(states).to(device)
                b_t, b_sp = torch.stack(type_acts).to(device), torch.stack(spat_acts).to(device)
                b_lp = torch.stack(log_ps).detach().view(-1).to(device)

                # 小分組訓練：Batch Size 64, 10 Epochs
                for _ in range(10):
                    indices = np.arange(len(states)); np.random.shuffle(indices)
                    for start in range(0, len(states), 64):
                        idx = indices[start : start + 64]
                        tl, sl, v = model(b_s[idx])
                        
                        nt_dist = Categorical(torch.softmax(tl, dim=-1))
                        ns_dist = Categorical(torch.softmax(sl, dim=-1))
                        new_lp = nt_dist.log_prob(b_t[idx]) + ns_dist.log_prob(b_sp[idx])
                        
                        ratio = torch.exp(new_lp - b_lp[idx])
                        surr1 = ratio * b_adv[idx]
                        surr2 = torch.clamp(ratio, 0.8, 1.2) * b_adv[idx]
                        
                        ent = nt_dist.entropy().mean() + ns_dist.entropy().mean()
                        
                        # 維度絕對對齊：使用 .flatten() 保護
                        loss = -torch.min(surr1, surr2).mean() + 0.5 * nn.MSELoss()(v.flatten(), b_ret[idx].flatten()) - 0.01 * ent
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                
                curr_loss, curr_ent = loss.item(), ent.item()

            with open(LOG_PATH, 'a', newline='') as f:
                csv.writer(f).writerow([ep, round(total_reward, 2), round(curr_loss, 4), round(curr_ent, 4), 3e-4])
            
            if total_reward > best_r:
                best_r = total_reward
                torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(), 'ep': ep, 'best_r': best_r}, MODEL_PATH)
            
            print(f"🏁 EP {ep} | 分數: {total_reward:.2f} | Entropy: {curr_ent:.4f} (更新完成)")

    except KeyboardInterrupt: pass
    finally:
        env.close()
        sys.exit()

if __name__ == "__main__":
    app.run(main)