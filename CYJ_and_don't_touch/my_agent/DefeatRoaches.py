import sys, os, time, csv, numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from absl import app, flags
from pysc2.env import sc2_env
from pysc2.lib import actions, features

# =========================================================
# 🛠 1. 環境路徑與系統設定
# =========================================================
# 根據你的路徑修正為 D:\Game\StarCraft II
SC2_ROOT = r"D:\Game\StarCraft II"
os.environ["SC2PATH"] = SC2_ROOT

# 模型與 Log 儲存路徑
BASE_SAVE_PATH = r"C:\RL_Projects\CYJ_and_don't_touch\my_agent\models\DefeatRoaches"
os.makedirs(BASE_SAVE_PATH, exist_ok=True)

MODEL_NEW = os.path.join(BASE_SAVE_PATH, "fullyconv_new_DefeatRoaches.pth")
MODEL_BEST = os.path.join(BASE_SAVE_PATH, "fullyconv_best_DefeatRoaches.pth")
LOG_CSV = os.path.join(BASE_SAVE_PATH, "training_log_DefeatRoaches.csv")

# 超參數
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA = 0.99
GAE_LAM = 0.95        # GAE 參數
PPO_EPS = 0.2         # PPO Clipped 範圍
BATCH_SIZE = 32       # Mini-batch 大小
UPDATE_EPOCHS = 4
STEP_MUL = 1          # 跳幀設定

# =========================================================
# 🧠 2. 模型架構：FullyConv (支援遷移學習)
# =========================================================
class ConvEncoder(nn.Module):
    """特徵提取器：遷移學習的核心，保留對地圖空間的理解 [cite: 65]"""
    def __init__(self, in_channels=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 5, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU()
        )
    def forward(self, x): return self.net(x)

class PPOAgentNet(nn.Module):
    def __init__(self, action_dim=3):
        super().__init__()
        self.encoder = ConvEncoder()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(32*64*64, 256), nn.ReLU())
        self.critic = nn.Linear(256, 1)
        self.actor_type = nn.Linear(256, action_dim)
        self.actor_spatial = nn.Conv2d(32, 1, 1) # 輸出 64x64 熱圖

    def forward(self, x):
        feat = self.encoder(x)
        latent = self.fc(feat)
        return self.actor_type(latent), self.actor_spatial(feat).view(-1, 64*64), self.critic(latent)

# =========================================================
# ⚙️ 3. 強化學習核心功能 (GAE & PPO)
# =========================================================
def compute_gae(rewards, masks, values, last_v=0):
    """計算廣義優勢估計 (GAE) [cite: 57, 80]"""
    advs = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        next_v = values[t+1] if t+1 < len(values) else last_v
        delta = rewards[t] + GAMMA * next_v * masks[t] - values[t]
        gae = delta + GAMMA * GAE_LAM * masks[t] * gae
        advs[t] = gae
    return advs, advs + values

def train_ppo(model, optimizer, memory):
    """小分組 PPO 更新"""
    s = torch.cat(memory['s'])
    at = torch.tensor(memory['at']).to(DEVICE)
    old_lp = torch.cat(memory['lp']).detach()
    rets = memory['ret']
    advs = memory['adv']

    l_val, e_val = 0, 0
    idx = np.arange(len(s))
    for _ in range(UPDATE_EPOCHS):
        np.random.shuffle(idx)
        for i in range(0, len(idx), BATCH_SIZE):
            mb = idx[i : i + BATCH_SIZE]
            t_logits, _, val = model(s[mb])
            
            dist = Categorical(F.softmax(t_logits, dim=-1))
            new_lp = dist.log_prob(at[mb])
            ratio = torch.exp(new_lp - old_lp[mb])
            
            # Clipped Objective
            s1 = ratio * advs[mb]
            s2 = torch.clamp(ratio, 1-PPO_EPS, 1+PPO_EPS) * advs[mb]
            
            loss = -torch.min(s1, s2).mean() + 0.5 * F.mse_loss(val.squeeze(), rets[mb]) - 0.01 * dist.entropy().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            l_val += loss.item(); e_val += dist.entropy().mean().item()
    return l_val / (UPDATE_EPOCHS * (len(idx)/BATCH_SIZE)), e_val / (UPDATE_EPOCHS * (len(idx)/BATCH_SIZE))

# =========================================================
# 🚀 4. 主執行程序
# =========================================================
def main(unused_argv):
    model = PPOAgentNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    start_ep, best_r = 0, -1.0

    # 繼承機制 (載入權重與優化器梯度動量)
    for p in [MODEL_BEST, MODEL_NEW]:
        if os.path.exists(p):
            ckpt = torch.load(p, map_location=DEVICE)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['opt'])
            start_ep, best_r = ckpt['ep'], ckpt['r']
            print(f"[*] 載入成功: {os.path.basename(p)} | 起始回合: {start_ep}")
            break

    log_f = open(LOG_CSV, 'a', newline=''); writer = csv.writer(log_f)
    if os.path.getsize(LOG_CSV) == 0: writer.writerow(['Episode', 'Reward', 'Loss', 'Entropy'])

    env = sc2_env.SC2Env(map_name="DefeatRoaches", step_mul=STEP_MUL,
                         players=[sc2_env.Agent(sc2_env.Race.terran)],
                         agent_interface_format=features.AgentInterfaceFormat(
                             feature_dimensions=features.Dimensions(screen=64, minimap=64), use_feature_units=True))

    try:
        for ep in range(start_ep + 1, 100000):
            obs = env.reset()[0]
            mem = {'s':[], 'at':[], 'lp':[], 'v':[], 'r':[], 'm':[]}
            ep_reward = 0
            
            while True:
                # 視覺處理與安全檢查
                screen = obs.observation.feature_screen
                rel = screen[features.SCREEN_FEATURES.player_relative.index]
                hp_idx = getattr(features.SCREEN_FEATURES, 'unit_hp_ratio', getattr(features.SCREEN_FEATURES, 'unit_hit_points_ratio', None)).index
                state = np.stack([rel==1, rel==4, screen[hp_idx]/255.0, screen[features.SCREEN_FEATURES.selected.index]], axis=0)
                state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                
                t_logits, s_logits, val = model(state_t)
                
                # Invalid Action Masking
                mask = torch.zeros(3).to(DEVICE)
                avail = obs.observation.available_actions
                if 0 in avail: mask[0] = 1 # NO_OP
                if 7 in avail: mask[1] = 1 # select_army
                if 12 in avail: mask[2] = 1 # Attack_screen
                
                prob = F.softmax(t_logits, dim=-1) * mask + 1e-9
                dist = Categorical(prob / prob.sum())
                a_type = dist.sample()
                
                if a_type == 1: act = actions.FUNCTIONS.select_army("select")
                elif a_type == 2:
                    s_idx = Categorical(F.softmax(s_logits, dim=-1)).sample().item()
                    act = actions.FUNCTIONS.Attack_screen("now", [s_idx%64, s_idx//64])
                else: act = actions.FUNCTIONS.no_op()

                obs = env.step(actions=[act])[0]
                mem['s'].append(state_t); mem['at'].append(a_type.item())
                mem['lp'].append(dist.log_prob(a_type).unsqueeze(0)); mem['v'].append(val)
                mem['r'].append(obs.reward); mem['m'].append(0 if obs.last() else 1)
                ep_reward += obs.reward
                if obs.last(): break

            # 計算 GAE 並存檔
            adv, ret = compute_gae(torch.tensor(mem['r']).to(DEVICE), torch.tensor(mem['m']).to(DEVICE), 
                                   torch.cat(mem['v']).detach().squeeze())
            mem['adv'] = adv; mem['ret'] = ret
            l_v, e_v = train_ppo(model, optimizer, mem)

            save_pkg = {'ep': ep, 'model': model.state_dict(), 'opt': optimizer.state_dict(), 'r': ep_reward}
            torch.save(save_pkg, MODEL_NEW)
            if ep_reward >= best_r:
                best_r = ep_reward; torch.save(save_pkg, MODEL_BEST)
                print(f"[*] 第 {ep} 回合: 更新最佳紀錄 ({best_r})")
            
            writer.writerow([ep, ep_reward, f"{l_v:.4f}", f"{e_v:.4f}"]); log_f.flush()
            print(f"Ep {ep} | Reward: {ep_reward} | Entropy: {e_v:.4f}")

    finally: env.close(); log_f.close()

if __name__ == "__main__":
    flags.DEFINE_string('f', '', 'kernel'); app.run(main)