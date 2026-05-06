import os, sys, csv, time, datetime, glob, torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from absl import app
from pysc2.env import sc2_env, environment
from pysc2.lib import actions, features, units

# =========================================================
# 🚩 1. 系統路徑與路徑設定
# =========================================================
os.environ["SC2PATH"] = r"D:\Game\StarCraft II"

SAVE_DIR = r"C:\RL_Projects\CYJ_and_don't_touch\my_agent\models\DefeatRoaches"
LOG_PATH = os.path.join(SAVE_DIR, "training_log.csv")
os.makedirs(SAVE_DIR, exist_ok=True)

# 訓練超參數 (完全對應 PPO 規範)
SCREEN_SIZE = 64
STEP_MUL = 8
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
ENTROPY_COEF = 0.05
BATCH_SIZE = 64      # 小分組訓練模式
LR_START = 1e-4
LR_MIN = 1e-5
ANNEAL_EPISODES = 30000 
MAX_EPISODES = 100000

# 戰鬥權重 (LTD2)
MARAUDER_DPS, ROACH_DPS = 20, 8
LTD2_ALPHA = 0.01

# =========================================================
# 🧠 2. FullyConv PPO 神經網路 (10通道輸入)
# =========================================================
class FullyConvPPO(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(FullyConvPPO, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU()
        )
        self.action_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * SCREEN_SIZE * SCREEN_SIZE, 256), nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        self.screen_head = nn.Conv2d(64, 1, kernel_size=1)
        self.minimap_head = nn.Conv2d(64, 1, kernel_size=1)
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * SCREEN_SIZE * SCREEN_SIZE, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x, action_mask):
        feat = self.conv(x)
        act_logits = torch.clamp(self.action_head(feat), -10, 10)
        # 🚩 動作遮罩：不可選動作機率設為 -1e9 (Logits)
        act_logits = act_logits.masked_fill(action_mask == 0, -1e9)
        act_probs = torch.softmax(act_logits, dim=-1)
        
        screen_probs = torch.softmax(self.screen_head(feat).view(-1, SCREEN_SIZE**2), dim=-1)
        minimap_probs = torch.softmax(self.minimap_head(feat).view(-1, SCREEN_SIZE**2), dim=-1)
        value = self.value_head(feat)
        return act_probs, screen_probs, minimap_probs, value

# =========================================================
# 🛠️ 3. 狀態與獎勵工具 (整合 LTD2 與 最短距離)
# =========================================================
def get_state(obs):
    """ 依要求提取 10 通道視覺空間 """
    s = obs.observation.feature_screen
    m = obs.observation.feature_minimap
    layers = [
        s.player_relative, s.unit_type, s.selected, s.unit_hit_points_ratio,
        s.height_map, s.pathable, 
        m.player_relative, m.camera, m.selected, m.visibility_map
    ]
    return torch.from_numpy(np.stack(layers).astype(np.float32))

class RewardCalculator:
    def __init__(self):
        self.last_f_ltd2 = 0
        self.last_e_ltd2 = 0
        self.last_min_dist = 99.0

    def calculate(self, obs):
        r_env = obs.reward 
        units_list = obs.observation.feature_units
        friendly = [u for u in units_list if u.alliance == 1]
        enemy = [u for u in units_list if u.alliance == 4]

        # 計算當前戰場 LTD2 價值
        curr_f_ltd2 = sum([u.health * MARAUDER_DPS for u in friendly])
        curr_e_ltd2 = sum([u.health * ROACH_DPS for u in enemy])

        # 🚩 最短敵我距離計算 (用於拉打)
        current_min_dist = 99.0
        if friendly and enemy:
            fx, fy = np.array([u.x for u in friendly]), np.array([u.y for u in friendly])
            ex, ey = np.array([u.x for u in enemy]), np.array([u.y for u in enemy])
            dists = np.sqrt((fx[:, None] - ex)**2 + (fy[:, None] - ey)**2)
            current_min_dist = np.min(dists)

        # 安全判定：開局或擊殺瞬間不計 LTD2 差分，防爆
        if obs.first() or r_env > 0:
            self.last_f_ltd2, self.last_e_ltd2, self.last_min_dist = curr_f_ltd2, curr_e_ltd2, current_min_dist
            return float(r_env)

        # LTD2 差分獎勵 (敵損 - 我損)
        r_ltd2 = ((self.last_e_ltd2 - curr_e_ltd2) - (self.last_f_ltd2 - curr_f_ltd2)) * LTD2_ALPHA
        
        # 距離拉打加分：射程邊緣 [5,6] 給予麵包屑獎勵
        r_dist = 0.01 if 5.0 <= current_min_dist <= 6.0 else (-0.01 if current_min_dist < 4.0 else 0)

        total_r = r_env + r_ltd2 + r_dist - 0.01 # 含時間懲罰
        self.last_f_ltd2, self.last_e_ltd2, self.last_min_dist = curr_f_ltd2, curr_e_ltd2, current_min_dist
        
        return float(np.clip(total_r, -0.5, 0.5))

def cleanup_old_models(save_dir, keep_last=5):
    files = sorted(glob.glob(os.path.join(save_dir, "defeat_roaches_ep*.pth")), key=os.path.getctime)
    if len(files) >= keep_last:
        for f in files[:(len(files) - keep_last + 1)]:
            try: os.remove(f)
            except: pass

# =========================================================
# 📈 4. PPO 更新核心 (包含 Mini-batch 與 梯度裁剪)
# =========================================================
def ppo_update(model, optimizer, states, actions, log_probs, returns, advantages, masks_list, device):
    st = torch.stack(states).to(device)
    ret = torch.tensor(returns, dtype=torch.float32).to(device).view(-1, 1)
    adv = torch.tensor(advantages, dtype=torch.float32).to(device).view(-1, 1)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    msk = torch.stack(masks_list).to(device)

    optimizer.zero_grad()
    p, sp, _, values = model(st, msk)
    
    # 🚩 解決 Simplex 錯誤
    def force_simplex(t):
        t64 = t.to(torch.float64) + 1e-12
        return (t64 / t64.sum(dim=-1, keepdim=True)).to(torch.float32)

    p, sp = force_simplex(p), force_simplex(sp)
    dist_a, dist_s = Categorical(p), Categorical(sp)
    
    a_t = torch.tensor([a[0] for a in actions], dtype=torch.long).to(device)
    s_t = torch.tensor([a[1] for a in actions], dtype=torch.long).to(device)
    
    new_lp = dist_a.log_prob(a_t) + dist_s.log_prob(s_t)
    old_lp = torch.tensor(log_probs, dtype=torch.float32).to(device)

    ratio = torch.exp(new_lp - old_lp)
    surr1, surr2 = ratio * adv.squeeze(), torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * adv.squeeze()
    
    p_loss = -torch.min(surr1, surr2).mean()
    v_loss = 0.5 * nn.MSELoss()(values, ret)
    ent = dist_a.entropy().mean()
    
    loss = p_loss + v_loss - ENTROPY_COEF * ent
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 0.5) # 防止梯度爆炸
    
    valid = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    if valid: optimizer.step()
    return p_loss.item(), v_loss.item(), loss.item(), ent.item()

# =========================================================
# 🕹️ 5. 訓練主程式 (含新高紀錄與全滅觸發)
# =========================================================
def main(unused_argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullyConvPPO(10, 6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR_START)
    reward_tool = RewardCalculator()
    
    start_ep, best_reward = 0, -999.0

    # 🔍 搜尋進度 (不讀取 MoveToBeacon)
    dr_files = sorted(glob.glob(os.path.join(SAVE_DIR, "defeat_roaches_ep*.pth")), key=os.path.getctime, reverse=True)
    if dr_files:
        try:
            ckpt = torch.load(dr_files[0], map_location=device)
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            start_ep = ckpt['epoch'] + 1
            best_reward = ckpt.get('best_reward', -999.0)
            print(f"[*] 🔄 續傳成功: {os.path.basename(dr_files[0])} (Ep {start_ep})")
        except: print("⚠️ [跳過] 損壞檔案...")

    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'w', newline='') as f:
            csv.writer(f).writerow(['Episode', 'Timestamp', 'Reward', 'PolicyLoss', 'ValueLoss', 'Loss', 'Entropy', 'LR'])

    env = sc2_env.SC2Env(map_name="DefeatRoaches", 
                        players=[sc2_env.Agent(sc2_env.Race.terran)],
                        agent_interface_format=features.AgentInterfaceFormat(
                            feature_dimensions=features.Dimensions(screen=64, minimap=64),
                            use_feature_units=True),
                        step_mul=STEP_MUL)

    try:
        for ep in range(start_ep, MAX_EPISODES):
            # 學習率線性退火
            lr_prog = min(1.0, ep / ANNEAL_EPISODES)
            current_lr = LR_START - (LR_START - LR_MIN) * lr_prog
            for pg in optimizer.param_groups: pg['lr'] = current_lr
            
            obs = env.reset()[0]
            done, ep_reward, last_act = False, 0, 0
            force_select = True 
            last_enemy_count = 99
            
            states, actions_taken, log_probs_list, rewards, values, masks_list, is_terminals = [], [], [], [], [], [], []

            print(f"\n--- [Ep {ep} 開始] LR: {current_lr:.2e} | Best: {best_reward:.2f} ---")

            while not done:
                state_cpu = get_state(obs)
                avail = obs.observation.available_actions
                units_list = obs.observation.feature_units
                
                # --- 🚩 核心：無效動作遮罩 ---
                mask = torch.zeros(6).to(device)
                is_selected = any(u.is_selected for u in units_list if u.alliance == 1)

                if force_select or not is_selected:
                    mask[1] = 1 # 沒選人時強制 Action 1
                else:
                    mask[0] = 1 # No-op
                    if 7 in avail and last_act != 1: mask[1] = 1 # 禁止連續 Action 1
                    if 331 in avail: mask[2] = 1
                    if 12 in avail:  mask[3] = 1 
                    if (12 in avail or 331 in avail): mask[0] = 0 # 交戰中禁 No-op

                with torch.no_grad():
                    p, sp, mp, v = model(state_cpu.unsqueeze(0).to(device), mask.unsqueeze(0))
                
                # 採樣
                p_s = (p + 1e-12) / (p.sum() + 1e-12)
                sp_s = (sp + 1e-12) / (sp.sum() + 1e-12)
                dist_a, dist_s = Categorical(p_s), Categorical(sp_s)
                act_idx, s_idx = dist_a.sample(), dist_s.sample()
                
                if force_select: force_select = False
                last_act = act_idx.item()
                target_s = [int(s_idx.item() % 64), int(s_idx.item() // 64)]
                
                if last_act == 1: act = actions.FunctionCall(7, [[0]])
                elif last_act == 2: act = actions.FunctionCall(331, [[0], target_s])
                elif last_act == 3: act = actions.FunctionCall(12, [[0], target_s])
                else: act = actions.FunctionCall(0, [])

                try: next_obs = env.step([act])[0]
                except ValueError: continue
                
                reward = reward_tool.calculate(next_obs)
                
                # 🚩 觸發：敵方全滅戰報
                curr_enemy_count = len([u for u in next_obs.observation.feature_units if u.alliance == 4])
                if last_enemy_count > 0 and curr_enemy_count == 0:
                    friendly_hp = sum([u.health for u in next_obs.observation.feature_units if u.alliance == 1])
                    print(f"  🔥 [戰報] 敵方全滅！ Step: {len(states)} | Reward: {reward:+.2f} | 殘存血量: {friendly_hp}")
                last_enemy_count = curr_enemy_count

                states.append(state_cpu); actions_taken.append((last_act, s_idx.item()))
                log_probs_list.append((dist_a.log_prob(act_idx) + dist_s.log_prob(s_idx)).item())
                rewards.append(reward); values.append(v.item()); masks_list.append(mask.cpu())
                is_terminals.append(0 if next_obs.last() else 1)
                
                ep_reward += reward
                obs = next_obs
                if next_obs.last(): done = True

            # 更新 PPO
            with torch.no_grad():
                next_v = model(get_state(obs).unsqueeze(0).to(device), torch.eye(6)[0].to(device).unsqueeze(0))[3].item()
            rets, gae, v_plus = [], 0, values + [next_v]
            for i in reversed(range(len(rewards))):
                delta = rewards[i] + GAMMA * v_plus[i+1] * is_terminals[i] - v_plus[i]
                gae = delta + GAMMA * GAE_LAMBDA * is_terminals[i] * gae
                rets.insert(0, gae + v_plus[i])
            
            advs = [r - v for r, v in zip(rets, values)]
            pl, vl, tl, ent = ppo_update(model, optimizer, states, actions_taken, log_probs_list, rets, advs, masks_list, device)

            # 🚩 破紀錄輸出
            if ep_reward > best_reward:
                best_reward = ep_reward
                torch.save({'epoch': ep, 'best_reward': best_reward, 'model_state': model.state_dict()}, 
                           os.path.join(SAVE_DIR, "defeat_roaches_best.pth"))
                print(f"⭐ [系統] 創下新高: {best_reward:.2f}")

            with open(LOG_PATH, 'a', newline='') as f:
                csv.writer(f).writerow([ep, time.strftime("%H:%M:%S"), round(ep_reward, 2), f"{pl:.6f}", f"{vl:.6f}", f"{tl:.6f}", f"{ent:.4f}", f"{current_lr:.2e}"])
            
            if ep % 10 == 0:
                cleanup_old_models(SAVE_DIR, 5)
                torch.save({'epoch': ep, 'best_reward': best_reward, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, 
                           os.path.join(SAVE_DIR, f"defeat_roaches_ep{ep}.pth"))
            torch.cuda.empty_cache()

    finally: env.close()

if __name__ == "__main__": app.run(main)