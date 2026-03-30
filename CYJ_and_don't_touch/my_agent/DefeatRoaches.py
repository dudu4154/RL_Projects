import sys
import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from absl import app, flags
from pysc2.env import sc2_env
from pysc2.lib import actions, features

# --- 彩色監控 ---
def log_status(tag, msg, color="32"):
    print(f"\033[1;{color}m[{time.strftime('%H:%M:%S')}] [{tag}] {msg}\033[0m")

sys.argv = [sys.argv[0]]
FLAGS = flags.FLAGS
if 'f' not in flags.FLAGS: flags.DEFINE_string('f', '', 'kernel')
FLAGS(sys.argv)

log_status("SYSTEM", "💤 V38: 數據補完版 | 實裝 PPO 更新 | 解決 Loss 歸零問題", "34")

# 1. 路徑鎖定
BASE_DIR = r"C:\RL_Projects\CYJ_and_don't_touch\my_agent\models\DefeatRoaches"
BEST_PATH = os.path.join(BASE_DIR, "fullyconv_best_DefeatRoaches.pth")
LOG_PATH = os.path.join(BASE_DIR, "training_log_DefeatRoaches.csv")
os.makedirs(BASE_DIR, exist_ok=True)

os.environ["SC2PATH"] = r"D:\Game\StarCraft II"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 模型定義：LightFullyConv (V37 節能架構) ---
class LightFullyConv(nn.Module):
    def __init__(self, screen_channels):
        super(LightFullyConv, self).__init__()
        self.spatial_path = nn.Sequential(
            nn.Conv2d(screen_channels, 32, 5, padding=2), nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        flat_size = 64 * 32 * 32
        self.actor_type = nn.Sequential(nn.Flatten(), nn.Linear(flat_size, 256), nn.ReLU(), nn.Linear(256, 2))
        self.actor_spatial = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Upsample(size=(64,64), mode='bilinear', align_corners=False)
        )
        self.critic = nn.Sequential(nn.Flatten(), nn.Linear(flat_size, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x):
        feat = self.spatial_path(x)
        return self.actor_type(feat), self.actor_spatial(feat).view(-1, 64*64), self.critic(feat)

def preprocess(obs):
    screen = obs.observation.feature_screen
    rel = screen[features.SCREEN_FEATURES.player_relative.index]
    try: hp_idx = features.SCREEN_FEATURES.unit_hit_points_ratio.index
    except AttributeError: hp_idx = features.SCREEN_FEATURES.unit_hp_ratio.index
    data = np.stack([rel == 1, rel == 4, screen[hp_idx] / 255.0], axis=0).astype(float)
    return torch.FloatTensor(data).unsqueeze(0).to(device)

def get_units_stats(obs):
    if 'feature_units' not in obs.observation: return 0, 0, 0, 0
    units = obs.observation.feature_units
    s_hp, e_hp, s_cnt, e_cnt = 0, 0, 0, 0
    for u in units:
        if u.alliance == 1: s_hp += u.health; s_cnt += 1
        elif u.alliance == 4: e_hp += u.health; e_cnt += 1
    return s_hp, e_hp, s_cnt, e_cnt

def main(unused_argv):
    env = sc2_env.SC2Env(
        map_name="DefeatZerglingsAndBanelings",
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=64, minimap=64),
            use_feature_units=True),
        step_mul=4, visualize=False 
    )

    file_exists = os.path.exists(LOG_PATH)
    log_file = open(LOG_PATH, 'a', newline='', buffering=1)
    csv_writer = csv.writer(log_file)
    if not file_exists: csv_writer.writerow(['Episode', 'Reward', 'Loss', 'Entropy'])

    model = LightFullyConv(3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_r = -float('inf')

    if os.path.exists(BEST_PATH):
        try:
            ckpt = torch.load(BEST_PATH, weights_only=False, map_location=device)
            model.load_state_dict(ckpt['model'])
            best_r = ckpt.get('best_r', -float('inf'))
            log_status("LOAD", f"繼承成功！目前最高分紀錄: {best_r:.1f}", "32")
        except: log_status("LOAD", "架構更動，啟動新訓練", "33")

    try:
        for ep in range(0, 50000):
            obs = env.reset()[0]
            obs = env.step([actions.FUNCTIONS.select_army("select")])[0]
            last_s_hp, last_e_hp, last_s_cnt, last_e_cnt = get_units_stats(obs)
            total_reward = 0
            states, type_acts, spat_acts, log_ps, rewards, values, masks = [], [], [], [], [], [], []

            # --- 1. 收集資料 (關閉梯度以省顯存) ---
            with torch.no_grad():
                while True:
                    state = preprocess(obs)
                    t_logits, s_logits, val = model(state)
                    
                    mask = torch.ones_like(t_logits) * -1e10
                    if 7 in obs.observation.available_actions: mask[0, 0] = 0
                    if 12 in obs.observation.available_actions and (len(obs.observation.single_select) > 0 or len(obs.observation.multi_select) > 0): mask[0, 1] = 0
                    if torch.max(mask) < -1e9: mask[0, 0] = 0
                    
                    dist_t = Categorical(torch.softmax(t_logits + mask, dim=-1))
                    dist_s = Categorical(torch.softmax(s_logits, dim=-1))
                    a_t, a_s = dist_t.sample(), dist_s.sample()
                    
                    y, x = a_s.item() // 64, a_s.item() % 64
                    sc2_act = actions.FUNCTIONS.select_army("select") if a_t == 0 else actions.FUNCTIONS.Attack_screen("now", (x, y))

                    obs = env.step([sc2_act])[0]
                    c_s_hp, c_e_hp, c_s_cnt, c_e_cnt = get_units_stats(obs)
                    step_r = (last_e_hp - c_e_hp) * 0.1 + (last_e_cnt - c_e_cnt) * 7.0 - ((last_s_hp - c_s_hp) * 0.05 + (last_s_cnt - c_s_cnt) * 3.0)
                    if obs.last() and c_e_cnt == 0: step_r += 50.0
                    
                    total_reward += step_r
                    states.append(state); type_acts.append(a_t); spat_acts.append(a_s)
                    log_ps.append(dist_t.log_prob(a_t) + dist_s.log_prob(a_s))
                    values.append(val); rewards.append(torch.FloatTensor([step_r]).to(device)); masks.append(torch.FloatTensor([1.0 - obs.last()]).to(device))
                    last_s_hp, last_e_hp, last_s_cnt, last_e_cnt = c_s_hp, c_e_hp, c_s_cnt, c_e_cnt
                    if obs.last(): break

            # --- 2. [修復] PPO 更新與數據寫入 ---
            curr_loss, curr_ent = 0.0, 0.0
            if len(states) > 1:
                with torch.no_grad(): _, _, next_v = model(preprocess(obs))
                v_t = torch.cat(values).view(-1).detach(); r_t = torch.cat(rewards).view(-1).detach(); m_t = torch.cat(masks).view(-1).detach()
                
                # GAE 優勢計算
                gae, adv = 0, []
                for i in reversed(range(len(rewards))):
                    nv = v_t[i+1] if i+1 < len(v_t) else next_v.view(-1).item()
                    delta = r_t[i] + 0.99 * nv * m_t[i] - v_t[i]
                    gae = delta + 0.99 * 0.95 * m_t[i] * gae
                    adv.insert(0, gae)
                
                b_adv = torch.FloatTensor(adv).to(device); b_ret = b_adv + v_t
                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-5)
                b_s, b_t, b_sp, b_lp = torch.cat(states).to(device), torch.stack(type_acts).to(device), torch.stack(spat_acts).to(device), torch.stack(log_ps).detach().view(-1).to(device)

                # 執行 5 次梯度更新
                for _ in range(5):
                    idx = np.random.permutation(len(states)); i = idx[:64]
                    tl, sl, v = model(b_s[i])
                    nt_d, ns_d = Categorical(torch.softmax(tl, -1)), Categorical(torch.softmax(sl, -1))
                    ratio = torch.exp(nt_d.log_prob(b_t[i]) + ns_d.log_prob(b_sp[i]) - b_lp[i])
                    surr = torch.min(ratio*b_adv[i], torch.clamp(ratio, 0.8, 1.2)*b_adv[i])
                    curr_ent = (nt_d.entropy().mean() + ns_d.entropy().mean()).item()
                    loss = -surr.mean() + 0.5 * nn.MSELoss()(v.view(-1), b_ret[i].view(-1)) - 0.01 * curr_ent
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                curr_loss = loss.item()

            # 寫入 CSV (不再是 0 了！)
            csv_writer.writerow([ep, round(total_reward, 2), round(curr_loss, 4), round(curr_ent, 4)])
            
            if total_reward > best_r:
                best_r = total_reward
                torch.save({'model':model.state_dict(), 'opt':optimizer.state_dict(), 'ep':ep, 'best_r':best_r}, BEST_PATH)
                log_status("BEST", f"🏆 破紀錄: {best_r:.1f} | Loss: {curr_loss:.4f}", "32")
            
            if ep % 5 == 0: log_status("PROGRESS", f"EP {ep} 分數: {total_reward:.1f} | 熵: {curr_ent:.4f}", "37")
            torch.cuda.empty_cache() # 關鍵：每局清理顯存垃圾

    except Exception as e: log_status("ERROR", f"崩潰: {e}", "31")
    finally: log_file.close(); env.close(); sys.exit()

if __name__ == "__main__":
    app.run(main)