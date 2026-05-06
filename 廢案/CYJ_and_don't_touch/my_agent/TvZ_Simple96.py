import os, sys, time, torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
from pysc2_zerg import ZergEnemy96Bot

os.environ["SC2PATH"] = r"D:\Game\StarCraft II" # 請確認！

# --- 1. AI 視覺神經網路 (FullyConv) ---
class FullyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU()
        )
        self.policy_type = nn.Sequential(nn.Linear(64*64*64, 256), nn.ReLU(), nn.Linear(256, 6))
        self.policy_spatial = nn.Conv2d(64, 1, 1)
        self.critic = nn.Sequential(nn.Linear(64*64*64, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x):
        feat = self.conv(x)
        feat_flat = feat.view(feat.size(0), -1)
        return self.policy_type(feat_flat), self.policy_spatial(feat).view(feat.size(0), -1), self.critic(feat_flat)

# --- 2. 視覺 X 因子處理 (AI 專用) ---
def get_screen_obs(obs):
    scr = obs.observation.feature_screen
    rel = scr[features.SCREEN_FEATURES.player_relative.index]
    utmap = scr[features.SCREEN_FEATURES.unit_type.index]
    hp = scr[features.SCREEN_FEATURES.unit_hit_points_ratio.index] / 255.0
    
    e_mask = (rel == 4).astype(float)
    f_mask = ((rel == 1) & (utmap == 51)).astype(float) # 51=Marauder
    # 4 通道：敵人位置, 我方位置, 敵人 HP, 我方 HP
    return torch.FloatTensor(np.stack([e_mask, f_mask, e_mask*hp, f_mask*hp])).unsqueeze(0)

# --- 3. 人族 RAW 腳本 (快速渡過前期) ---
def terran_raw_script(obs):
    units_list = obs.observation.raw_units
    marines = [u for u in units_list if u.unit_type == 51 and u.alliance == 1]
    if len(marines) >= 5: return None # 夠了，交給 AI
    
    minerals = obs.observation.player.minerals
    depots = [u for u in units_list if u.unit_type == 19]
    barracks = [u for u in units_list if u.unit_type == 21]
    techlabs = [u for u in units_list if u.unit_type == 37]
    scvs = [u for u in units_list if u.unit_type == 45]
    cc = [u for u in units_list if u.unit_type == 18][0]

    if not depots and minerals >= 100:
        return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now", scvs[0].tag, (cc.x+5, cc.y+5))
    if depots and not barracks and minerals >= 150:
        return actions.RAW_FUNCTIONS.Build_Barracks_pt("now", scvs[0].tag, (cc.x-5, cc.y+5))
    if barracks and not techlabs and minerals >= 50:
        return actions.RAW_FUNCTIONS.Build_TechLab_quick("now", barracks[0].tag)
    if techlabs and len(marines) < 5 and minerals >= 100:
        return actions.RAW_FUNCTIONS.Train_Marauder_quick("now", barracks[0].tag)
    return actions.RAW_FUNCTIONS.no_op()

# --- 4. 主執行流程 ---
def main(unused_argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullyConvNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    zerg_bot = ZergEnemy96Bot()

    with sc2_env.SC2Env(
        map_name="Simple96",
        players=[sc2_env.Agent(sc2_env.Race.terran), sc2_env.Agent(sc2_env.Race.zerg)],
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=64, minimap=64),
            use_feature_units=True, use_raw_units=True, use_raw_actions=True),
        step_mul=8, visualize=False
    ) as env:

        for ep in range(1000):
            timesteps = env.reset()
            is_ai_phase = False
            # PPO Buffer
            states, type_acts, spat_acts, log_ps, rewards, values, masks = [], [], [], [], [], [], []

            while True:
                obs_t, obs_z = timesteps[0], timesteps[1]
                z_act = zerg_bot.step(obs_z)

                if not is_ai_phase:
                    # 【腳本階段：RAW 指令】
                    t_act = terran_raw_script(obs_t)
                    if t_act is None: 
                        is_ai_phase = True; continue
                else:
                    # 【AI 階段：Screen 指令 + PPO】
                    # A. 自動相機跟隨 (不計入 PPO 訓練)
                    marines = [u for u in obs_t.observation.raw_units if u.unit_type == 51 and u.alliance == 1]
                    if obs_t.observation.game_loop[0] % 4 == 0 and marines:
                        mx, my = np.mean([u.x for u in marines]), np.mean([u.y for u in marines])
                        obs_t = env.step([actions.FUNCTIONS.move_camera((int(mx), int(my))), actions.RAW_FUNCTIONS.no_op()])[0]

                    # B. PPO Forward
                    state = get_screen_obs(obs_t).to(device)
                    t_logits, s_logits, val = model(state)
                    
                    # C. 無效動作遮罩
                    mask = torch.ones_like(t_logits) * -1e10
                    selected = len(obs_t.observation.multi_select) > 0 or len(obs_t.observation.single_select) > 0
                    if selected: mask[0, 0:4] = 0 # 1-4 動作可用
                    mask[0, 4:6] = 0 # 5-6 動作永遠可用
                    
                    dist_t = Categorical(torch.softmax(t_logits + mask, dim=-1))
                    dist_s = Categorical(torch.softmax(s_logits, dim=-1))
                    a_t, a_s = dist_t.sample(), dist_s.sample()
                    y, x = a_s.item() // 64, a_s.item() % 64

                    # D. 動作映射 (Screen Actions)
                    if a_t == 0: t_act = actions.FUNCTIONS.Move_screen("now", (10, 10)) # 簡化邏輯
                    elif a_t == 1: t_act = actions.FUNCTIONS.Move_screen("now", (50, 50))
                    elif a_t == 2: t_act = actions.FUNCTIONS.Attack_screen("now", (x, y))
                    elif a_t == 3: t_act = actions.FUNCTIONS.Attack_screen("now", (x, y))
                    elif a_t == 4: t_act = actions.FUNCTIONS.select_army("select")
                    else: t_act = actions.FUNCTIONS.select_point("select", (x, y))

                    # 紀錄資料供訓練
                    states.append(state); type_acts.append(a_t); spat_acts.append(a_s)
                    log_ps.append(dist_t.log_prob(a_t) + dist_s.log_prob(a_s))
                    values.append(val); rewards.append(torch.FloatTensor([obs_t.reward]).to(device))
                    masks.append(torch.FloatTensor([1.0 - obs_t.last()]).to(device))

                timesteps = env.step([t_act, z_act])
                if obs_t.last(): break
            
            # 每局結束執行 PPO Update (此處省略具體矩陣運算，同之前 PPO 邏輯)
            print(f"Episode {ep} Finished.")

if __name__ == "__main__":
    app.run(main)