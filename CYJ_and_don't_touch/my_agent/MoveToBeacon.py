import os
import sys
import csv
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from absl import app

# =========================================================
# 🚩 1. 環境變數設定
# =========================================================
os.environ["SC2PATH"] = r"D:\Game\StarCraft II"
if not os.path.exists(os.environ["SC2PATH"]):
    print(f"❌ 警告：找不到路徑 {os.environ['SC2PATH']}，請確認硬碟掛載狀況")

from pysc2.env import sc2_env, environment  
from pysc2.lib import actions, features

# =========================================================
# 🚩 2. 常數與路徑設定
# =========================================================
SAVE_DIR = r"C:\RL_Projects\CYJ_and_don't_touch\my_agent\models\MoveToBeacon"
os.makedirs(SAVE_DIR, exist_ok=True)
LOG_PATH = os.path.join(SAVE_DIR, "training_log.csv")

SCREEN_SIZE = 64
STEP_MUL = 8
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
ENTROPY_COEF = 0.005  # 調低熵係數，讓模型穩定收斂

# =========================================================
# 🎮 輸出空間 1：離散動作 (Discrete Action)
# =========================================================
ACTION_MAP = {
    0: actions.FUNCTIONS.no_op.id,          
    1: actions.FUNCTIONS.select_army.id,    
    2: actions.FUNCTIONS.Move_screen.id,    
    3: actions.FUNCTIONS.Attack_screen.id,  
    4: actions.FUNCTIONS.move_camera.id,    
    5: actions.FUNCTIONS.Move_minimap.id    
}
NUM_ACTIONS = len(ACTION_MAP)

# =========================================================
# 🧠 3. FullyConv PPO 神經網路
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
        
        act_logits = self.action_head(feat)
        act_logits = act_logits.masked_fill(action_mask == 0, -1e9)
        act_probs = torch.softmax(act_logits, dim=-1)
        
        screen_probs = torch.softmax(self.screen_head(feat).view(-1, 4096), dim=-1)
        minimap_probs = torch.softmax(self.minimap_head(feat).view(-1, 4096), dim=-1)
        
        value = self.value_head(feat)
        return act_probs, screen_probs, minimap_probs, value

# =========================================================
# 🛠️ 4. 輔助工具函數
# =========================================================
def get_state(obs):
    s = obs.observation.feature_screen
    m = obs.observation.feature_minimap
    layers = [
        s.player_relative, s.unit_type, s.selected, s.unit_hit_points_ratio,
        s.height_map, s.visibility_map,
        m.player_relative, m.camera, m.selected, m.visibility_map
    ]
    state = np.stack(layers).astype(np.float32)
    return torch.from_numpy(state)

def get_dist(obs):
    player_relative = obs.observation.feature_screen.player_relative
    unit_y, unit_x = (player_relative == 1).nonzero()
    beacon_y, beacon_x = (player_relative == 3).nonzero()
    
    if len(unit_x) > 0 and len(beacon_x) > 0:
        return np.sqrt((unit_x.mean() - beacon_x.mean())**2 + (unit_y.mean() - beacon_y.mean())**2)
    return None 

def get_latest_model_path(save_dir):
    """搜尋資料夾，找出數字最大的存檔 (例如 pysc2_ppo_ep120.pth)"""
    files = glob.glob(os.path.join(save_dir, "pysc2_ppo_ep*.pth"))
    if not files:
        return None
    # 依照檔名中的數字排序
    latest_file = max(files, key=lambda f: int(os.path.basename(f).split('_ep')[1].split('.pth')[0]))
    return latest_file

def cleanup_old_models(save_dir, keep_last=5):
    """刪除舊存檔，只保留最新的幾份"""
    files = glob.glob(os.path.join(save_dir, "pysc2_ppo_ep*.pth"))
    if len(files) > keep_last:
        files.sort(key=lambda f: int(os.path.basename(f).split('_ep')[1].split('.pth')[0]))
        for f in files[:-keep_last]:
            os.remove(f)
            print(f"🗑️ [清理] 刪除舊存檔: {os.path.basename(f)}")

# =========================================================
# 📈 5. PPO 更新函數
# =========================================================
def compute_gae(next_value, rewards, masks, values):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + GAMMA * values[step + 1] * masks[step] - values[step]
        gae = delta + GAMMA * GAE_LAMBDA * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def ppo_update(model, optimizer, states, actions, log_probs, returns, advantages, masks_list):
    states = torch.stack(states)
    returns = torch.tensor(returns, dtype=torch.float32).to(states.device).view(-1, 1)
    advantages = torch.tensor(advantages, dtype=torch.float32).to(states.device).view(-1, 1)
    
    historical_masks = torch.stack(masks_list).to(states.device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) 

    optimizer.zero_grad()
    
    new_act_probs, new_s_probs, new_m_probs, values = model(states, historical_masks)
    
    act_dist = Categorical(new_act_probs)
    scr_dist = Categorical(new_s_probs)
    min_dist = Categorical(new_m_probs)
    
    act_tensors = torch.tensor([a[0] for a in actions], dtype=torch.long).to(states.device)
    scr_tensors = torch.tensor([a[1] for a in actions], dtype=torch.long).to(states.device)
    min_tensors = torch.tensor([a[2] for a in actions], dtype=torch.long).to(states.device)
    
    new_log_probs = act_dist.log_prob(act_tensors) + scr_dist.log_prob(scr_tensors) + min_dist.log_prob(min_tensors)
    old_log_probs = torch.tensor(log_probs, dtype=torch.float32).to(states.device)

    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages.squeeze()
    surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages.squeeze()
    
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = nn.MSELoss()(values, returns)
    entropy = act_dist.entropy().mean() + scr_dist.entropy().mean() + min_dist.entropy().mean()
    
    loss = policy_loss + 0.5 * value_loss - ENTROPY_COEF * entropy
    loss.backward()
    
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()
    
    return policy_loss.item(), value_loss.item(), loss.item(), entropy.item()

# =========================================================
# 🕹️ 6. 訓練主程式
# =========================================================
def main(unused_argv):
    print("--- [系統] 初始化星際爭霸 II 環境... ---")
    env = sc2_env.SC2Env(
        map_name="MoveToBeacon",
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=SCREEN_SIZE, minimap=SCREEN_SIZE),
            use_feature_units=True),
        step_mul=STEP_MUL,
        visualize=False) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullyConvPPO(10, NUM_ACTIONS).to(device) 
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    start_ep = 0
    latest_model = get_latest_model_path(SAVE_DIR)
    
    if latest_model:
        try:
            checkpoint = torch.load(latest_model)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            # 如果存檔是 ep10，我們就從 ep11 繼續
            start_ep = checkpoint['epoch'] + 1 
            print(f"--- [系統] 成功繼承最新存檔: {os.path.basename(latest_model)}，從 Episode {start_ep} 繼續 ---")
        except Exception as e:
            print(f"❌ 讀取存檔失敗: {e}")
            env.close()
            return
    else:
        print("--- [系統] 找不到任何存檔，全新開始訓練 ---")

    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'w', newline='') as f:
            csv.writer(f).writerow(['Episode', 'Timestamp', 'Reward', 'PolicyLoss', 'ValueLoss', 'Loss', 'Entropy'])

    try:
        has_selected_army = False 
        last_dist = None
        
        for ep in range(start_ep, 100000):
            
            # 學習率線性退火
            lr = 3e-4 * (1.0 - (ep / 100000))
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(lr, 1e-6)
                
            obs = env.reset()[0]
            done = False
            total_reward = 0
            step_count = 0 
            
            last_action_was_select = False 
            last_dist = get_dist(obs)
            
            states, actions_taken, log_probs_list, rewards, values, is_terminals, masks_list = [], [], [], [], [], [], []

            print(f"\n========== 🎬 開始 Episode {ep} (LR: {max(lr, 1e-6):.2e}) ==========")

            while not done:
                step_count += 1
                
                if obs.step_type == environment.StepType.FIRST and step_count > 1:
                    last_action_was_select = False      
                    last_dist = get_dist(obs)      

                state = get_state(obs).to(device)
                
                avail = obs.observation.available_actions
                mask = torch.zeros(NUM_ACTIONS).to(device)
                
                can_select = (ACTION_MAP[1] in avail) and (not last_action_was_select)
                can_move = (ACTION_MAP[2] in avail) and last_action_was_select
                
                if can_select:
                    mask[1] = 1 
                elif can_move:
                    mask[2] = 1 
                else:
                    mask[0] = 1 
                
                with torch.no_grad():
                    probs, s_probs, m_probs, val = model(state.unsqueeze(0), mask.unsqueeze(0))
                
                act_dist = Categorical(probs)
                scr_dist = Categorical(s_probs)
                min_dist = Categorical(m_probs)
                
                act_idx = act_dist.sample()
                screen_idx = scr_dist.sample()
                minimap_idx = min_dist.sample()
                
                total_log_prob = act_dist.log_prob(act_idx) + scr_dist.log_prob(screen_idx) + min_dist.log_prob(minimap_idx)
                
                last_action_was_select = (act_idx.item() == 1)
                
                target_s = [screen_idx.item() % SCREEN_SIZE, screen_idx.item() // SCREEN_SIZE]
                target_m = [minimap_idx.item() % SCREEN_SIZE, minimap_idx.item() // SCREEN_SIZE]
                
                act_id = ACTION_MAP[act_idx.item()]
                
                if act_id == actions.FUNCTIONS.select_army.id: 
                    final_action = actions.FunctionCall(act_id, [[0]])
                elif act_id in [actions.FUNCTIONS.Move_screen.id, actions.FUNCTIONS.Attack_screen.id]: 
                    final_action = actions.FunctionCall(act_id, [[0], target_s]) 
                elif act_id == actions.FUNCTIONS.move_camera.id: 
                    final_action = actions.FunctionCall(act_id, [target_m]) 
                elif act_id == actions.FUNCTIONS.Move_minimap.id: 
                    final_action = actions.FunctionCall(act_id, [[0], target_m]) 
                else:
                    final_action = actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
                
                next_obs = env.step([final_action])[0]
                
                curr_dist = get_dist(next_obs)
                pysc2_reward = next_obs.reward
                is_done = (next_obs.step_type == environment.StepType.LAST)

                if next_obs.step_type == environment.StepType.FIRST:
                    reward = 0.0
                    last_dist = curr_dist
                elif pysc2_reward > 0: 
                    reward = 1.0  
                    last_dist = curr_dist 
                else:
                    if last_dist is not None and curr_dist is not None:
                        reward = (last_dist - curr_dist) * 0.1 - 0.01
                    else:
                        reward = -0.01
                    last_dist = curr_dist

                states.append(state)
                actions_taken.append((act_idx.item(), screen_idx.item(), minimap_idx.item()))
                log_probs_list.append(total_log_prob.item())
                rewards.append(reward)
                values.append(val.item())
                masks_list.append(mask) 
                
                act_name_dict = {0: 'no_op', 1: 'select', 2: 'move_s', 3: 'atk_s', 4: 'cam_m', 5: 'move_m'}
                act_name = act_name_dict[act_idx.item()]
                prob_str = ", ".join([f"{p:.2f}" for p in probs[0].tolist()]) 
                print(f"[Step {step_count:03d}] 動作: {act_name:<7} | 機率分佈: [{prob_str}] | 獎勵: {reward:+.3f}")
                
                obs = next_obs
                total_reward += reward
                
                if is_done:
                    done = True
                    is_terminals.append(0) 
                else:
                    is_terminals.append(1)

            with torch.no_grad():
                final_mask = torch.zeros(NUM_ACTIONS).to(device)
                final_mask[0] = 1 
                _, _, _, next_val = model(get_state(obs).to(device).unsqueeze(0), final_mask.unsqueeze(0))
                next_val = next_val.item()
            
            returns = compute_gae(next_val, rewards, is_terminals, values)
            advantages = [ret - val for ret, val in zip(returns, values)]
            
            p_loss, v_loss, t_loss, ent = ppo_update(model, optimizer, states, actions_taken, log_probs_list, returns, advantages, masks_list)
            
            print(f"✅ Episode {ep} 結算 | 總獎勵: {total_reward:.2f} | 總損失: {t_loss:.4f} | Entropy: {ent:.4f}")
            
            with open(LOG_PATH, 'a', newline='') as f:
                csv.writer(f).writerow([ep, time.strftime("%Y-%m-%d %H:%M:%S"), round(total_reward, 2), p_loss, v_loss, t_loss, ent])
                
            # 每 10 回合存檔一次，並自動清理舊檔案
            if ep > 0 and ep % 10 == 0:
                save_path = os.path.join(SAVE_DIR, f"pysc2_ppo_ep{ep}.pth")
                torch.save({
                    'epoch': ep,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                }, save_path)
                print(f"💾 [系統] 已儲存進度: {os.path.basename(save_path)}")
                
                # 清理舊存檔，只保留最新的 5 份
                cleanup_old_models(SAVE_DIR, keep_last=5)

    except KeyboardInterrupt:
        print("\n--- [系統] 訓練已手動停止 ---")
    finally:
        env.close()

if __name__ == "__main__":
    app.run(main)