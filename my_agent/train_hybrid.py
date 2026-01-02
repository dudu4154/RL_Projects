import sys
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import datetime
import traceback
import math 
import time

from absl import app
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from torch.utils.tensorboard import SummaryWriter

# ==========================================
# ===   【使用者設定區：熱啟動與存檔】   ===
# ==========================================

# 1. 是否載入之前的模型繼續訓練？ (True = 是, False = 從頭開始)
LOAD_EXISTING_MODEL = False 

# 2. 如果上面是 True，請填寫 .pth 檔案的完整路徑
#    例如: "./Models/hybrid_model_ep50.pth"
MODEL_PATH_TO_LOAD = "./Models/hybrid_model_ep100.pth"

# 3. 載入模型後的探索率 (Epsilon)
#    如果您載入的是已經很聰明的模型，建議設低一點 (例如 0.1 或 0.3)
#    如果您希望它載入後還是大量亂試，可以設高一點 (例如 0.8)
LOADED_EPSILON = 0.3 

# 4. 存檔頻率 (分鐘)
SAVE_INTERVAL_MINUTES = 15
MAX_TO_KEEP = 5

# ==========================================

# === 1. 全域參數與常數設定 ===
LR = 0.0003
GAMMA = 0.99
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOTAL_EPISODES = 10000 
PROBE_TARGET = 15 

# --- Epsilon 設定 (從頭訓練時的預設值) ---
EPSILON_START = 0.9
EPSILON_END = 0.05      
EPSILON_DECAY = 0.999    

# --- ID 定義 ---
NEXUS_ID = 59 
PYLON_ID = 60
ASSIMILATOR_ID = 61      
GATEWAY_ID = 62          
CYBERNETICS_CORE_ID = 72 
STALKER_ID = 74          
PROBE_ID = 84
MINERAL_FIELD_ID = 341
GEYSER_ID = 342          

COMMAND_CENTER_ID = 18
SUPPLY_DEPOT_ID = 19
REFINERY_ID = 20
BARRACKS_ID = 21         
BARRACKS_TECHLAB_ID = 37 
SCV_ID = 45
MARAUDER_ID = 51         

# Actions (PySC2)
BUILD_PYLON_ACTION = actions.FUNCTIONS.Build_Pylon_screen.id
BUILD_ASSIMILATOR_ACTION = actions.FUNCTIONS.Build_Assimilator_screen.id
BUILD_GATEWAY_ACTION = actions.FUNCTIONS.Build_Gateway_screen.id 
BUILD_CYBERNETICS_CORE_ACTION = actions.FUNCTIONS.Build_CyberneticsCore_screen.id 
TRAIN_PROBE_ACTION = actions.FUNCTIONS.Train_Probe_quick.id
TRAIN_STALKER_ACTION = actions.FUNCTIONS.Train_Stalker_quick.id 

BUILD_SUPPLYDEPOT_ACTION = actions.FUNCTIONS.Build_SupplyDepot_screen.id
BUILD_REFINERY_ACTION = actions.FUNCTIONS.Build_Refinery_screen.id
BUILD_BARRACKS_ACTION = actions.FUNCTIONS.Build_Barracks_screen.id
BUILD_TECHLAB_ACTION = actions.FUNCTIONS.Build_TechLab_quick.id
TRAIN_SCV_ACTION = actions.FUNCTIONS.Train_SCV_quick.id
TRAIN_MARAUDER_ACTION = actions.FUNCTIONS.Train_Marauder_quick.id 
MOVE_CAMERA_ACTION = actions.FUNCTIONS.move_camera.id 

HARVEST_ACTION = actions.FUNCTIONS.Harvest_Gather_screen.id 
NO_OP = actions.FUNCTIONS.no_op()

# --- 地圖中心點 (Simple64) ---
MAP_CENTER_MINIMAP = (32, 32)
MAP_CENTER_SCREEN = (42, 42)

# === RL 動作輸出定義 (AI 部分) ===
ACTION_DO_NOTHING = 0
ACTION_BUILD_PROBE = 1
ACTION_BUILD_PYLON = 2
ACTION_BUILD_GATEWAY = 3
ACTION_BUILD_ASSIMILATOR = 4
ACTION_EXPLORE = 5           
ACTION_FOLLOW_CAM = 6        

# === 2. PPO 模型 ===
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 64)
        self.fc2 = nn.Linear(64, 128)
        self.actor = nn.Linear(128, num_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        if self.fc1.in_features != x.shape[1]:
            self.fc1 = nn.Linear(x.shape[1], 64).to(DEVICE)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.actor(x), dim=-1), self.critic(x)

# === 3. 主角 Agent (Protoss) ===
class ProtossHybridAgent:
    def __init__(self):
        self.model = ActorCritic(num_inputs=84*84, num_actions=7).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.memory_states = []
        self.memory_actions = []
        self.memory_logprobs = []
        self.memory_rewards = [] 
        self.memory_values = []
        self.epsilon = EPSILON_START
        self.reset_game_variables()

    # --- 【新增功能】載入舊模型 ---
    def load_model(self, path, new_epsilon=None):
        if os.path.exists(path):
            try:
                self.model.load_state_dict(torch.load(path, map_location=DEVICE))
                self.model.train() # 設定為訓練模式
                print(f"✅ 成功載入模型權重: {path}")
                
                if new_epsilon is not None:
                    self.epsilon = new_epsilon
                    print(f"   -> 探索率 (Epsilon) 已更新為: {self.epsilon}")
                return True
            except Exception as e:
                print(f"❌ 載入模型失敗: {e}")
                return False
        else:
            print(f"⚠️ 找不到模型檔案: {path}，將從頭開始訓練。")
            return False

    def reset_game(self):
        self.reset_game_variables()
        self.memory_states = []
        self.memory_actions = []
        self.memory_logprobs = []
        self.memory_rewards = []
        self.memory_values = []
        
        self.prev_stats = {
            "my_stalker_hp": 0, 
            "my_stalker_shield": 0,
            "enemy_total_hp": 0, 
            "enemy_marauder_count": 0
        }
        self.enemy_visible_last_step = False 

    def reset_game_variables(self):
        self.use_ai = False
        self.state = -1
        self.pylon_target_a = None 
        self.pylon_target_b = None
        self.assimilator_target = None   
        self.assimilator_target_2 = None 
        self.gas_probes_assigned = 0     
        self.selecting_probe = True      
        self.recent_selected_coords = [] 
        self.stalkers_trained = 0        
        self.select_attempts = 0 
        self.initial_mineral_coords = None
        self.nexus_x_screen = 0
        self.nexus_y_screen = 0
        self.is_first_step = True

    def update_epsilon(self):
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY
            if self.epsilon < EPSILON_END:
                self.epsilon = EPSILON_END

    def get_health_stats(self, obs):
        stats = { 
            "my_stalker_hp": 0, 
            "my_stalker_shield": 0,
            "enemy_total_hp": 0,
            "enemy_marauder_count": 0
        }
        
        if hasattr(obs.observation, 'raw_units'):
            for unit in obs.observation.raw_units:
                if unit.alliance == 1: 
                    if unit.unit_type == STALKER_ID:
                        stats["my_stalker_hp"] += unit.health
                        stats["my_stalker_shield"] += unit.shield
                elif unit.alliance == 4: 
                    if unit.unit_type == MARAUDER_ID: 
                        stats["enemy_total_hp"] += unit.health
                        stats["enemy_marauder_count"] += 1
        return stats

    def calculate_custom_reward(self, obs):
        current_stats = self.get_health_stats(obs)
        reward = 0
        
        if "my_stalker_shield" not in self.prev_stats:
            self.prev_stats["my_stalker_shield"] = 0
        
        # 1. 時間懲罰
        reward -= 0.0001 

        # 2. 追獵者損失護盾
        if current_stats["my_stalker_shield"] < self.prev_stats["my_stalker_shield"]:
            shield_loss = self.prev_stats["my_stalker_shield"] - current_stats["my_stalker_shield"]
            reward -= shield_loss * 0.001

        # 3. 追獵者損失生命
        if current_stats["my_stalker_hp"] < self.prev_stats["my_stalker_hp"]:
            hp_loss = self.prev_stats["my_stalker_hp"] - current_stats["my_stalker_hp"]
            reward -= hp_loss * 0.005

        # 4. 造成敵人傷害
        if current_stats["enemy_total_hp"] < self.prev_stats["enemy_total_hp"]:
            damage_dealt = self.prev_stats["enemy_total_hp"] - current_stats["enemy_total_hp"]
            if damage_dealt > 0:
                reward += damage_dealt * 0.001

        # 5. 擊殺掠奪者
        if current_stats["enemy_marauder_count"] < self.prev_stats["enemy_marauder_count"]:
            kills = self.prev_stats["enemy_marauder_count"] - current_stats["enemy_marauder_count"]
            reward += (kills * 0.1)

        # 6. 全滅掠奪者
        if self.prev_stats["enemy_marauder_count"] > 0 and current_stats["enemy_marauder_count"] == 0:
            reward += 1.0

        self.prev_stats = current_stats
        return reward
    
    def preprocess(self, obs):
        screen = obs.observation.feature_screen.unit_type
        return torch.from_numpy(screen).float().unsqueeze(0).to(DEVICE)

    def step(self, obs):
        if self.use_ai:
            return self.step_ai(obs)
        else:
            return self.step_script(obs)

    # === AI 邏輯 ===
    def step_ai(self, obs):
        state_tensor = self.preprocess(obs)
        action_probs, state_value = self.model(state_tensor)
        
        if random.random() < self.epsilon:
            action_index = torch.tensor(random.randint(0, 6)).to(DEVICE) 
            dist = torch.distributions.Categorical(action_probs)
            log_prob = dist.log_prob(action_index)
        else:
            dist = torch.distributions.Categorical(action_probs)
            action_index = dist.sample()
            log_prob = dist.log_prob(action_index)
        
        self.memory_states.append(state_tensor)
        self.memory_actions.append(action_index)
        self.memory_logprobs.append(log_prob)
        self.memory_values.append(state_value)
        
        return self.execute_ai_action(action_index.item(), obs)

    def execute_ai_action(self, action_cmd, obs):
        minerals = obs.observation.player.minerals
        unit_type = obs.observation.feature_screen.unit_type

        def get_random_pos(unit_id):
            y, x = (unit_type == unit_id).nonzero()
            if x.any():
                idx = random.randint(0, len(x)-1)
                return (x[idx], y[idx])
            return None

        if action_cmd == ACTION_BUILD_PROBE:
            if minerals >= 50 and TRAIN_PROBE_ACTION in obs.observation.available_actions:
                return actions.FUNCTIONS.Train_Probe_quick("now")
            return NO_OP
            
        elif action_cmd == ACTION_BUILD_PYLON:
            if minerals >= 100 and BUILD_PYLON_ACTION in obs.observation.available_actions:
                pos = get_random_pos(PROBE_ID)
                if pos:
                    tx = int(np.clip(pos[0] + random.randint(-5, 5), 0, 83))
                    ty = int(np.clip(pos[1] + random.randint(-5, 5), 0, 83))
                    return actions.FUNCTIONS.Build_Pylon_screen("now", (tx, ty))
            return NO_OP

        elif action_cmd == ACTION_BUILD_GATEWAY:
            if minerals >= 150 and BUILD_GATEWAY_ACTION in obs.observation.available_actions:
                pos = get_random_pos(PROBE_ID)
                if pos:
                    tx = int(np.clip(pos[0] + random.randint(-5, 5), 0, 83))
                    ty = int(np.clip(pos[1] + random.randint(-5, 5), 0, 83))
                    return actions.FUNCTIONS.Build_Gateway_screen("now", (tx, ty))
            return NO_OP

        elif action_cmd == ACTION_BUILD_ASSIMILATOR:
            return NO_OP

        elif action_cmd == ACTION_EXPLORE:
            if actions.FUNCTIONS.select_army.id in obs.observation.available_actions:
                 return actions.FUNCTIONS.select_army("select")
            
            if actions.FUNCTIONS.Attack_minimap.id in obs.observation.available_actions:
                 rand_x = random.randint(5, 59)
                 rand_y = random.randint(5, 59)
                 return actions.FUNCTIONS.Attack_minimap("now", (rand_x, rand_y))
        
        elif action_cmd == ACTION_FOLLOW_CAM:
            if MOVE_CAMERA_ACTION in obs.observation.available_actions:
                y_stalkers, x_stalkers = (unit_type == STALKER_ID).nonzero()
                if x_stalkers.any():
                    mean_x = int(x_stalkers.mean())
                    mean_y = int(y_stalkers.mean())
                    return actions.FUNCTIONS.move_camera((mean_x, mean_y))
                else:
                    return actions.FUNCTIONS.move_camera(MAP_CENTER_MINIMAP)

        return NO_OP

    # === 腳本邏輯 ===
    def step_script(self, obs):
        unit_type_screen = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
        current_supply_cap = obs.observation.player.food_cap
        current_minerals = obs.observation.player.minerals
        current_vespene = obs.observation.player.vespene
        current_workers = obs.observation.player.food_workers
        
        def count_units(unit_id):
            return np.sum((unit_type_screen == unit_id).astype(int))

        def get_select_nexus_action():
            y_coords, x_coords = (unit_type_screen == NEXUS_ID).nonzero()
            if x_coords.any():
                target_x = int(x_coords.mean())
                target_y = int(y_coords.mean())
                self.nexus_x_screen = target_x
                self.nexus_y_screen = target_y
                x_start = max(0, target_x - 5)
                y_start = max(0, target_y - 5)
                x_end = min(83, target_x + 5)
                y_end = min(83, target_y + 5)
                return actions.FUNCTIONS.select_rect("select", (x_start, y_start), (x_end, y_end))
            return NO_OP

        def get_select_probe_action(select_type="select"):
            y_coords, x_coords = (unit_type_screen == PROBE_ID).nonzero() 
            if x_coords.any():
                target_x, target_y = x_coords[0], y_coords[0] 
                return actions.FUNCTIONS.select_point(select_type, (target_x, target_y))
            return NO_OP

        def get_select_gateway_action():
            y_coords, x_coords = (unit_type_screen == GATEWAY_ID).nonzero()
            if x_coords.any():
                target_x = int(x_coords.mean())
                target_y = int(y_coords.mean())
                return actions.FUNCTIONS.select_point("select", (target_x, target_y))
            return NO_OP
        
        if self.is_first_step:
            self.is_first_step = False
            mineral_coords = (unit_type_screen == MINERAL_FIELD_ID).nonzero()
            if mineral_coords[0].any():
                self.initial_mineral_coords = (mineral_coords[1][0], mineral_coords[0][0])
            nexus_coords = (unit_type_screen == NEXUS_ID).nonzero()
            if nexus_coords[0].any():
                self.nexus_x_screen = nexus_coords[1].mean().astype(int)
                self.nexus_y_screen = nexus_coords[0].mean().astype(int)

        # --- 狀態機 (State Machine) ---
        if self.state == -1:
            if HARVEST_ACTION in obs.observation.available_actions and self.initial_mineral_coords:
                self.state = 0
                return actions.FUNCTIONS.Harvest_Gather_screen("now", self.initial_mineral_coords)
            else:
                return get_select_probe_action("select_all_type")

        elif self.state == 0:
            if current_workers < PROBE_TARGET:
                if TRAIN_PROBE_ACTION in obs.observation.available_actions:
                    if current_minerals >= 50:
                        return actions.FUNCTIONS.Train_Probe_quick("now")
                elif current_minerals >= 50:
                    return get_select_nexus_action()
            elif current_workers >= PROBE_TARGET and current_minerals >= 100:
                if self.nexus_x_screen != 0: 
                    min_y_all, min_x_all = (unit_type_screen == MINERAL_FIELD_ID).nonzero()
                    if min_x_all.any():
                        mineral_mean_x = min_x_all.mean()
                        mineral_mean_y = min_y_all.mean()
                    else:
                        mineral_mean_x, mineral_mean_y = self.initial_mineral_coords
                    
                    diff_x = self.nexus_x_screen - mineral_mean_x
                    diff_y = self.nexus_y_screen - mineral_mean_y
                    offset_x = 20 if diff_x > 0 else -20
                    offset_y = 20 if diff_y > 0 else -20
                    self.pylon_target_a = (
                        np.clip(self.nexus_x_screen + offset_x, 0, 83).astype(int),
                        np.clip(self.nexus_y_screen + offset_y, 0, 83).astype(int)
                    )
                    self.pylon_target_b = (
                        np.clip(self.nexus_x_screen + offset_x * 0.5, 0, 83).astype(int),
                        np.clip(self.nexus_y_screen + offset_y * 1.5, 0, 83).astype(int)
                    )
                    self.state = 1
                else:
                    return get_select_nexus_action()

        elif self.state == 1:
            if BUILD_PYLON_ACTION in obs.observation.available_actions:
                self.state = 1.1
                self.select_attempts = 0
                return actions.FUNCTIONS.Build_Pylon_screen("now", self.pylon_target_a)
            else:
                return get_select_probe_action()

        elif self.state == 1.1:
            if current_supply_cap >= 21:
                self.state = 1.2
                self.select_attempts = 0
                return NO_OP
            self.select_attempts += 1
            if self.select_attempts > 20: self.state = 1; return NO_OP

        elif self.state == 1.2:
            if BUILD_PYLON_ACTION in obs.observation.available_actions:
                if current_minerals >= 100:
                    self.state = 1.3
                    self.select_attempts = 0
                    return actions.FUNCTIONS.Build_Pylon_screen("now", self.pylon_target_b)
            else:
                return get_select_probe_action()

        elif self.state == 1.3:
            if current_supply_cap >= 31:
                self.state = 2
                self.select_attempts = 0
                return NO_OP
            self.select_attempts += 1
            if self.select_attempts > 30:
                self.state = 1.2
                new_offset_x = -20 if (self.nexus_x_screen - self.pylon_target_b[0]) > 0 else 20
                self.pylon_target_b = (np.clip(self.nexus_x_screen + new_offset_x, 0, 83).astype(int), self.pylon_target_b[1])
                return NO_OP

        elif self.state == 2:
            if BUILD_ASSIMILATOR_ACTION in obs.observation.available_actions:
                if current_minerals >= 75:
                    if self.assimilator_target is None:
                        y_geo, x_geo = (unit_type_screen == GEYSER_ID).nonzero()
                        if x_geo.any():
                             first_x, first_y = x_geo[0], y_geo[0]
                             mask = (np.abs(x_geo - first_x) < 10) & (np.abs(y_geo - first_y) < 10)
                             target_x = int(x_geo[mask].mean())
                             target_y = int(y_geo[mask].mean())
                             self.assimilator_target = (target_x, target_y)
                        else:
                             return NO_OP 
                    self.state = 2.1
                    self.select_attempts = 0
                    return actions.FUNCTIONS.Build_Assimilator_screen("now", self.assimilator_target)
            else:
                return get_select_probe_action()
        
        elif self.state == 2.1:
            if count_units(ASSIMILATOR_ID) >= 1:
                self.state = 3
                self.gas_probes_assigned = 0
                self.selecting_probe = True
                self.recent_selected_coords = []
                return NO_OP
            self.select_attempts += 1
            if self.select_attempts > 20: self.state = 2; self.assimilator_target = None; return NO_OP

        elif self.state == 3:
            if self.gas_probes_assigned < 3:
                if self.selecting_probe:
                    y_coords, x_coords = (unit_type_screen == PROBE_ID).nonzero() 
                    if x_coords.any():
                        distances = np.sqrt((x_coords - self.assimilator_target[0])**2 + (y_coords - self.assimilator_target[1])**2)
                        mask = distances > 15
                        for past_x, past_y in self.recent_selected_coords:
                            dist_from_past = np.sqrt((x_coords - past_x)**2 + (y_coords - past_y)**2)
                            mask = mask & (dist_from_past > 8)
                        
                        if mask.any():
                            valid_x = x_coords[mask]
                            valid_y = y_coords[mask]
                            idx = random.randint(0, len(valid_x) - 1)
                            target_x, target_y = valid_x[idx], valid_y[idx]
                        else:
                            idx = random.randint(0, len(x_coords) - 1)
                            target_x, target_y = x_coords[idx], y_coords[idx]
                        
                        self.recent_selected_coords.append((target_x, target_y))
                        self.selecting_probe = False 
                        return actions.FUNCTIONS.select_point("select", (target_x, target_y))
                    return NO_OP
                else:
                    if HARVEST_ACTION in obs.observation.available_actions:
                        self.gas_probes_assigned += 1
                        self.selecting_probe = True 
                        return actions.FUNCTIONS.Harvest_Gather_screen("now", self.assimilator_target)
                    else:
                        self.selecting_probe = True
                        return NO_OP
            else:
                self.state = 4
                self.select_attempts = 0
                self.recent_selected_coords = []
                return NO_OP

        elif self.state == 4:
            if BUILD_ASSIMILATOR_ACTION in obs.observation.available_actions:
                if current_minerals >= 75:
                    if self.assimilator_target_2 is None:
                        y_geo, x_geo = (unit_type_screen == GEYSER_ID).nonzero()
                        if x_geo.any():
                            first_tx, first_ty = self.assimilator_target
                            distances = np.sqrt((x_geo - first_tx)**2 + (y_geo - first_ty)**2)
                            mask = distances > 15
                            if mask.any():
                                target_x = int(x_geo[mask].mean())
                                target_y = int(y_geo[mask].mean())
                                self.assimilator_target_2 = (target_x, target_y)
                            else:
                                return NO_OP
                        else:
                            return NO_OP
                    self.state = 4.1
                    self.select_attempts = 0
                    return actions.FUNCTIONS.Build_Assimilator_screen("now", self.assimilator_target_2)
            else:
                return get_select_probe_action()

        elif self.state == 4.1:
            if count_units(ASSIMILATOR_ID) >= 2:
                self.state = 5
                self.gas_probes_assigned = 0
                self.selecting_probe = True
                self.recent_selected_coords = []
                return NO_OP
            self.select_attempts += 1
            if self.select_attempts > 20: self.state = 4; self.assimilator_target_2 = None; return NO_OP

        elif self.state == 5:
            if self.gas_probes_assigned < 3:
                if self.selecting_probe:
                    y_coords, x_coords = (unit_type_screen == PROBE_ID).nonzero() 
                    if x_coords.any():
                        dist1 = np.sqrt((x_coords - self.assimilator_target[0])**2 + (y_coords - self.assimilator_target[1])**2)
                        dist2 = np.sqrt((x_coords - self.assimilator_target_2[0])**2 + (y_coords - self.assimilator_target_2[1])**2)
                        mask = (dist1 > 15) & (dist2 > 15)
                        for past_x, past_y in self.recent_selected_coords:
                            dist_from_past = np.sqrt((x_coords - past_x)**2 + (y_coords - past_y)**2)
                            mask = mask & (dist_from_past > 8)
                        
                        if mask.any():
                            valid_x = x_coords[mask]
                            valid_y = y_coords[mask]
                            idx = random.randint(0, len(valid_x) - 1)
                            target_x, target_y = valid_x[idx], valid_y[idx]
                        else:
                            idx = random.randint(0, len(x_coords) - 1)
                            target_x, target_y = x_coords[idx], y_coords[idx]

                        self.recent_selected_coords.append((target_x, target_y))
                        self.selecting_probe = False 
                        return actions.FUNCTIONS.select_point("select", (target_x, target_y))
                    return NO_OP
                else:
                    if HARVEST_ACTION in obs.observation.available_actions:
                        self.gas_probes_assigned += 1
                        self.selecting_probe = True 
                        return actions.FUNCTIONS.Harvest_Gather_screen("now", self.assimilator_target_2)
                    else:
                        self.selecting_probe = True
                        return NO_OP
            else:
                self.state = 6 
                return NO_OP

        elif self.state == 6:
            if BUILD_GATEWAY_ACTION in obs.observation.available_actions:
                if current_minerals >= 150:
                    if self.pylon_target_a:
                        px, py = self.pylon_target_a
                        offset_gateway_y = 12 if py < 42 else -12
                        target_x = np.clip(px, 0, 83) 
                        target_y = np.clip(py + offset_gateway_y, 0, 83)
                        target = (target_x, target_y)
                        self.state = 6.05
                        self.select_attempts = 0
                        return actions.FUNCTIONS.Build_Gateway_screen("now", target)
            else:
                 return get_select_probe_action()
        
        elif self.state == 6.05:
            if HARVEST_ACTION in obs.observation.available_actions and self.assimilator_target:
                self.state = 6.1
                return actions.FUNCTIONS.Harvest_Gather_screen("queued", self.assimilator_target)
            self.state = 6.1
            return NO_OP

        elif self.state == 6.1:
             if count_units(GATEWAY_ID) > 0:
                 self.state = 7
                 return NO_OP
             self.select_attempts += 1
             if self.select_attempts > 20: self.state = 6; return NO_OP

        elif self.state == 7:
            if BUILD_CYBERNETICS_CORE_ACTION in obs.observation.available_actions:
                if current_minerals >= 150:
                    if self.pylon_target_a:
                        px, py = self.pylon_target_a
                        offset_cyber_x = 12 if px < 42 else -12
                        target_x = np.clip(px + offset_cyber_x, 0, 83)
                        target_y = np.clip(py, 0, 83) 
                        target = (target_x, target_y)
                        self.state = 7.05
                        self.select_attempts = 0
                        return actions.FUNCTIONS.Build_CyberneticsCore_screen("now", target)
            else:
                return get_select_probe_action()

        elif self.state == 7.05:
            if HARVEST_ACTION in obs.observation.available_actions and self.assimilator_target:
                self.state = 7.1
                return actions.FUNCTIONS.Harvest_Gather_screen("queued", self.assimilator_target)
            self.state = 7.1
            return NO_OP

        elif self.state == 7.1:
            if count_units(CYBERNETICS_CORE_ID) > 0:
                 self.state = 8
                 self.stalkers_trained = 0
                 return NO_OP
            self.select_attempts += 1
            if self.select_attempts > 80: self.state = 7; return NO_OP

        elif self.state == 8:
            if self.stalkers_trained < 5:
                if TRAIN_STALKER_ACTION in obs.observation.available_actions:
                    if current_minerals >= 125 and current_vespene >= 50:
                        if current_supply_cap - current_workers - self.stalkers_trained * 2 >= 2: 
                             self.stalkers_trained += 1
                             return actions.FUNCTIONS.Train_Stalker_quick("now")
                    return NO_OP
                else:
                    return get_select_gateway_action()
            else:
                self.state = 9
                print(f"[Protoss] Stalkers Ready. Preparing to Move Out.")
                return NO_OP

        elif self.state == 9:
            # Step 1: F2 全選
            if actions.FUNCTIONS.select_army.id in obs.observation.available_actions:
                self.state = 10
                return actions.FUNCTIONS.select_army("select")
            return NO_OP

        elif self.state == 10:
            # Step 2: 移動鏡頭到中間
            if MOVE_CAMERA_ACTION in obs.observation.available_actions:
                self.state = 11
                return actions.FUNCTIONS.move_camera(MAP_CENTER_MINIMAP)
            return NO_OP

        elif self.state == 11:
            # Step 3: A-Move 到中間
            if actions.FUNCTIONS.Attack_screen.id in obs.observation.available_actions:
                self.state = 12
                return actions.FUNCTIONS.Attack_screen("now", MAP_CENTER_SCREEN)
            elif actions.FUNCTIONS.Attack_minimap.id in obs.observation.available_actions:
                self.state = 12
                return actions.FUNCTIONS.Attack_minimap("now", MAP_CENTER_MINIMAP)
            return NO_OP

        elif self.state == 12:
            # Step 4: 切換 AI
            self.use_ai = True
            print("====================================")
            print("   Protoss in Position -> AI Taking Over")
            print("====================================")
            return NO_OP

        return NO_OP

# === 4. 對手 (Terran Bot) ===
class TerranBot:
    def __init__(self):
        self.state = -1
        self.supply_depot_target = None
        self.refinery_target = None
        self.barracks_target = None
        self.gas_workers_assigned = 0
        self.marauders_trained = 0     
        self.selecting_worker = True
        self.recent_selected_coords = []
        self.busy_depot_locations = [] 
        self.depots_built = 0          
        self.last_depot_pixels = 0     
        self.select_attempts = 0
        self.initial_mineral_coords = None
        self.first_depot_coords = None 
        self.build_dir = (1, 1)        
        self.base_minimap_coords = None 
        self.cc_x_screen = 0
        self.cc_y_screen = 0
        self.camera_centered = False   
        self.is_first_step = True
        self.attack_coordinates = MAP_CENTER_MINIMAP

    def step(self, obs):
        unit_type_screen = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
        current_minerals = obs.observation.player.minerals
        current_vespene = obs.observation.player.vespene
        current_workers = obs.observation.player.food_workers
        current_supply_cap = obs.observation.player.food_cap

        def count_units(unit_id):
            return np.sum((unit_type_screen == unit_id).astype(int))

        def get_select_cc_action():
            y_coords, x_coords = (unit_type_screen == COMMAND_CENTER_ID).nonzero()
            if x_coords.any():
                target_x = int(x_coords.mean())
                target_y = int(y_coords.mean())
                self.cc_x_screen = target_x
                self.cc_y_screen = target_y
                return actions.FUNCTIONS.select_point("select", (target_x, target_y))
            return NO_OP

        def get_select_scv_action(select_type="select"):
            y_coords, x_coords = (unit_type_screen == SCV_ID).nonzero()
            if x_coords.any():
                mask = np.ones(len(x_coords), dtype=bool) 
                for busy_x, busy_y in self.busy_depot_locations:
                    dist_busy = np.sqrt((x_coords - busy_x)**2 + (y_coords - busy_y)**2)
                    mask = mask & (dist_busy > 10)
                if self.supply_depot_target:
                    dist_target = np.sqrt((x_coords - self.supply_depot_target[0])**2 + (y_coords - self.supply_depot_target[1])**2)
                    mask = mask & (dist_target > 10)
                if self.refinery_target:
                    dist_gas = np.sqrt((x_coords - self.refinery_target[0])**2 + (y_coords - self.refinery_target[1])**2)
                    mask = mask & (dist_gas > 15) 

                if mask.any():
                    valid_indices = np.where(mask)[0]
                    idx = random.choice(valid_indices)
                    target_x, target_y = int(x_coords[idx]), int(y_coords[idx])
                    return actions.FUNCTIONS.select_point(select_type, (target_x, target_y))
                
                idx = random.randint(0, len(x_coords) - 1)
                target_x, target_y = int(x_coords[idx]), int(y_coords[idx]) 
                return actions.FUNCTIONS.select_point(select_type, (target_x, target_y))
            return NO_OP
        
        def get_select_barracks_action():
            y_coords, x_coords = (unit_type_screen == BARRACKS_ID).nonzero()
            if x_coords.any():
                target_x = int(x_coords.mean())
                target_y = int(y_coords.mean())
                return actions.FUNCTIONS.select_point("select", (target_x, target_y))
            return NO_OP

        if self.is_first_step:
            self.is_first_step = False
            mineral_coords = (unit_type_screen == MINERAL_FIELD_ID).nonzero()
            if mineral_coords[0].any():
                self.initial_mineral_coords = (mineral_coords[1][0], mineral_coords[0][0])
            cc_coords = (unit_type_screen == COMMAND_CENTER_ID).nonzero()
            if cc_coords[0].any():
                self.cc_x_screen = int(cc_coords[1].mean())
                self.cc_y_screen = int(cc_coords[0].mean())
            player_relative = obs.observation.feature_minimap[features.MINIMAP_FEATURES.player_relative.index]
            y_mini, x_mini = (player_relative == features.PlayerRelative.SELF).nonzero()
            if x_mini.any():
                self.base_minimap_coords = (int(x_mini.mean()), int(y_mini.mean()))
            self.last_depot_pixels = count_units(SUPPLY_DEPOT_ID)

        if self.state == -1:
            if HARVEST_ACTION in obs.observation.available_actions and self.initial_mineral_coords:
                self.state = 0
                return actions.FUNCTIONS.Harvest_Gather_screen("now", self.initial_mineral_coords)
            else:
                return get_select_scv_action("select_all_type")

        elif self.state == 0:
            if current_workers < 15:
                if TRAIN_SCV_ACTION in obs.observation.available_actions:
                    if current_minerals >= 50:
                        return actions.FUNCTIONS.Train_SCV_quick("now")
                elif current_minerals >= 50:
                    return get_select_cc_action()
            elif current_minerals >= 100:
                self.state = 1
                offset_x = 15 if self.initial_mineral_coords[0] < self.cc_x_screen else -15
                offset_y = 15 if self.initial_mineral_coords[1] < self.cc_y_screen else -15
                self.supply_depot_target = (np.clip(self.cc_x_screen + offset_x, 0, 83).astype(int), np.clip(self.cc_y_screen + offset_y, 0, 83).astype(int))
                self.first_depot_coords = self.supply_depot_target
                dir_x = 1 if offset_x > 0 else -1
                dir_y = 1 if offset_y > 0 else -1
                self.build_dir = (dir_x, dir_y)
                return get_select_scv_action()
            return NO_OP

        elif self.state == 1:
            if BUILD_SUPPLYDEPOT_ACTION in obs.observation.available_actions:
                if current_minerals >= 100:
                    self.state = 1.1
                    self.select_attempts = 0 
                    return actions.FUNCTIONS.Build_SupplyDepot_screen("now", self.supply_depot_target)
            else:
                if current_minerals >= 100: return get_select_scv_action()
                return NO_OP

        elif self.state == 1.1:
            current_pixels = count_units(SUPPLY_DEPOT_ID)
            if current_pixels > self.last_depot_pixels + 5: 
                self.depots_built += 1
                self.last_depot_pixels = current_pixels
                self.busy_depot_locations.append(self.supply_depot_target)
                if self.depots_built < 3:
                    dir_x, dir_y = self.build_dir
                    base_x, base_y = self.first_depot_coords
                    if self.depots_built == 1:
                        next_x, next_y = base_x + (12 * dir_x), base_y
                    else:
                        next_x, next_y = base_x + (6 * dir_x), base_y + (12 * dir_y)
                    self.supply_depot_target = (np.clip(next_x, 0, 83).astype(int), np.clip(next_y, 0, 83).astype(int))
                    self.state = 1
                    return get_select_scv_action() 
                else:
                    self.state = 2
                    return NO_OP
            self.select_attempts += 1
            if self.select_attempts > 100:
                old_x, old_y = self.supply_depot_target
                self.supply_depot_target = (np.clip(old_x + random.randint(-15, 15), 0, 83), np.clip(old_y + random.randint(-15, 15), 0, 83))
                self.state = 1 
                return get_select_scv_action()
            return NO_OP

        elif self.state == 2:
            if BUILD_REFINERY_ACTION in obs.observation.available_actions:
                if current_minerals >= 75:
                    if self.refinery_target is None:
                        y_geo, x_geo = (unit_type_screen == GEYSER_ID).nonzero()
                        if x_geo.any():
                             first_x, first_y = x_geo[0], y_geo[0]
                             mask = (np.abs(x_geo - first_x) < 10) & (np.abs(y_geo - first_y) < 10)
                             target_x = int(x_geo[mask].mean())
                             target_y = int(y_geo[mask].mean())
                             self.refinery_target = (target_x, target_y)
                        else:
                             return NO_OP
                    self.state = 2.1
                    return actions.FUNCTIONS.Build_Refinery_screen("now", self.refinery_target)
            else:
                return get_select_scv_action()

        elif self.state == 2.1:
            if count_units(REFINERY_ID) > 0:
                self.state = 3
                self.gas_workers_assigned = 0
                self.selecting_worker = True
                self.recent_selected_coords = []
                return NO_OP
            return NO_OP

        elif self.state == 3:
            if self.gas_workers_assigned < 3:
                if self.selecting_worker:
                    y_coords, x_coords = (unit_type_screen == SCV_ID).nonzero()
                    if x_coords.any():
                        distances = np.sqrt((x_coords - self.refinery_target[0])**2 + (y_coords - self.refinery_target[1])**2)
                        mask = distances > 15 
                        for past_x, past_y in self.recent_selected_coords:
                            dist_from_past = np.sqrt((x_coords - past_x)**2 + (y_coords - past_y)**2)
                            mask = mask & (dist_from_past > 5)
                        if mask.any():
                            valid_x = x_coords[mask]
                            valid_y = y_coords[mask]
                            idx = random.randint(0, len(valid_x) - 1)
                            target_x, target_y = int(valid_x[idx]), int(valid_y[idx])
                            self.recent_selected_coords.append((target_x, target_y))
                            self.selecting_worker = False
                            return actions.FUNCTIONS.select_point("select", (target_x, target_y))
                        idx = random.randint(0, len(x_coords) - 1)
                        target_x, target_y = int(x_coords[idx]), int(y_coords[idx])
                        self.recent_selected_coords.append((target_x, target_y))
                        self.selecting_worker = False
                        return actions.FUNCTIONS.select_point("select", (target_x, target_y))
                    return NO_OP
                else:
                    if HARVEST_ACTION in obs.observation.available_actions:
                        self.gas_workers_assigned += 1
                        self.selecting_worker = True
                        return actions.FUNCTIONS.Harvest_Gather_screen("now", self.refinery_target)
                    else:
                        self.selecting_worker = True
                        return NO_OP
            else:
                self.state = 4
                return NO_OP
        
        elif self.state == 4:
            if self.base_minimap_coords is None:
                player_relative = obs.observation.feature_minimap[features.MINIMAP_FEATURES.player_relative.index]
                y_mini, x_mini = (player_relative == features.PlayerRelative.SELF).nonzero()
                if x_mini.any():
                    self.base_minimap_coords = (int(x_mini.mean()), int(y_mini.mean()))
                else:
                    self.base_minimap_coords = (15, 15)

            if not self.camera_centered:
                target_x, target_y = self.base_minimap_coords
                self.camera_centered = True
                return actions.FUNCTIONS.move_camera((target_x, target_y))
            
            if BUILD_BARRACKS_ACTION in obs.observation.available_actions:
                if current_minerals >= 150:
                    if self.barracks_target is None:
                        minimap_x = self.base_minimap_coords[0]
                        offset_x = 30 
                        offset_y = 0
                        if minimap_x > 32:
                            offset_x = -30
                        rand_off_x = random.randint(-5, 5)
                        rand_off_y = random.randint(-5, 5)
                        self.barracks_target = (np.clip(42 + offset_x + rand_off_x, 0, 83).astype(int), np.clip(42 + offset_y + rand_off_y, 0, 83).astype(int))

                    self.state = 4.1
                    self.select_attempts = 0
                    return actions.FUNCTIONS.Build_Barracks_screen("now", self.barracks_target)
            else:
                return get_select_scv_action()

        elif self.state == 4.1:
             if count_units(BARRACKS_ID) > 0:
                 self.state = 5
                 return NO_OP
             self.select_attempts += 1
             if self.select_attempts > 100: self.state = 4; self.barracks_target = None; return NO_OP
             return NO_OP

        elif self.state == 5:
            if count_units(BARRACKS_TECHLAB_ID) > 0:
                self.state = 7 
                return NO_OP
            if BUILD_TECHLAB_ACTION in obs.observation.available_actions:
                if current_minerals >= 50 and current_vespene >= 25:
                    self.state = 6 
                    return actions.FUNCTIONS.Build_TechLab_quick("now")
            else:
                return get_select_barracks_action()
            return NO_OP

        elif self.state == 6:
            if count_units(BARRACKS_TECHLAB_ID) > 0:
                self.state = 7
                return NO_OP
            return NO_OP

        elif self.state == 7:
            if self.marauders_trained < 5:
                if TRAIN_MARAUDER_ACTION in obs.observation.available_actions:
                    if current_minerals >= 100 and current_vespene >= 25:
                        if current_supply_cap - current_workers - self.marauders_trained * 2 >= 2:
                            self.marauders_trained += 1
                            return actions.FUNCTIONS.Train_Marauder_quick("now")
                    return NO_OP
                else:
                    return get_select_barracks_action()
            else:
                self.state = 8
                print(f"[Terran] Marauders Ready. Moving to Center.")
                return NO_OP

        elif self.state == 8:
            if actions.FUNCTIONS.select_army.id in obs.observation.available_actions:
                self.state = 9
                return actions.FUNCTIONS.select_army("select")
            return NO_OP

        elif self.state == 9:
            if MOVE_CAMERA_ACTION in obs.observation.available_actions:
                self.state = 10
                return actions.FUNCTIONS.move_camera(MAP_CENTER_MINIMAP)
            return NO_OP
        
        elif self.state == 10:
            if actions.FUNCTIONS.Attack_minimap.id in obs.observation.available_actions:
                self.state = 11 
                return actions.FUNCTIONS.Attack_minimap("now", MAP_CENTER_MINIMAP)
            return NO_OP

        elif self.state == 11:
            if random.random() < 0.02: 
                if actions.FUNCTIONS.Attack_minimap.id in obs.observation.available_actions:
                    return actions.FUNCTIONS.Attack_minimap("now", MAP_CENTER_MINIMAP)
            return NO_OP

        return NO_OP

# === 5. PPO 更新與主程式 ===
def update_ppo(agent):
    if len(agent.memory_rewards) == 0: return 0
    rewards = []
    discounted_reward = 0
    for reward in reversed(agent.memory_rewards):
        discounted_reward = reward + GAMMA * discounted_reward
        rewards.insert(0, discounted_reward)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
    if len(rewards) > 1:
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
    
    min_len = min(len(agent.memory_states), len(rewards))
    if min_len == 0: return 0
    
    old_states = torch.cat(agent.memory_states[:min_len]).detach()
    old_actions = torch.stack(agent.memory_actions[:min_len]).detach()
    old_logprobs = torch.stack(agent.memory_logprobs[:min_len]).detach()
    old_values = torch.cat(agent.memory_values[:min_len]).detach()
    rewards = rewards[:min_len]
    
    advantages = rewards - old_values.squeeze()
    loss_actor = -(old_logprobs * advantages.detach()).mean()
    loss_critic = F.mse_loss(old_values.squeeze(), rewards)
    total_loss = loss_actor + 0.5 * loss_critic
    
    agent.optimizer.zero_grad()
    total_loss.backward()
    agent.optimizer.step()
    return total_loss.item()

def save_model(agent, model_dir, episode):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    filename = os.path.join(model_dir, f"hybrid_model_ep{episode}.pth")
    torch.save(agent.model.state_dict(), filename)
    print(f"Model saved: {filename}")
    
    files = sorted(glob.glob(os.path.join(model_dir, "hybrid_model_ep*.pth")), key=os.path.getmtime)
    if len(files) > MAX_TO_KEEP:
        for f in files[:-MAX_TO_KEEP]:
            try:
                os.remove(f)
            except OSError as e:
                print(f"Error removing old model file: {e}")

def main(argv):
    print("=== SC2 Hybrid Agent (Center Combat) ===")
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M_Hybrid")
    
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", current_time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created model directory: {model_dir}")

    writer = SummaryWriter(log_dir)

    bot_hybrid = ProtossHybridAgent()
    bot_opponent = TerranBot() 

    # --- 【新增功能】檢查是否需要載入舊模型 ---
    if LOAD_EXISTING_MODEL:
        if bot_hybrid.load_model(MODEL_PATH_TO_LOAD, new_epsilon=LOADED_EPSILON):
            print(">>> 舊模型載入成功，準備繼續訓練 <<<")
        else:
            print(">>> 舊模型載入失敗，將使用新模型開始 <<<")

    last_save_time = time.time()
    
    try:
        with sc2_env.SC2Env(
            map_name="Simple64",
            players=[
                sc2_env.Agent(sc2_env.Race.protoss), 
                sc2_env.Agent(sc2_env.Race.terran)
            ],
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64),
                use_raw_units=True, 
            ),
            step_mul=8,
            game_steps_per_episode=8000,
            visualize=True
        ) as env:
            for episode in range(TOTAL_EPISODES):
                obs_list = env.reset()
                bot_hybrid.reset_game()
                bot_opponent = TerranBot()

                total_reward = 0
                step_count = 0
                
                while True:
                    step_count += 1
                    
                    if time.time() - last_save_time > SAVE_INTERVAL_MINUTES * 60:
                        save_model(bot_hybrid, model_dir, episode + 1)
                        last_save_time = time.time()
                        print(f"[Auto-Save] Saved model at Episode {episode+1}, Step {step_count}")
                    
                    try:
                        action_protoss = bot_hybrid.step(obs_list[0])
                        action_terran = bot_opponent.step(obs_list[1])
                        
                        obs_list = env.step([action_protoss, action_terran]) 

                    except Exception as e:
                        print(f"Game Loop Error: {e}")
                        traceback.print_exc()
                        break 

                    if bot_hybrid.use_ai:
                        env_reward = obs_list[0].reward
                        hp_reward = bot_hybrid.calculate_custom_reward(obs_list[0])
                        final_step_reward = env_reward + hp_reward
                        bot_hybrid.memory_rewards.append(final_step_reward)
                        total_reward += final_step_reward
                    
                    if obs_list[0].last(): break
                
                if bot_hybrid.use_ai:
                    loss = update_ppo(bot_hybrid)
                    bot_hybrid.update_epsilon()
                    print(f"Ep {episode+1} | Reward: {total_reward:.4f} | Loss: {loss:.4f} | Epsilon: {bot_hybrid.epsilon:.4f}")
                    writer.add_scalar('Hybrid/Reward', total_reward, episode)
                else:
                    print(f"Ep {episode+1} | Script Running (No AI update)")
                
                writer.flush()
                if time.time() - last_save_time > SAVE_INTERVAL_MINUTES * 60:
                    save_model(bot_hybrid, model_dir, episode + 1)
                    last_save_time = time.time()

    except KeyboardInterrupt:
        print("User interrupted.")
    finally:
        writer.close()

if __name__ == "__main__":
    app.run(main)