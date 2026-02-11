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
import re

from absl import app
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from torch.utils.tensorboard import SummaryWriter

# ===   【使用者設定區】   ===

SAVE_INTERVAL_MINUTES = 15
MAX_TO_KEEP = 50
MAX_STEPS_PER_EPISODE = 10000 

# === 1. 全域參數與常數設定 ===
LR = 0.0003
GAMMA = 0.99
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOTAL_EPISODES = 1000000 
PROBE_TARGET = 15 

# --- Epsilon (探索率) 設定 ---
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

# === AI 動作列表 (擴充版) ===
# 0: 發呆
ACTION_DO_NOTHING = 0    
# 1: 進攻 (小地圖) -> 用於大範圍轉移
ACTION_ATTACK_MINIMAP = 1 
# 2: 全選 (F2) -> 用於集結
ACTION_SELECT_ARMY = 2   
# 3: 進攻 (螢幕) -> 微操 A-Move
ACTION_ATTACK_SCREEN = 3
# 4: 點選單位 (螢幕) -> 微操 單點拉兵
ACTION_SELECT_UNIT = 4
# 5: 移動視角 (小地圖) -> AI 自主決定看哪裡
ACTION_MOVE_CAMERA = 5

NUM_ACTIONS = 6  # 總動作數 6

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
        self.model = ActorCritic(num_inputs=84*84, num_actions=NUM_ACTIONS).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.memory_states = []
        self.memory_actions = []
        self.memory_logprobs = []
        self.memory_rewards = [] 
        self.memory_values = []
        self.epsilon = EPSILON_START
        self.episode_counter = 0 
        self.reset_game_variables()

    def auto_load_latest_model(self, model_dir):
        files = glob.glob(os.path.join(model_dir, "hybrid_model_ep*.pth"))
        if not files:
            print(">>> ⚠️ 沒有找到舊存檔，將從【第 1 場】開始全新訓練。")
            return False

        latest_file = max(files, key=os.path.getmtime)
        
        try:
            checkpoint = torch.load(latest_file, map_location=DEVICE)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                print(f"📂 讀取完整存檔 (包含 Epsilon 與優化器)...")
                # 檢查權重形狀是否匹配
                saved_actor_weight = checkpoint['model_state_dict']['actor.weight']
                if saved_actor_weight.shape[0] != NUM_ACTIONS:
                    print(f"⚠️ 偵測到動作空間改變 ({saved_actor_weight.shape[0]} -> {NUM_ACTIONS})，將重置 Actor 最後一層。")
                    del checkpoint['model_state_dict']['actor.weight']
                    del checkpoint['model_state_dict']['actor.bias']
                    self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                self.epsilon = checkpoint.get('epsilon', EPSILON_START)
                self.episode_counter = checkpoint.get('episode', 0)
            else:
                print("⚠️ 舊版存檔不相容，建議重新訓練。")
                return False
                
            self.model.train()
            print(f"==================================================")
            print(f"✅ 自動載入存檔成功！")
            print(f"📂 檔案: {latest_file}")
            print(f"🔢 進度: 第 {self.episode_counter} 場 | 🎲 當前 Epsilon: {self.epsilon:.5f}")
            print(f"==================================================")
            return True
        except Exception as e:
            print(f"❌ 載入存檔失敗 ({latest_file}): {e}")
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
        self.enemy_base_minimap = (32, 32) # 預設敵人位置

    def update_epsilon(self, total_reward):
        if total_reward > 0:
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

        # === 新增：選取探測機懲罰 (-0.5) ===
        if hasattr(obs.observation, 'raw_units'):
            for unit in obs.observation.raw_units:
                # 檢查是否有被選取且是 Probe
                if unit.is_selected and unit.unit_type == PROBE_ID:
                    reward -= 0.5
                    break
        
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

    def step_ai(self, obs):
        state_tensor = self.preprocess(obs)
        action_probs, state_value = self.model(state_tensor)
        
        # === Epsilon 探索：排除 0 (發呆)，嘗試 1 ~ NUM_ACTIONS-1 ===
        if random.random() < self.epsilon:
            action_index = torch.tensor(random.randint(1, NUM_ACTIONS - 1)).to(DEVICE) 
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
        if action_cmd == ACTION_DO_NOTHING:
            return NO_OP
        elif action_cmd == ACTION_ATTACK_MINIMAP:
            target = self.enemy_base_minimap if self.enemy_base_minimap else MAP_CENTER_MINIMAP
            if actions.FUNCTIONS.Attack_minimap.id in obs.observation.available_actions:
                return actions.FUNCTIONS.Attack_minimap("now", target)
            elif actions.FUNCTIONS.select_army.id in obs.observation.available_actions:
                return actions.FUNCTIONS.select_army("select")
        elif action_cmd == ACTION_SELECT_ARMY:
            if actions.FUNCTIONS.select_army.id in obs.observation.available_actions:
                return actions.FUNCTIONS.select_army("select")
        elif action_cmd == ACTION_ATTACK_SCREEN:
            if actions.FUNCTIONS.Attack_screen.id in obs.observation.available_actions:
                player_relative = obs.observation.feature_screen.player_relative
                y_enemy, x_enemy = (player_relative == features.PlayerRelative.ENEMY).nonzero()
                if x_enemy.any():
                    idx = random.randint(0, len(x_enemy) - 1)
                    target = (x_enemy[idx], y_enemy[idx])
                else:
                    target = MAP_CENTER_SCREEN 
                return actions.FUNCTIONS.Attack_screen("now", target)
            elif actions.FUNCTIONS.select_army.id in obs.observation.available_actions:
                return actions.FUNCTIONS.select_army("select")
        elif action_cmd == ACTION_SELECT_UNIT:
            if actions.FUNCTIONS.select_point.id in obs.observation.available_actions:
                unit_type = obs.observation.feature_screen.unit_type
                y_self, x_self = (unit_type == STALKER_ID).nonzero()
                if not x_self.any():
                    player_relative = obs.observation.feature_screen.player_relative
                    y_self, x_self = (player_relative == features.PlayerRelative.SELF).nonzero()
                if x_self.any():
                    idx = random.randint(0, len(x_self) - 1)
                    target = (x_self[idx], y_self[idx])
                    return actions.FUNCTIONS.select_point("select", target)
            return NO_OP
        elif action_cmd == ACTION_MOVE_CAMERA:
            if MOVE_CAMERA_ACTION in obs.observation.available_actions:
                player_relative_mini = obs.observation.feature_minimap.player_relative
                y_self, x_self = (player_relative_mini == features.PlayerRelative.SELF).nonzero()
                if random.random() < 0.5 and x_self.any():
                    target = (int(x_self.mean()), int(y_self.mean()))
                else:
                    target = self.enemy_base_minimap
                return actions.FUNCTIONS.move_camera(target)

        return NO_OP
    
    # === 腳本邏輯 (含基地修正與嚴格兵力控制) ===
    def step_script(self, obs):
        unit_type_screen = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
        current_supply_cap = obs.observation.player.food_cap
        current_minerals = obs.observation.player.minerals
        current_vespene = obs.observation.player.vespene
        current_workers = obs.observation.player.food_workers
        current_army_supply = obs.observation.player.food_army
        
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

            player_relative_mini = obs.observation.feature_minimap.player_relative
            y_mini, x_mini = (player_relative_mini == features.PlayerRelative.SELF).nonzero()
            
            if x_mini.any():
                mean_x = x_mini.mean()
                if mean_x < 32:
                    print("[Script] Base: Top-Left -> Enemy: Bottom-Right")
                    self.enemy_base_minimap = (45, 45) 
                else:
                    print("[Script] Base: Bottom-Right -> Enemy: Top-Left")
                    self.enemy_base_minimap = (19, 23)

        # --- 狀態機 ---
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
            # === 修改：嚴格控制 5 隻追獵者生成 ===
            # 只有當人口 >= 10 且 實體數量 >= 5 時才出發
            if current_army_supply >= 10:
                real_stalker_count = 0
                if hasattr(obs.observation, 'raw_units'):
                    for unit in obs.observation.raw_units:
                        if unit.unit_type == STALKER_ID and unit.alliance == 1:
                            real_stalker_count += 1
                
                if real_stalker_count >= 5:
                    self.state = 9
                    print(f"[Protoss] 5 Stalkers Ready & Spawned. Launching Attack.")
                    return NO_OP
                else:
                    # 等待生成
                    return NO_OP
            
            # 不到 5 隻，允許訓練
            elif TRAIN_STALKER_ACTION in obs.observation.available_actions:
                if current_minerals >= 125 and current_vespene >= 50:
                    if current_supply_cap - current_workers - current_army_supply >= 2: 
                         self.stalkers_trained += 1 
                         return actions.FUNCTIONS.Train_Stalker_quick("now")
                return NO_OP
            else:
                return get_select_gateway_action()

        elif self.state == 9:
            # Step 1: F2 全選
            if actions.FUNCTIONS.select_army.id in obs.observation.available_actions:
                self.state = 10
                return actions.FUNCTIONS.select_army("select")
            return NO_OP

        elif self.state == 10:
            # Step 2: A-Move 到敵人基地 (Minimap)
            target = self.enemy_base_minimap
            if actions.FUNCTIONS.Attack_minimap.id in obs.observation.available_actions:
                self.state = 11
                return actions.FUNCTIONS.Attack_minimap("now", target)
            return NO_OP

        elif self.state == 11:
            # Step 3: 鏡頭移過去
            if MOVE_CAMERA_ACTION in obs.observation.available_actions:
                self.state = 12
                return actions.FUNCTIONS.move_camera(self.enemy_base_minimap)
            self.state = 12
            return NO_OP

        elif self.state == 12:
            # Step 4: 交給 AI
            self.use_ai = True
            print("====================================")
            print("   Protoss Attacking -> AI Taking Over")
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
        self.own_minimap_coords = None
        self.enemy_minimap_coords = None
        
        self.form_active = False
        self.form_type = None
        self.form_queue = []
        self.form_idx = 0
        self.form_cooldown = 0

    def _clip(self, x, y, lo=0, hi=83): return (int(np.clip(x, lo, hi)), int(np.clip(y, lo, hi)))

    def _connected_centroids(self, mask_bool):
        h, w = mask_bool.shape; seen = np.zeros((h, w), dtype=np.uint8); cents = []
        for y in range(h):
            for x in range(w):
                if not mask_bool[y, x] or seen[y, x]: continue
                stack = [(x, y)]; seen[y, x] = 1; pix = []
                while stack:
                    px, py = stack.pop(); pix.append((px, py))
                    for nx, ny in ((px-1,py),(px+1,py),(px,py-1),(px,py+1)):
                        if 0 <= nx < w and 0 <= ny < h and mask_bool[ny, nx] and not seen[ny, nx]: seen[ny, nx] = 1; stack.append((nx, ny))
                xs, ys = [p[0] for p in pix], [p[1] for p in pix]
                cents.append((int(round(sum(xs)/len(xs))), int(round(sum(ys)/len(ys)))))
        return cents

    def _get_5_marauder_points(self, unit_type_screen):
        mask = (unit_type_screen == MARAUDER_ID); pts = self._connected_centroids(mask)
        return sorted(pts, key=lambda p: (p[0], p[1]))[:5]

    def _dir_and_perp(self):
        if self.own_minimap_coords and self.enemy_minimap_coords:
            dx = self.enemy_minimap_coords[0] - self.own_minimap_coords[0]
            dy = self.enemy_minimap_coords[1] - self.own_minimap_coords[1]
            dx = 0 if dx == 0 else (1 if dx > 0 else -1)
            dy = 0 if dy == 0 else (1 if dy > 0 else -1)
        else: dx, dy = 0, -1
        return (dx, dy), (-dy, dx)

    def _build_slots_marauder(self, form_type, center_xy):
        cx, cy = center_xy; (dx, dy), (px, py) = self._dir_and_perp(); s = 8 if form_type == "SPREAD" else 6
        if form_type == "CONCAVE": offsets = [(-2*s, +s), (2*s, +s), (-s, -s), (0, -s), (s, -s)]
        else: offsets = [(-2*s, 0), (-s, 0), (0, 0), (s, 0), (2*s, 0)]
        slots = [(cx + ox*px + oy*dx, cy + ox*py + oy*dy) for (ox, oy) in offsets]
        return sorted([self._clip(x, y) for (x, y) in slots], key=lambda p: p[0])

    def _start_marauder_formation(self, unit_type_screen, form_type):
        marauders = self._get_5_marauder_points(unit_type_screen)
        if len(marauders) < 5: return False
        marauders = sorted(marauders, key=lambda p: p[0])
        cx = int(round(sum([p[0] for p in marauders]) / 5)); cy = int(round(sum([p[1] for p in marauders]) / 5))
        slots = self._build_slots_marauder(form_type, self._clip(cx, cy))
        self.form_active = True; self.form_type = form_type; self.form_queue = list(zip(marauders, slots)); self.form_idx = 0; self.form_cooldown = 16
        return True

    def update_camera_to_army(self, obs):
        player_relative = obs.observation.feature_minimap[features.MINIMAP_FEATURES.player_relative.index]
        y_mini, x_mini = (player_relative == features.PlayerRelative.SELF).nonzero()
        if not x_mini.any(): return False
        if MOVE_CAMERA_ACTION in obs.observation.available_actions:
            target_x, target_y = int(x_mini.mean()), int(y_mini.mean())
            camera_map = obs.observation.feature_minimap.camera
            y_cam, x_cam = camera_map.nonzero()
            if x_cam.any():
                curr_x, curr_y = x_cam.mean(), y_cam.mean()
                if math.sqrt((target_x-curr_x)**2 + (target_y-curr_y)**2) < 5: return NO_OP
            return actions.FUNCTIONS.move_camera((target_x, target_y))
        return NO_OP

    def step(self, obs):
        unit_type_screen = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
        current_minerals = obs.observation.player.minerals
        current_vespene = obs.observation.player.vespene
        current_workers = obs.observation.player.food_workers
        current_supply_cap = obs.observation.player.food_cap
        player_relative_mini = obs.observation.feature_minimap[features.MINIMAP_FEATURES.player_relative.index]
        y_self, x_self = (player_relative_mini == features.PlayerRelative.SELF).nonzero()
        if x_self.any(): self.own_minimap_coords = (int(x_self.mean()), int(y_self.mean()))
        y_enemy, x_enemy = (player_relative_mini == features.PlayerRelative.ENEMY).nonzero()
        if x_enemy.any(): self.enemy_minimap_coords = (int(x_enemy.mean()), int(y_enemy.mean()))

        def count_units(unit_id): return np.sum((unit_type_screen == unit_id).astype(int))
        
        def get_select_scv_action(select_type="select"):
            y, x = (unit_type_screen == SCV_ID).nonzero()
            if x.any():
                mask = np.ones(len(x), dtype=bool) 
                for busy_x, busy_y in self.busy_depot_locations: mask &= (np.sqrt((x - busy_x)**2 + (y - busy_y)**2) > 10)
                if self.supply_depot_target: mask &= (np.sqrt((x - self.supply_depot_target[0])**2 + (y - self.supply_depot_target[1])**2) > 10)
                if self.refinery_target: mask &= (np.sqrt((x - self.refinery_target[0])**2 + (y - self.refinery_target[1])**2) > 15)
                if mask.any(): idx = random.choice(np.where(mask)[0]); return actions.FUNCTIONS.select_point(select_type, (x[idx], y[idx]))
                idx = random.randint(0, len(x) - 1); return actions.FUNCTIONS.select_point(select_type, (x[idx], y[idx]))
            return NO_OP

        if self.form_cooldown > 0: self.form_cooldown -= 1
        if self.form_active:
            if self.form_idx >= len(self.form_queue): self.form_active = False
            else:
                (sx, sy), (tx, ty) = self.form_queue[self.form_idx]
                if actions.FUNCTIONS.select_point.id in obs.observation.available_actions: return actions.FUNCTIONS.select_point("select", (sx, sy))
                if actions.FUNCTIONS.Attack_screen.id in obs.observation.available_actions: self.form_idx += 1; return actions.FUNCTIONS.Attack_screen("now", (tx, ty))
                return NO_OP
        
        if self.is_first_step:
            self.is_first_step = False
            mineral_coords = (unit_type_screen == MINERAL_FIELD_ID).nonzero()
            if mineral_coords[0].any(): self.initial_mineral_coords = (mineral_coords[1][0], mineral_coords[0][0])
            cc_coords = (unit_type_screen == COMMAND_CENTER_ID).nonzero()
            if cc_coords[0].any(): self.cc_x_screen, self.cc_y_screen = int(cc_coords[1].mean()), int(cc_coords[0].mean())
            if self.own_minimap_coords: self.base_minimap_coords = self.own_minimap_coords
            self.last_depot_pixels = count_units(SUPPLY_DEPOT_ID)

        if self.state == -1:
            if HARVEST_ACTION in obs.observation.available_actions and self.initial_mineral_coords: self.state = 0; return actions.FUNCTIONS.Harvest_Gather_screen("now", self.initial_mineral_coords)
            else: return get_select_scv_action("select_all_type")
        elif self.state == 0:
            if current_workers < 15:
                if TRAIN_SCV_ACTION in obs.observation.available_actions and current_minerals >= 50: return actions.FUNCTIONS.Train_SCV_quick("now")
                elif current_minerals >= 50: 
                    y, x = (unit_type_screen == COMMAND_CENTER_ID).nonzero()
                    if x.any(): return actions.FUNCTIONS.select_point("select", (int(x.mean()), int(y.mean())))
            elif current_minerals >= 100:
                self.state = 1
                off_x = 15 if self.initial_mineral_coords[0] < self.cc_x_screen else -15; off_y = 15 if self.initial_mineral_coords[1] < self.cc_y_screen else -15
                self.supply_depot_target = (np.clip(self.cc_x_screen+off_x,0,83).astype(int), np.clip(self.cc_y_screen+off_y,0,83).astype(int))
                self.first_depot_coords = self.supply_depot_target; self.build_dir = (1 if off_x>0 else -1, 1 if off_y>0 else -1)
                return get_select_scv_action()
            return NO_OP
        elif self.state == 1:
            if BUILD_SUPPLYDEPOT_ACTION in obs.observation.available_actions and current_minerals >= 100: self.state = 1.1; self.select_attempts = 0; return actions.FUNCTIONS.Build_SupplyDepot_screen("now", self.supply_depot_target)
            elif current_minerals >= 100: return get_select_scv_action()
            return NO_OP
        elif self.state == 1.1:
            cur = count_units(SUPPLY_DEPOT_ID)
            if cur > self.last_depot_pixels + 5:
                self.depots_built += 1; self.last_depot_pixels = cur; self.busy_depot_locations.append(self.supply_depot_target)
                if self.depots_built < 3:
                    dx, dy = self.build_dir; bx, by = self.first_depot_coords
                    if self.depots_built == 1: nx, ny = bx + 12*dx, by
                    else: nx, ny = bx + 6*dx, by + 12*dy
                    self.supply_depot_target = (np.clip(nx,0,83).astype(int), np.clip(ny,0,83).astype(int))
                    self.state = 1; return get_select_scv_action()
                else: self.state = 2; return NO_OP
            self.select_attempts += 1
            if self.select_attempts > 100:
                self.supply_depot_target = (np.clip(self.supply_depot_target[0]+random.randint(-15,15),0,83), np.clip(self.supply_depot_target[1]+random.randint(-15,15),0,83))
                self.state = 1; return get_select_scv_action()
            return NO_OP
        elif self.state == 2:
            if BUILD_REFINERY_ACTION in obs.observation.available_actions and current_minerals >= 75:
                if not self.refinery_target:
                    y, x = (unit_type_screen == GEYSER_ID).nonzero()
                    if x.any(): 
                        mask = (np.abs(x-x[0])<10)&(np.abs(y-y[0])<10); self.refinery_target = (int(x[mask].mean()), int(y[mask].mean()))
                    else: return NO_OP
                self.state = 2.1; return actions.FUNCTIONS.Build_Refinery_screen("now", self.refinery_target)
            else: return get_select_scv_action()
        elif self.state == 2.1:
            if count_units(REFINERY_ID) > 0: self.state = 3; self.gas_workers_assigned = 0; self.selecting_worker = True; self.recent_selected_coords = []; return NO_OP
            return NO_OP
        elif self.state == 3:
            if self.gas_workers_assigned < 3:
                if self.selecting_worker:
                    y, x = (unit_type_screen == SCV_ID).nonzero()
                    if x.any():
                        mask = np.sqrt((x-self.refinery_target[0])**2+(y-self.refinery_target[1])**2) > 15
                        for px, py in self.recent_selected_coords: mask &= (np.sqrt((x-px)**2+(y-py)**2) > 5)
                        if mask.any(): idx = random.choice(np.where(mask)[0]); tx, ty = x[idx], y[idx]
                        else: idx = random.randint(0, len(x)-1); tx, ty = x[idx], y[idx]
                        self.recent_selected_coords.append((tx,ty)); self.selecting_worker=False; return actions.FUNCTIONS.select_point("select", (tx,ty))
                    return NO_OP
                else:
                    if HARVEST_ACTION in obs.observation.available_actions: self.gas_workers_assigned+=1; self.selecting_worker=True; return actions.FUNCTIONS.Harvest_Gather_screen("now", self.refinery_target)
                    else: self.selecting_worker=True; return NO_OP
            else: self.state = 4; return NO_OP
        elif self.state == 4:
            if not self.camera_centered:
                 if self.base_minimap_coords: self.camera_centered=True; return actions.FUNCTIONS.move_camera(self.base_minimap_coords)
            if BUILD_BARRACKS_ACTION in obs.observation.available_actions and current_minerals >= 150:
                if not self.barracks_target:
                    mx = self.base_minimap_coords[0] if self.base_minimap_coords else 15
                    off = 30 if mx <= 32 else -30
                    self.barracks_target = (np.clip(42+off+random.randint(-5,5),0,83).astype(int), np.clip(42+random.randint(-5,5),0,83).astype(int))
                self.state = 4.1; self.select_attempts = 0; return actions.FUNCTIONS.Build_Barracks_screen("now", self.barracks_target)
            else: return get_select_scv_action()
        elif self.state == 4.1:
            if count_units(BARRACKS_ID) > 0: self.state = 5; return NO_OP
            self.select_attempts += 1
            if self.select_attempts > 100: self.state = 4; self.barracks_target=None; return NO_OP
            return NO_OP
        elif self.state == 5:
            if count_units(BARRACKS_TECHLAB_ID) > 0: self.state = 7; return NO_OP
            if BUILD_TECHLAB_ACTION in obs.observation.available_actions and current_minerals >= 50 and current_vespene >= 25:
                self.state = 6; return actions.FUNCTIONS.Build_TechLab_quick("now")
            else:
                 y, x = (unit_type_screen == BARRACKS_ID).nonzero()
                 if x.any(): return actions.FUNCTIONS.select_point("select", (int(x.mean()), int(y.mean())))
            return NO_OP
        elif self.state == 6:
            if count_units(BARRACKS_TECHLAB_ID) > 0: self.state = 7; return NO_OP
            return NO_OP
        elif self.state == 7:
            if self.marauders_trained < 5:
                if TRAIN_MARAUDER_ACTION in obs.observation.available_actions:
                    if current_minerals >= 100 and current_vespene >= 25 and current_supply_cap - current_workers - self.marauders_trained*2 >= 2:
                        self.marauders_trained += 1; return actions.FUNCTIONS.Train_Marauder_quick("now")
                    return NO_OP
                else: 
                     y, x = (unit_type_screen == BARRACKS_ID).nonzero()
                     if x.any(): return actions.FUNCTIONS.select_point("select", (int(x.mean()), int(y.mean())))
            else: self.state = 8; return NO_OP
        elif self.state == 8:
            cam_act = self.update_camera_to_army(obs)
            if cam_act != NO_OP: return cam_act
            if not self.enemy_minimap_coords: return NO_OP
            if (not self.form_active) and self.form_cooldown == 0:
                if self.own_minimap_coords:
                    d2 = (self.enemy_minimap_coords[0]-self.own_minimap_coords[0])**2 + (self.enemy_minimap_coords[1]-self.own_minimap_coords[1])**2
                    if d2 <= 100: desired = "SPREAD"
                    elif d2 <= 400: desired = "CONCAVE"
                    else: desired = "LINE"
                    if self._start_marauder_formation(unit_type_screen, desired): return NO_OP
            self.state = 9
            if actions.FUNCTIONS.select_army.id in obs.observation.available_actions: return actions.FUNCTIONS.select_army("select")
            return NO_OP
        elif self.state == 9:
             if self.enemy_minimap_coords and actions.FUNCTIONS.Attack_minimap.id in obs.observation.available_actions:
                 return actions.FUNCTIONS.Attack_minimap("now", self.enemy_minimap_coords)
             self.state = 8
             return NO_OP
        return NO_OP

# === 5. PPO 更新與主程式 ===
def update_ppo(agent):
    if len(agent.memory_rewards) < 5: return 0
    rewards = []
    discounted_reward = 0
    for reward in reversed(agent.memory_rewards):
        discounted_reward = reward + GAMMA * discounted_reward
        rewards.insert(0, discounted_reward)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
    if len(rewards) > 1:
        std_val = rewards.std()
        if torch.isnan(std_val) or std_val < 1e-5: rewards = rewards - rewards.mean()
        else: rewards = (rewards - rewards.mean()) / (std_val + 1e-7)
    else: rewards = rewards - rewards.mean()

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
    torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 0.5)
    agent.optimizer.step()
    return total_loss.item()

def save_model(agent, model_dir, episode):
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    filename = os.path.join(model_dir, f"hybrid_model_ep{episode}.pth")
    
    save_dict = {
        'model_state_dict': agent.model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'episode': episode
    }
    
    torch.save(save_dict, filename)
    print(f"Model saved: {filename}")
    
    files = sorted(glob.glob(os.path.join(model_dir, "hybrid_model_ep*.pth")), key=os.path.getmtime)
    if len(files) > MAX_TO_KEEP:
        for f in files[:-MAX_TO_KEEP]:
            try: os.remove(f)
            except OSError as e: print(f"Error removing: {e}")

def main(argv):
    print("=== SC2 Hybrid Agent (Infinite Loop - Safe Save Version) ===")
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M_Hybrid")
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", current_time_str)
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models")
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    writer = SummaryWriter(log_dir)

    bot_hybrid = ProtossHybridAgent()
    bot_opponent = TerranBot() 

    if bot_hybrid.auto_load_latest_model(model_dir): start_episode = bot_hybrid.episode_counter
    else: start_episode = 0

    global_step = 0 
    
    while True: # 無限循環防崩潰
        try:
            with sc2_env.SC2Env(
                map_name="Simple64",
                players=[sc2_env.Agent(sc2_env.Race.protoss), sc2_env.Agent(sc2_env.Race.terran)],
                agent_interface_format=sc2_env.AgentInterfaceFormat(
                    feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64), use_raw_units=True),
                step_mul=8,
                game_steps_per_episode=None,
                visualize=True
            ) as env:
                for episode in range(start_episode, TOTAL_EPISODES):
                    bot_hybrid.episode_counter = episode
                    obs_list = env.reset()
                    bot_hybrid.reset_game()
                    bot_opponent = TerranBot()
                    total_reward = 0
                    step_count = 0
                    
                    while True:
                        step_count += 1
                        global_step += 1 
                        
                        try:
                            action_protoss = bot_hybrid.step(obs_list[0])
                            action_terran = bot_opponent.step(obs_list[1])
                            obs_list = env.step([action_protoss, action_terran]) 
                        except Exception as e:
                            print(f"Game Loop Error: {e}")
                            print("⚠️ 偵測到遊戲內錯誤，嘗試緊急存檔...")
                            save_model(bot_hybrid, model_dir, episode)
                            break 

                        if bot_hybrid.use_ai:
                            env_reward = obs_list[0].reward
                            hp_reward = bot_hybrid.calculate_custom_reward(obs_list[0])
                            final_step_reward = env_reward + hp_reward
                            bot_hybrid.memory_rewards.append(final_step_reward)
                            total_reward += final_step_reward
                            if step_count % 100 == 0: writer.add_scalar('RealTime/Running_Score', total_reward, global_step)
                        
                        if step_count >= MAX_STEPS_PER_EPISODE:
                            print(f"⚠️ 強制重置 (Episode {episode+1})")
                            break
                            
                        if obs_list[0].last():
                            print(f"🏁 遊戲結束 (Step: {step_count})")
                            break
                    
                    # === 每場遊戲結束後直接存檔 ===
                    save_model(bot_hybrid, model_dir, episode + 1)

                    if bot_hybrid.use_ai:
                        loss = update_ppo(bot_hybrid)
                        bot_hybrid.update_epsilon(total_reward)
                        print(f"Ep {episode+1} | Reward: {total_reward:.4f} | Loss: {loss:.4f} | Epsilon: {bot_hybrid.epsilon:.4f}")
                        writer.add_scalar('Hybrid/Reward', total_reward, episode)
                    else: print(f"Ep {episode+1} | Script Running")
                    
                    writer.flush()
                    start_episode = episode + 1

        except Exception as e:
            print(f"❌ 發生致命錯誤，嘗試緊急存檔後 5秒重啟: {e}")
            try:
                save_model(bot_hybrid, model_dir, bot_hybrid.episode_counter)
            except:
                print("無法執行緊急存檔。")
            traceback.print_exc()
            time.sleep(5)
            continue
        except KeyboardInterrupt:
            print("User interrupted.")
            save_model(bot_hybrid, model_dir, bot_hybrid.episode_counter)
            break
        finally: writer.close()

if __name__ == "__main__":
    app.run(main)