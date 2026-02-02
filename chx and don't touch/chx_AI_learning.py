import sys
import random

# =========================================================
# 🚑 救命補丁：修復 Python 3.11+ 的 Random.shuffle 錯誤
# (這段一定要放在最上面，比 pysc2 還要早執行)
# =========================================================
_original_shuffle = random.shuffle
def _patched_shuffle(x, random=None):
    _original_shuffle(x)
random.shuffle = _patched_shuffle
# =========================================================

import numpy as np
import pandas as pd
import os
import time
from absl import app
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

# --- 遊戲常數定義 ---
COMMAND_CENTER_ID = 18
SUPPLY_DEPOT_ID = 19
REFINERY_ID = 20
BARRACKS_ID = 21
BARRACKS_TECHLAB_ID = 37
SCV_ID = 45
MARAUDER_ID = 51
VESPENE_GEYSER_ID = 342

# --- Q-Learning 大腦 ---
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=float)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table.loc[state] = [0] * len(self.actions)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

# --- 智能代理人 (Marauder Agent) ---
class MarauderAgent:
    def __init__(self):
        self.action_space = [0, 1, 2, 3, 4, 5, 6]
        self.q_learn = QLearningTable(actions=list(range(len(self.action_space))))
        self.previous_state = None
        self.previous_action = None
        self.reset_episode_vars()
        self.cc_y, self.cc_x = None, None

    def reset_episode_vars(self):
        self.marauder_count = 0
        self.barracks_rewarded = False
        self.techlab_rewarded = False
        self.last_depots = 0
        self.last_refineries = 0

    def get_state(self, obs):
        unit_type = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
        player = obs.observation.player
        
        depot_count = int(np.sum(unit_type == SUPPLY_DEPOT_ID) / 60)
        barracks_count = int(np.sum(unit_type == BARRACKS_ID) / 100)
        techlab_count = int(np.sum(unit_type == BARRACKS_TECHLAB_ID) / 50)
        refinery_count = int(np.sum(unit_type == REFINERY_ID) / 80)
        marauder_count = int(np.sum(unit_type == MARAUDER_ID) / 20)
        can_afford = 1 if player.minerals >= 100 and player.vespene >= 25 else 0
        
        return str((depot_count, barracks_count, techlab_count, refinery_count, marauder_count, can_afford))

    def step(self, obs):
        current_state = self.get_state(obs)
        unit_type = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
        
        curr_marauders = int(np.sum(unit_type == MARAUDER_ID) / 20)
        curr_barracks = int(np.sum(unit_type == BARRACKS_ID) / 100)
        curr_techlabs = int(np.sum(unit_type == BARRACKS_TECHLAB_ID) / 50)
        curr_depots = int(np.sum(unit_type == SUPPLY_DEPOT_ID) / 60)
        curr_refineries = int(np.sum(unit_type == REFINERY_ID) / 80)

        # 計分邏輯
        reward = 0
        if curr_marauders > self.marauder_count:
            reward += 100
            print(f"🎉 成功生產掠奪者！(+100分)")
        if curr_barracks > 0 and not self.barracks_rewarded:
            reward += 20
            self.barracks_rewarded = True
            print("🏗️ 蓋出兵營 (+20分)")
        if curr_techlabs > 0 and not self.techlab_rewarded:
            reward += 20
            self.techlab_rewarded = True
            print("🧪 研發科技 (+20分)")
        if curr_depots > self.last_depots and curr_depots <= 2:
            reward += 5
        if curr_refineries > self.last_refineries and curr_refineries <= 2:
            reward += 5
        reward -= 0.05

        self.marauder_count = curr_marauders
        self.last_depots = curr_depots
        self.last_refineries = curr_refineries

        if self.previous_state:
            self.q_learn.learn(self.previous_state, self.previous_action, reward, current_state)

        action_id = self.q_learn.choose_action(current_state)
        self.previous_state = current_state
        self.previous_action = action_id

        return self.do_action(obs, action_id)

    def do_action(self, obs, action_id):
        unit_type = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
        available = obs.observation.available_actions
        
        cc_y, cc_x = (unit_type == COMMAND_CENTER_ID).nonzero()
        if cc_x.any():
            self.cc_x, self.cc_y = int(cc_x.mean()), int(cc_y.mean())
        
        if action_id == 0: return actions.FUNCTIONS.no_op()
        elif action_id == 1:
            if actions.FUNCTIONS.Train_SCV_quick.id in available: return actions.FUNCTIONS.Train_SCV_quick("now")
            return self._select_unit(unit_type, COMMAND_CENTER_ID)
        elif action_id == 2:
            if actions.FUNCTIONS.Build_SupplyDepot_screen.id in available and self.cc_x:
                target = (np.clip(self.cc_x + 15, 0, 83), np.clip(self.cc_y + 15, 0, 83))
                return actions.FUNCTIONS.Build_SupplyDepot_screen("now", target)
            return self._select_worker(unit_type)
        elif action_id == 3:
            if actions.FUNCTIONS.Build_Refinery_screen.id in available:
                y, x = (unit_type == VESPENE_GEYSER_ID).nonzero()
                if x.any(): return actions.FUNCTIONS.Build_Refinery_screen("now", (x[0], y[0]))
            return self._select_worker(unit_type)
        elif action_id == 4:
            if actions.FUNCTIONS.Build_Barracks_screen.id in available and self.cc_x:
                target = (np.clip(self.cc_x + 25, 0, 83), np.clip(self.cc_y, 0, 83))
                return actions.FUNCTIONS.Build_Barracks_screen("now", target)
            return self._select_worker(unit_type)
        elif action_id == 5:
            if actions.FUNCTIONS.Build_TechLab_quick.id in available: return actions.FUNCTIONS.Build_TechLab_quick("now")
            return self._select_unit(unit_type, BARRACKS_ID)
        elif action_id == 6:
            if actions.FUNCTIONS.Train_Marauder_quick.id in available: return actions.FUNCTIONS.Train_Marauder_quick("now")
            return self._select_unit(unit_type, BARRACKS_ID)
        return actions.FUNCTIONS.no_op()

    def _select_unit(self, unit_type, uid):
        y, x = (unit_type == uid).nonzero()
        if x.any(): return actions.FUNCTIONS.select_point("select", (int(x.mean()), int(y.mean())))
        return actions.FUNCTIONS.no_op()
        
    def _select_worker(self, unit_type):
        y, x = (unit_type == SCV_ID).nonzero()
        if x.any(): 
            idx = random.randint(0, len(x)-1)
            return actions.FUNCTIONS.select_point("select", (x[idx], y[idx]))
        return actions.FUNCTIONS.no_op()

# --- 主程式 ---
def main(argv):
    agent = MarauderAgent()
    try:
        with sc2_env.SC2Env(
            map_name="Simple64",
            players=[sc2_env.Agent(sc2_env.Race.terran),
                     sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.very_easy)],
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64),
                use_raw_units=False),
            step_mul=8,
            game_steps_per_episode=0, 
            visualize=False,
            realtime=False,
        ) as env:
            
            for episode in range(50):
                print(f"🔥 開始第 {episode+1} 場訓練...")
                obs_list = env.reset()
                agent.reset_episode_vars()
                agent.previous_state = None
                
                while True:
                    sc2_action = agent.step(obs_list[0])
                    obs_list = env.step([sc2_action])
                    if obs_list[0].last():
                        print(f"第 {episode+1} 場結束。")
                        agent.q_learn.q_table.to_csv(f"q_table.csv")
                        break
                        
    except KeyboardInterrupt:
        print("停止訓練")

if __name__ == "__main__":
    app.run(main)