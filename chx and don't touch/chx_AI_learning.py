import sys
import random
import os

# =========================================================
# 🚑 救命補丁 1：指定 D 槽路徑
# =========================================================
os.environ["SC2PATH"] = "D:/StarCraft II"

# =========================================================
# 🚑 救命補丁 2：修復 Python 版本相容性
# =========================================================
try:
    _original_shuffle = random.shuffle
    def _patched_shuffle(x, random=None):
        _original_shuffle(x)
    random.shuffle = _patched_shuffle
except Exception:
    pass

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from collections import defaultdict
from absl import app
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

# =========================================================
# 🏗️ 遊戲單位 ID 對照表
# =========================================================
COMMAND_CENTER_ID = 18       
SUPPLY_DEPOT_ID = 19         
REFINERY_ID = 20             
BARRACKS_ID = 21             
BARRACKS_TECHLAB_ID = 37     
SCV_ID = 45                  
MARAUDER_ID = 51             
VESPENE_GEYSER_ID = 342      

# =========================================================
# 📊 數據收集器
# =========================================================
class DataCollector:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(os.path.dirname(__file__), "logs")
        self.results_dir = os.path.join(os.path.dirname(__file__), "results")
        
        if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)
        if not os.path.exists(self.results_dir): os.makedirs(self.results_dir)
        
        self.training_log_file = os.path.join(self.log_dir, f"training_log_{self.timestamp}.csv")
        self.episode_summary_file = os.path.join(self.log_dir, f"episode_summary_{self.timestamp}.csv")
        self.q_table_file = os.path.join(self.log_dir, f"q_table_{self.timestamp}.csv")
        self.excel_file = os.path.join(self.results_dir, f"excel_report_{self.timestamp}.xlsx")
        
        self.episode_data = []
        self.step_data = []
        self.q_table_history = []
        self._write_csv_headers()

    def _write_csv_headers(self):
        with open(self.training_log_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Step", "Game_Loop", "Minerals", "Vespene", "Workers", 
                             "Depots", "Barracks", "Techlabs", "Refineries", "Marauders", 
                             "Action_ID", "Action_Name", "Reward", "Total_Reward", "Is_Complete"])
        
        with open(self.episode_summary_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Total_Steps", "Max_Reward", "Final_Marauders", 
                             "Completion_Time", "Win_Rate", "Avg_Reward_Per_Step"])

    def log_step(self, **kwargs):
        self.step_data.append(kwargs)

    def log_episode_summary(self, **kwargs):
        self.episode_data.append(kwargs)

    def save_q_table(self, q_table, episode):
        q_table_copy = q_table.copy()
        q_table_copy['Episode'] = episode
        self.q_table_history.append(q_table_copy)

    def export_to_excel(self):
        try:
            with pd.ExcelWriter(self.excel_file, engine='openpyxl') as writer:
                if self.step_data: pd.DataFrame(self.step_data).to_excel(writer, sheet_name='Training_Log', index=False)
                if self.episode_data: pd.DataFrame(self.episode_data).to_excel(writer, sheet_name='Episode_Summary', index=False)
                if self.q_table_history: pd.concat(self.q_table_history, ignore_index=True).to_excel(writer, sheet_name='Q_Table_History', index=False)
                if self.episode_data:
                    df = pd.DataFrame(self.episode_data)
                    stats = {
                        'Total_Episodes': len(df),
                        'Avg_Reward': df['Max_Reward'].mean(),
                        'Max_Reward': df['Max_Reward'].max(),
                        'Completion_Rate': (df['Final_Marauders'] >= 5).sum() / len(df) * 100
                    }
                    pd.DataFrame([stats]).to_excel(writer, sheet_name='Statistics', index=False)
        except Exception as e:
            print(f"⚠️ Excel 存檔失敗 (可能是檔案被開啟或缺少 openpyxl): {e}")

    def generate_plots(self):
        if not self.episode_data: return
        df = pd.DataFrame(self.episode_data)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1.plot(df['Episode'], df['Max_Reward'], 'b-', label='Max Reward')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(df['Episode'], df['Final_Marauders'], 'r-', label='Marauders Count')
        ax2.axhline(y=5, color='g', linestyle='--', label='Target (5)')
        ax2.set_title('Marauder Production')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plot_file = os.path.join(self.results_dir, f"learning_curve_{self.timestamp}.png")
        plt.savefig(plot_file)
        print(f"📊 圖表已保存: {plot_file}")

# =========================================================
# 🧠 Q-Learning 大腦
# =========================================================
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=float)
        
        self.action_names = {
            0: "No_Op",           
            1: "Train_SCV",       
            2: "Build_Depot",     
            3: "Build_Refinery",  
            4: "Build_Barracks",  
            5: "Build_TechLab",   
            6: "Train_Marauder"   
        }

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

    def get_action_name(self, action_id):
        return self.action_names.get(action_id, f"Action_{action_id}")

# =========================================================
# 🎮 智能代理人 (Marauder Agent) - [超級通膨獎勵版]
# =========================================================
class MarauderAgent:
    def __init__(self):
        self.action_space = [0, 1, 2, 3, 4, 5, 6]
        self.q_learn = QLearningTable(actions=list(range(len(self.action_space))))
        self.previous_state = None
        self.previous_action = None
        self.reset_episode_vars()
        self.cc_y, self.cc_x = None, None
        
        # 🔥 自動載入記憶 (依然保留這個功能)
        self.load_brain()

    def load_brain(self):
        """讀取舊的 Q-Table，讓 AI 繼承記憶"""
        brain_path = "q_table_latest.csv" 
        
        if os.path.isfile(brain_path):
            print(f"🧠 發現大腦存檔 ({brain_path})，正在讀取記憶...")
            try:
                # 讀取 CSV，並將第一欄設為 Index (狀態)
                self.q_learn.q_table = pd.read_csv(brain_path, index_col=0)
                # 確保欄位名稱是整數 (動作 ID)
                self.q_learn.q_table.columns = self.q_learn.q_table.columns.astype(int)
                print("✅ 記憶讀取成功！準備進行高獎勵特訓！")
            except Exception as e:
                print(f"⚠️ 讀取失敗: {e}，將使用全新大腦。")
        else:
            print("✨ 沒找到 q_table_latest.csv，AI 將從零開始學習。")

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
        
        free_supply = int(player.food_cap - player.food_used)
        can_afford = 1 if player.minerals >= 100 and player.vespene >= 25 and free_supply >= 2 else 0
        
        return str((depot_count, barracks_count, techlab_count, refinery_count, marauder_count, can_afford, free_supply))

    def step(self, obs):
        current_state = self.get_state(obs)
        unit_type = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
        player = obs.observation.player
        
        curr_marauders = int(np.sum(unit_type == MARAUDER_ID) / 20)
        curr_barracks = int(np.sum(unit_type == BARRACKS_ID) / 100)
        curr_techlabs = int(np.sum(unit_type == BARRACKS_TECHLAB_ID) / 50)
        curr_depots = int(np.sum(unit_type == SUPPLY_DEPOT_ID) / 60)
        curr_refineries = int(np.sum(unit_type == REFINERY_ID) / 80)
        
        free_supply = int(player.food_cap - player.food_used)

        # --- 🏆 獎勵機制核心 (超級通膨版) ---
        reward = 0
        
        # 1. 生產出掠奪者：給 2000 分 (原本 100)
        if curr_marauders > self.marauder_count:
            reward += 2000
            print(f"🎉 成功生產掠奪者！目前總數: {curr_marauders} (+2000分)")

        # 2. 蓋出兵營：給 500 分 (原本 20)
        if curr_barracks > 0 and not self.barracks_rewarded:
            reward += 500
            self.barracks_rewarded = True
            print("🏗️ 蓋出兵營 (+500分)")

        # 3. 研發掛件：給 500 分 (原本 20)
        if curr_techlabs > 0 and not self.techlab_rewarded:
            reward += 500
            self.techlab_rewarded = True
            print("🧪 研發科技掛件 (+500分)")

        # 4. 基礎建設 (小獎勵)
        if curr_depots > self.last_depots:
            reward += 10
        if curr_refineries > self.last_refineries and curr_refineries <= 2:
            reward += 10
            
        # 5. 卡人口懲罰 (維持不變)
        if free_supply == 0 and player.food_cap < 200:
            reward -= 1

        # 6. 時間懲罰 (維持不變)
        reward -= 0.05

        self.marauder_count = curr_marauders
        self.last_depots = curr_depots
        self.last_refineries = curr_refineries

        if self.previous_state:
            self.q_learn.learn(self.previous_state, self.previous_action, reward, current_state)

        action_id = self.q_learn.choose_action(current_state)
        self.previous_state = current_state
        self.previous_action = action_id

        return self.do_action(obs, action_id), reward, curr_marauders

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

# =========================================================
# 🚀 主程式
# =========================================================
def main(argv):
    collector = DataCollector()
    agent = MarauderAgent()
    
    # 🔥 偵測到讀檔時，探索率設為 0.5 (不至於完全不試新東西，但主要依賴經驗)
    if os.path.isfile("q_table_latest.csv"):
        agent.q_learn.epsilon = 0.5 
        print("🚀 檢測到記憶繼承，已降低隨機探索率 (Epsilon -> 0.5)")
    
    print(f"🚀 開始訓練 AI，目標：生產 5 隻掠奪者 (超級通膨版)")
    print(f"📊 數據將保存到：{collector.log_dir}")
    
    try:
        with sc2_env.SC2Env(
            map_name="Simple96",
            players=[sc2_env.Agent(sc2_env.Race.terran),
                     sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.very_easy)],
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64),
                use_raw_units=False),
            step_mul=8,
            game_steps_per_episode=0, 
            visualize=True,
            realtime=False,
        ) as env:
            
            # 訓練 100 場
            for episode in range(100): 
                print(f"\n🔥 開始第 {episode+1}/100 場訓練...")
                obs_list = env.reset()
                agent.reset_episode_vars()
                agent.previous_state = None
                
                total_reward = 0
                final_marauders = 0
                step_count = 0
                max_reward = 0
                is_complete = False
                
                while True:
                    sc2_action, reward, marauders = agent.step(obs_list[0])
                    obs_list = env.step([sc2_action])
                    
                    step_count += 1
                    total_reward += reward
                    final_marauders = marauders
                    max_reward = max(max_reward, total_reward)
                    
                    if marauders >= 5 and not is_complete:
                        is_complete = True
                    
                    if step_count % 10 == 0:
                        action_name = agent.q_learn.get_action_name(agent.previous_action)
                        collector.log_step(
                            Episode=episode + 1, Step=step_count, Marauders=marauders, 
                            Action_Name=action_name, Reward=reward, Total_Reward=total_reward,
                            Is_Complete=is_complete
                        )

                    if obs_list[0].last():
                        win_rate = 1.0 if final_marauders >= 5 else 0.0
                        print(f"第 {episode+1} 場結束 - 總分: {round(total_reward, 2)} - 掠奪者: {final_marauders}")
                        
                        collector.log_episode_summary(
                            Episode=episode + 1, Total_Steps=step_count, Max_Reward=max_reward, 
                            Final_Marauders=final_marauders, Win_Rate=win_rate,
                            Completion_Time=step_count, Avg_Reward_Per_Step=total_reward/step_count
                        )
                        break
                
                agent.q_learn.epsilon = max(0.1, agent.q_learn.epsilon * 0.98)
                
    except KeyboardInterrupt:
        print("⚠️ 訓練被中斷")
    
    print("\n🎉 訓練完成！正在生成報告...")
    collector.export_to_excel()
    collector.generate_plots()
    
    agent.q_learn.q_table.to_csv(collector.q_table_file)

if __name__ == "__main__":
    app.run(main)