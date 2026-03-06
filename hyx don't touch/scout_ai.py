#可以照路線走，不會卡在礦區了
import math
from absl import app
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

import math
from absl import app
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

class ScoutAI:
    def __init__(self):
        self.scout_tag = None  
        # 擴展到 22 ~ 60 的大範圍路徑點
        self.targets = [
            (32, 32), (45, 32), (60, 32), 
            (60, 45), (60, 60), (45, 60), 
            (32, 60), (22, 60), (22, 45), 
            (22, 32), (22, 22), (32, 22), (45, 22), (60, 22)
        ]
        self.target_index = 0
        self.direction = 1  
        self.missing_count = 0 

    def step(self, obs):
        all_units = obs.observation.raw_units
        my_scvs = [u for u in all_units 
                   if u.unit_type == units.Terran.SCV and u.alliance == features.PlayerRelative.SELF]
        
        if not my_scvs: return actions.RAW_FUNCTIONS.no_op()

        # 1. 英雄換代：如果原本的死了，鎖定新英雄並【重置進度】
        if self.scout_tag is None:
            self.scout_tag = my_scvs[0].tag
            self.target_index = 0  # 核心修正：新英雄必須從站點 0 (基地) 重新出發
            self.direction = 1     # 核心修正：重置為正向巡邏
            self.missing_count = 0
            print(f"✨ 委派新英雄: {self.scout_tag}，從基地重新開始巡邏。")

        scout = next((u for u in my_scvs if u.tag == self.scout_tag), None)

        # 2. 英雄消失判定
        if scout is None:
            self.missing_count += 1
            if self.missing_count > 30: # 消失約 1.5 秒即判定陣亡
                self.scout_tag = None
            return actions.RAW_FUNCTIONS.no_op()
        
        self.missing_count = 0
        
        # 3. 往返巡邏邏輯
        target = self.targets[self.target_index]
        dist = math.sqrt((scout.x - target[0])**2 + (scout.y - target[1])**2)

        # 判定抵達 (範圍大，判定稍微放寬到 4.0 避免卡死)
        if dist < 4.0:
            print(f"✅ 抵達站點 {self.target_index}: {target}")
            
            # 檢查是否到達路徑端點
            if self.target_index == len(self.targets) - 1:
                self.direction = -1
                print("🔄 到達 60 邊界終點，開始原路折返...")
            elif self.target_index == 0 and self.direction == -1:
                self.direction = 1
                print("🔄 回到基地起始點，重新出發...")

            # 根據方向移動索引
            self.target_index += self.direction
            # 確保索引不越界
            self.target_index = max(0, min(len(self.targets) - 1, self.target_index))
            
            new_target = self.targets[self.target_index]
            return actions.RAW_FUNCTIONS.Move_pt("now", self.scout_tag, new_target)

        # 4. 每 10 幀重複發送指令，對抗 SCV 想回去採礦的本能
        if obs.observation.game_loop % 10 == 0:
            return actions.RAW_FUNCTIONS.Move_pt("now", self.scout_tag, target)
        
        return actions.RAW_FUNCTIONS.no_op()

def main(argv):
    agent_interface = features.AgentInterfaceFormat(
        feature_dimensions=features.Dimensions(screen=84, minimap=64),
        use_raw_units=True,
        use_raw_actions=True
    )
    
    scout_agent = ScoutAI()

    with sc2_env.SC2Env(
        map_name="Simple64",
        players=[sc2_env.Agent(sc2_env.Race.terran), 
                 sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy)],
        agent_interface_format=agent_interface,
        step_mul=8,
        visualize=True,
        game_steps_per_episode=0
    ) as env:
        timesteps = env.reset()
        while True:
            obs = timesteps[0]
            if obs.last(): break
            action = scout_agent.step(obs)
            timesteps = env.step([action])

if __name__ == "__main__":
    app.run(main)
# -------------------------------------------------
# 主程式
# -------------------------------------------------
from absl import app
from pysc2.env import sc2_env
from pysc2.lib import features


def main(argv):
    # 1. 定義介面
    agent_interface = features.AgentInterfaceFormat(
        feature_dimensions=features.Dimensions(screen=64, minimap=64),
        use_feature_units=True,
        use_raw_units=True  # 務必加上這行，否則你的 ScoutAI 會抓不到單位
    )

    # 2. 實例化你的 AI
    scout_agent = ScoutAI()

    # 3. 開啟環境
    with sc2_env.SC2Env(
        map_name="Simple64",
        players=[
            sc2_env.Agent(sc2_env.Race.terran),
            sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy)
        ],
        agent_interface_format=agent_interface,
        step_mul=8,
        visualize=True
    ) as env:

        print("SC2 已成功開啟")
        timesteps = env.reset()
        
        # --- 重點：while 迴圈必須在 with 裡面 ---
        while True:
            step_actions = []
            
            # 取得當前觀測值 (第一個玩家的 obs)
            obs = timesteps[0]
            
            if obs.last():
                print("遊戲結束")
                break
            
            # 呼叫你的 ScoutAI 邏輯
            # 注意：scout_agent.step 會回傳 Move_pt 或 no_op
            scout_action = scout_agent.step(obs)
            step_actions.append(scout_action)

            # 執行動作
            timesteps = env.step(step_actions)

if __name__ == "__main__":
    app.run(main)