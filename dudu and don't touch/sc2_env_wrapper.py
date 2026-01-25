import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pysc2.env import sc2_env
from production_ai import ProductionAI # 導入你的邏輯類別

class SC2MarauderEnv(gym.Env):
    def __init__(self):
        super(SC2MarauderEnv, self).__init__()
        # 1. 動作空間：0-9 (對應你 get_action 的 action_id)
        self.action_space = spaces.Discrete(10)
        
        # 2. 觀察空間：[礦物, 瓦斯, 工兵數, 掠奪者數, 兵營是否存在, 科技室是否存在]
        self.observation_space = spaces.Box(
            low=0, 
            high=np.array([10000, 10000, 200, 10, 1, 1], dtype=np.float32), 
            dtype=np.float32
        )

        self.agent_logic = ProductionAI()
        self.sc2_env = None
        self.last_marauder_count = 0

    def _get_obs(self, obs):
        player = obs.observation.player
        return np.array([
            float(player.minerals), 
            float(player.vespene), 
            float(player.food_workers), 
            float(self.agent_logic.marauders_produced),
            1.0 if self.agent_logic.barracks_built else 0.0,
            1.0 if self.agent_logic.techlab_built else 0.0
        ], dtype=np.float32)

    def step(self, action_id):
        # 執行動作
        sc2_action = self.agent_logic.get_action(self.last_sc2_obs, action_id)
        obs_list = self.sc2_env.step([sc2_action])
        self.last_sc2_obs = obs_list[0]
        
        new_obs = self._get_obs(self.last_sc2_obs)
        
        # --- 獎勵設計 (Reward Shaping) ---
        reward = 0.0
        # 每多出一隻掠奪者，獎勵 +20
        if self.agent_logic.marauders_produced > self.last_marauder_count:
            reward += 20.0
            self.last_marauder_count = self.agent_logic.marauders_produced
            
        # 達成 5 隻大目標，獎勵 +100
        if self.agent_logic.marauder_production_complete:
            reward += 100.0
        
        # 每一秒鐘扣 0.01，鼓勵 AI 越快完成越好
        reward -= 0.01 
        
        terminated = self.last_sc2_obs.last() or self.agent_logic.marauder_production_complete
        return new_obs, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.sc2_env is None:
            self.sc2_env = sc2_env.SC2Env(
                map_name="Simple64",
                players=[sc2_env.Agent(sc2_env.Race.terran),
                         sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.easy)],
                agent_interface_format=sc2_env.AgentInterfaceFormat(
                    feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64),
                    use_raw_units=False),
                step_mul=16, realtime=False)
        
        obs_list = self.sc2_env.reset()
        self.last_sc2_obs = obs_list[0]
        self.last_marauder_count = 0
        self.agent_logic.marauders_produced = 0
        self.agent_logic.marauder_production_complete = False
        
        return self._get_obs(self.last_sc2_obs), {}

    def close(self):
        if self.sc2_env: self.sc2_env.close()