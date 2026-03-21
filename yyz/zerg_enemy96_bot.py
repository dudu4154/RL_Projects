# zerg_enemy96_bot.py

from pysc2.agents import base_agent
from pysc2.lib import actions, units

FUNCTIONS = actions.FUNCTIONS


class ZergEnemy96Bot(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        self.stage = 0
        self.attack_launched = False
        self.base_top_left = None

    def step(self, obs):
        super().step(obs)

        if self.base_top_left is None:
            self.base_top_left = self.get_base_top_left(obs)

        # 自動補人口
        if self.supply_left(obs) <= 1 and self.can_do(obs, "Train_Overlord_quick"):
            return FUNCTIONS.Train_Overlord_quick("now")

        # ===== Build Order =====
        # 13 王蟲
        if self.stage == 0:
            if self.food(obs) >= 13 and self.can_do(obs, "Train_Overlord_quick"):
                self.stage = 1
                return FUNCTIONS.Train_Overlord_quick("now")

        # 16 二礦
        elif self.stage == 1:
            if self.food(obs) >= 16 and self.can_do(obs, "Build_Hatchery_screen"):
                self.stage = 2
                return FUNCTIONS.Build_Hatchery_screen("now", self.natural())

        # 17 狗池
        elif self.stage == 2:
            if self.food(obs) >= 17 and self.can_do(obs, "Build_SpawningPool_screen"):
                self.stage = 3
                return FUNCTIONS.Build_SpawningPool_screen("now", self.pool_pos())

        # 18 瓦斯
        elif self.stage == 3:
            if self.food(obs) >= 18:
                gas_action = self.build_gas(obs)
                if gas_action is not None:
                    self.stage = 4
                    return gas_action

        # 19 王蟲
        elif self.stage == 4:
            if self.food(obs) >= 19 and self.can_do(obs, "Train_Overlord_quick"):
                self.stage = 5
                return FUNCTIONS.Train_Overlord_quick("now")

        # 2:00 Queen
        elif self.stage == 5:
            if self.time(obs) >= 120 and self.can_do(obs, "Train_Queen_quick"):
                self.stage = 6
                return FUNCTIONS.Train_Queen_quick("now")

        # 23 小狗
        elif self.stage == 6:
            if self.food(obs) >= 23 and self.can_do(obs, "Train_Zergling_quick"):
                self.stage = 7
                return FUNCTIONS.Train_Zergling_quick("now")

        # 29 王蟲
        elif self.stage == 7:
            if self.food(obs) >= 29 and self.can_do(obs, "Train_Overlord_quick"):
                self.stage = 8
                return FUNCTIONS.Train_Overlord_quick("now")

        # 30 二本
        elif self.stage == 8:
            if self.food(obs) >= 30 and self.can_do(obs, "Morph_Lair_quick"):
                self.stage = 9
                return FUNCTIONS.Morph_Lair_quick("now")

        # 35 蟑螂場
        elif self.stage == 9:
            if self.food(obs) >= 35 and self.can_do(obs, "Build_RoachWarren_screen"):
                self.stage = 10
                return FUNCTIONS.Build_RoachWarren_screen("now", self.roach_warren_pos())

        # 40 王蟲
        elif self.stage == 10:
            if self.food(obs) >= 40 and self.can_do(obs, "Train_Overlord_quick"):
                self.stage = 11
                return FUNCTIONS.Train_Overlord_quick("now")

        # 44 蟑螂速度
        elif self.stage == 11:
            if self.food(obs) >= 44 and self.can_do(obs, "Research_GlialReconstitution_quick"):
                self.stage = 12
                return FUNCTIONS.Research_GlialReconstitution_quick("now")

        # 48 蟑螂
        elif self.stage == 12:
            if self.food(obs) >= 48 and self.can_do(obs, "Train_Roach_quick"):
                self.stage = 13
                return FUNCTIONS.Train_Roach_quick("now")

        # 4:50 出門
        elif self.stage == 13:
            if self.time(obs) >= 290 and not self.attack_launched and self.can_do(obs, "Attack_minimap"):
                self.attack_launched = True
                return self.attack()

            # 出門後繼續補兵
            if self.can_do(obs, "Train_Roach_quick"):
                return FUNCTIONS.Train_Roach_quick("now")
            elif self.can_do(obs, "Train_Zergling_quick"):
                return FUNCTIONS.Train_Zergling_quick("now")

        return FUNCTIONS.no_op()

    # =========================
    # 基本資訊
    # =========================
    def food(self, obs):
        return obs.observation.player.food_used

    def supply_left(self, obs):
        p = obs.observation.player
        return p.food_cap - p.food_used

    def time(self, obs):
        return obs.observation.game_loop[0] / 22.4

    # =========================
    # feature_units
    # =========================
    def get_units(self, obs):
        if hasattr(obs.observation, "feature_units"):
            return obs.observation.feature_units
        return []

    def get_units_by_type(self, obs, unit_type):
        return [u for u in self.get_units(obs) if u.unit_type == unit_type]

    def get_base_top_left(self, obs):
        hatch = self.get_units_by_type(obs, units.Zerg.Hatchery)
        if not hatch:
            return True
        return hatch[0].x < 48

    # =========================
    # action 可用性
    # =========================
    def can_do(self, obs, action_name):
        action_id = getattr(FUNCTIONS, action_name).id
        return action_id in obs.observation.available_actions

    # =========================
    # 96x96 座標
    # =========================
    def mirror_pos(self, x, y):
        if self.base_top_left:
            return (x, y)
        return (96 - x, 96 - y)

    def pool_pos(self):
        return self.mirror_pos(22, 22)

    def roach_warren_pos(self):
        return self.mirror_pos(28, 22)

    def natural(self):
        return (24, 36) if self.base_top_left else (72, 60)

    def attack(self):
        target = (80, 80) if self.base_top_left else (15, 15)
        return FUNCTIONS.Attack_minimap("now", target)

    # =========================
    # 建氣礦
    # =========================
    def build_gas(self, obs):
        if not self.can_do(obs, "Build_Extractor_screen"):
            return None

        geysers = [
            u for u in self.get_units(obs)
            if u.unit_type in (
                units.Neutral.VespeneGeyser,
                units.Neutral.RichVespeneGeyser
            )
        ]

        if geysers:
            g = geysers[0]
            return FUNCTIONS.Build_Extractor_screen("now", (int(g.x), int(g.y)))

        return None