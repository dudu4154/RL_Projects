from absl import flags
FLAGS = flags.FLAGS
FLAGS(["run_agent"])   # ✅ 手動 parse flags，避免 UnparsedFlagAccessError

from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions, features, units


# -------------------------
# 1) 1~15 決策函式（先只回傳數字）
# -------------------------
def decide_action_code(my_units: int,
                       enemy_units: int,
                       enemy_has_aoe: bool,
                       under_heavy_fire: bool,
                       enemy_is_ranged: bool) -> int:
    power_ratio = my_units / max(enemy_units, 1)

    if enemy_has_aoe and power_ratio < 1.0:
        return 2      # 分散
    if under_heavy_fire and power_ratio < 0.8:
        return 12     # 全撤
    if enemy_is_ranged and power_ratio < 1.0:
        return 11     # kite
    if power_ratio >= 1.5:
        return 15     # 全面進攻
    if 0.9 <= power_ratio <= 1.1:
        return 3      # 凹形
    if 1.1 < power_ratio < 1.5:
        return 1      # 正推
    if 0.7 <= power_ratio < 0.9:
        return 4      # 拉扯撤退
    return 8          # 防守


# -------------------------
# 2) PySC2 Agent：偵查戰況 & 輸出 code
# -------------------------
class ScoutActionCodeAgent(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        self.prev_my_total_hp = None

        self.AOE_THREATS = {
            units.Terran.SiegeTank,
            units.Terran.SiegeTankSieged,
            units.Protoss.Colossus,
            units.Protoss.Disruptor,
            units.Zerg.Baneling,
            units.Zerg.Infestor,
        }

        self.RANGED_UNITS = {
            units.Terran.Marine,
            units.Terran.Marauder,
            units.Terran.Reaper,
            units.Terran.Cyclone,
            units.Protoss.Stalker,
            units.Protoss.Adept,
            units.Protoss.Immortal,
            units.Zerg.Hydralisk,
            units.Zerg.Roach,
            units.Zerg.Queen,
        }

    def step(self, obs):
        super().step(obs)

        raw_units = obs.observation.raw_units

        me = features.PlayerRelative.SELF
        enemy = features.PlayerRelative.ENEMY

        my_army = []
        enemy_army = []

        for u in raw_units:
            if u.alliance == me:
                # 你也可以只看 Stalker：u.unit_type == units.Protoss.Stalker
                if u.unit_type in (units.Protoss.Stalker, units.Protoss.Zealot,
                                   units.Protoss.Adept, units.Protoss.Immortal):
                    my_army.append(u)
            elif u.alliance == enemy:
                enemy_army.append(u)

        my_units = len(my_army)
        enemy_units = len(enemy_army)

        enemy_has_aoe = any(u.unit_type in self.AOE_THREATS for u in enemy_army)

        ranged_count = sum(1 for u in enemy_army if u.unit_type in self.RANGED_UNITS)
        enemy_is_ranged = (ranged_count / max(enemy_units, 1)) >= 0.6

        my_total_hp = sum((u.health + u.shield) for u in my_army)
        if self.prev_my_total_hp is None:
            under_heavy_fire = False
        else:
            under_heavy_fire = (self.prev_my_total_hp - my_total_hp) >= 15
        self.prev_my_total_hp = my_total_hp

        code = decide_action_code(
            my_units=my_units,
            enemy_units=enemy_units,
            enemy_has_aoe=enemy_has_aoe,
            under_heavy_fire=under_heavy_fire,
            enemy_is_ranged=enemy_is_ranged
        )

        print(code)
        return actions.RAW_FUNCTIONS.no_op()


# -------------------------
# 3) 啟動環境
# -------------------------
def main():
    with sc2_env.SC2Env(
        map_name="Simple64",
        players=[
            sc2_env.Agent(sc2_env.Race.protoss),
            sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.easy)
        ],
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=84, minimap=64),
            use_feature_units=True,
            use_raw_units=True,
            raw_resolution=64,
        ),

        step_mul=8,
        game_steps_per_episode=0,
        visualize=True
    ) as env:
        agent = ScoutActionCodeAgent()
        run_loop.run_loop([agent], env, max_episodes=1)


if __name__ == "__main__":
    main()