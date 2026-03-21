# run_enemy_bot.py

from absl import app
from pysc2.env import sc2_env, run_loop
from pysc2.lib import features
from zerg_enemy96_bot import ZergEnemy96Bot


def main(argv):
    agent = ZergEnemy96Bot()

    with sc2_env.SC2Env(
        map_name="Simple96",   # 這裡一定要改成你實際有的 96x96 地圖名稱
        players=[
            sc2_env.Agent(sc2_env.Race.zerg),
            sc2_env.Bot(sc2_env.Race.protoss, sc2_env.Difficulty.easy)
        ],
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=96, minimap=96),
            use_feature_units=True
        ),
        step_mul=8,
        visualize=True
    ) as env:
        run_loop.run_loop([agent], env, max_episodes=1)


if __name__ == "__main__":
    app.run(main)