from absl import app
from pysc2.env import sc2_env, run_loop
from pysc2.lib import features, actions

from zerg_enemy96_bot import ZergEnemy96Bot
from dummy_bot import DummyBot


def main(argv):
    agent1 = ZergEnemy96Bot()
    agent2 = DummyBot()

    with sc2_env.SC2Env(
        map_name="Simple96",
        players=[
            sc2_env.Agent(sc2_env.Race.zerg),
            sc2_env.Agent(sc2_env.Race.protoss),
        ],
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=96, minimap=96),
            use_feature_units=True,
            use_raw_units=True,
            action_space=actions.ActionSpace.RAW,
        ),
        step_mul=8,
        visualize=True,
    ) as env:
        run_loop.run_loop([agent1, agent2], env, max_episodes=1)


if __name__ == "__main__":
    app.run(main)