from pysc2.agents import base_agent
from pysc2.lib import actions

FUNCTIONS = actions.FUNCTIONS


class DummyBot(base_agent.BaseAgent):
    def step(self, obs):
        super().step(obs)
        return FUNCTIONS.no_op()
    