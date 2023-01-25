from abc import ABC
from rlenv.models import RLEnv


class RLEnvWrapper(RLEnv, ABC):
    """Parent class for all RLEnv wrappers"""
    def __init__(self, env: RLEnv) -> None:
        super().__init__()
        self.env = env

    @property
    def n_actions(self):
        return self.env.n_actions

    @property
    def n_agents(self):
        return self.env.n_agents

    @property
    def state_shape(self):
        return self.env.state_shape

    @property
    def observation_shape(self):
        return self.env.observation_shape

    @property
    def name(self):
        return f"{self.__class__.__name__}({self.env.name})"

    def step(self, actions):
        return self.env.step(actions)

    def reset(self):
        return self.env.reset()

    def get_state(self):
        return self.env.get_state()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def render(self, mode: str = "human"):
        return self.env.render(mode)

    def seed(self, seed_value: int):
        return self.env.seed(seed_value)
