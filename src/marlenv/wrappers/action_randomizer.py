from .rlenv_wrapper import RLEnvWrapper, AS, MARLEnv
import numpy as np


class ActionRandomizer(RLEnvWrapper[AS]):
    def __init__(self, env: MARLEnv[AS], p: float):
        super().__init__(env)
        self.p = p

    def step(self, action):
        if np.random.rand() < self.p:
            action = self.action_space.sample()
        return super().step(action)

    def seed(self, seed_value: int):
        np.random.seed(seed_value)
        super().seed(seed_value)
