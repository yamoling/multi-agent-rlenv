from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from .rlenv_wrapper import MARLEnv, RLEnvWrapper

A = TypeVar("A")


@dataclass
class TimePenalty(RLEnvWrapper[A]):
    penalty: float | np.ndarray

    def __init__(self, env: MARLEnv[A], penalty: float | list[float]):
        super().__init__(env)

        if env.is_multi_objective:
            if isinstance(penalty, (float, int)):
                penalty = [float(penalty)] * env.reward_space.size
            self.penalty = np.array(penalty, dtype=np.float32)
        else:
            assert isinstance(penalty, (float, int))
            self.penalty = penalty

    def step(self, action):
        step = self.wrapped.step(action)
        step.reward = step.reward - self.penalty
        return step
