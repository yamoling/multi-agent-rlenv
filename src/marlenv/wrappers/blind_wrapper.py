import random
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from marlenv.models import MARLEnv

from .rlenv_wrapper import RLEnvWrapper

A = TypeVar("A")


@dataclass
class Blind(RLEnvWrapper[A]):
    p: float

    def __init__(self, env: MARLEnv[A], p: float | int):
        super().__init__(env)
        self.p = float(p)

    def step(self, action):
        step = super().step(action)
        if random.random() < self.p:
            step.obs.data = np.zeros_like(step.obs.data)
        return step

    def get_observation(self):
        obs = super().get_observation()
        if random.random() < self.p:
            obs.data = np.zeros_like(obs.data)
        return obs
