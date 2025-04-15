import random
from typing_extensions import TypeVar
import numpy as np
from dataclasses import dataclass

from marlenv.models import MARLEnv, Space
from .rlenv_wrapper import RLEnvWrapper


AS = TypeVar("AS", bound=Space, default=Space)


@dataclass
class Blind(RLEnvWrapper[AS]):
    p: float

    def __init__(self, env: MARLEnv[AS], p: float | int):
        super().__init__(env)
        self.p = float(p)

    def step(self, actions):
        step = super().step(actions)
        if random.random() < self.p:
            step.obs.data = np.zeros_like(step.obs.data)
        return step

    def get_observation(self):
        obs = super().get_observation()
        if random.random() < self.p:
            obs.data = np.zeros_like(obs.data)
        return obs
