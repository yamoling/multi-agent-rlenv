import random
from typing import TypeVar
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

from marlenv.models import RLEnv, ActionSpace
from .rlenv_wrapper import RLEnvWrapper


A = TypeVar("A", bound=ActionSpace)


@dataclass
class Blind(RLEnvWrapper[A]):
    p: float

    def __init__(self, env: RLEnv[A], p: float | int):
        super().__init__(env)
        self.p = float(p)

    def step(self, actions: npt.NDArray[np.int64,]):
        obs, r, done, trunc, info = super().step(actions)
        if random.random() < self.p:
            obs.data = np.zeros_like(obs.data)
        return obs, r, done, trunc, info
