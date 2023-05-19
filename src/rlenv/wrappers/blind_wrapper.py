import random
from typing import TypeVar
import numpy as np

from rlenv.models import Observation, RLEnv, ActionSpace
from .rlenv_wrapper import RLEnvWrapper, RLEnv


A = TypeVar("A", bound=ActionSpace)


class BlindWrapper(RLEnvWrapper[A]):
    def __init__(self, env: RLEnv[A], p: float) -> None:
        super().__init__(env)
        self.p = p

    def step(self, actions: np.ndarray[np.int32,]) -> tuple[Observation, float, bool, bool, dict]:
        obs, r, done, trunc, info = super().step(actions)
        if random.random() < self.p:
            obs.data = np.zeros_like(obs.data)
        return obs, r, done, trunc, info
