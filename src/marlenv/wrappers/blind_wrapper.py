import random
from typing import TypeVar
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

from marlenv.models import MARLEnv, ActionSpace
from .rlenv_wrapper import RLEnvWrapper


A = TypeVar("A", bound=ActionSpace)
S = TypeVar("S")
R = TypeVar("R", bound=float | npt.NDArray)


@dataclass
class Blind(RLEnvWrapper[A, npt.NDArray, S, R]):
    p: float

    def __init__(self, env: MARLEnv[A, npt.NDArray, S, R], p: float | int):
        super().__init__(env)
        self.p = float(p)

    def step(self, actions: npt.NDArray[np.int64,]):
        step = super().step(actions)
        if random.random() < self.p:
            step.obs.data = np.zeros_like(step.obs.data)
        return step
