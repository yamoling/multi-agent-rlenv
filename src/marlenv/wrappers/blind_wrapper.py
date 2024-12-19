import random
from typing_extensions import TypeVar
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

from marlenv.models import MARLEnv, ActionSpace
from .rlenv_wrapper import RLEnvWrapper


A = TypeVar("A", default=npt.NDArray)
AS = TypeVar("AS", bound=ActionSpace, default=ActionSpace)


@dataclass
class Blind(RLEnvWrapper[A, AS]):
    p: float

    def __init__(self, env: MARLEnv[A, AS], p: float | int):
        super().__init__(env)
        self.p = float(p)

    def step(self, actions: A):
        step = super().step(actions)
        if random.random() < self.p:
            step.obs.data = np.zeros_like(step.obs.data)
        return step
