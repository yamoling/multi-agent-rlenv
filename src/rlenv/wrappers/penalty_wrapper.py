from dataclasses import dataclass
from serde import serde
import numpy as np
import numpy.typing as npt
from .rlenv_wrapper import RLEnvWrapper, RLEnv


@serde
@dataclass
class TimePenalty(RLEnvWrapper):
    penalty: float

    def __init__(self, env: RLEnv, penalty: float):
        super().__init__(env)
        self.penalty = penalty

    def step(self, action: npt.NDArray[np.int64]):
        obs, reward, *rest = self.wrapped.step(action)
        reward = [r - self.penalty for r in reward]
        return obs, reward, *rest
