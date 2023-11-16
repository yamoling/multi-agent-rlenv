import numpy as np
import numpy.typing as npt
from .rlenv_wrapper import RLEnvWrapper, RLEnv


class TimePenaltyWrapper(RLEnvWrapper):
    def __init__(self, env: RLEnv, penalty: float) -> None:
        super().__init__(env)
        self.penalty = penalty

    def step(self, action: npt.NDArray[np.int32]) -> tuple:
        obs, reward, *rest = self.wrapped.step(action)
        reward -= self.penalty
        return obs, reward, *rest
