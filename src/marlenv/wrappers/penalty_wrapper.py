from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from marlenv.models import ActionSpace
from .rlenv_wrapper import RLEnvWrapper, MARLEnv
# from ..models.rl_env import MOMARLEnv

from typing import TypeVar

A = TypeVar("A", bound=ActionSpace)
D = TypeVar("D")
S = TypeVar("S")
R = TypeVar("R", bound=float | np.ndarray)


@dataclass
class TimePenalty(RLEnvWrapper[A, D, S, R]):
    penalty: float | np.ndarray

    def __init__(self, env: MARLEnv[A, D, S, R], penalty: float | list[float]):
        super().__init__(env)

        if env.is_multi_objective:
            if isinstance(penalty, float):
                penalty = [penalty] * env.reward_space.size
            self.penalty = np.array(penalty, dtype=np.float32)
        else:
            assert isinstance(penalty, float)
            self.penalty = penalty

    def step(self, action: npt.NDArray[np.int64]):
        step = self.wrapped.step(action)
        return step.with_attrs(reward=step.reward - self.penalty)  # type: ignore
