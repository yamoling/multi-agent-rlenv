from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from marlenv.models import ActionSpace
from .rlenv_wrapper import RLEnvWrapper, MARLEnv
# from ..models.rl_env import MOMARLEnv

from typing_extensions import TypeVar

A = TypeVar("A", default=npt.NDArray)
AS = TypeVar("AS", bound=ActionSpace, default=ActionSpace)


@dataclass
class TimePenalty(RLEnvWrapper[A, AS]):
    penalty: float | np.ndarray

    def __init__(self, env: MARLEnv[A, AS], penalty: float | list[float]):
        super().__init__(env)

        if env.is_multi_objective:
            if isinstance(penalty, (float, int)):
                penalty = [float(penalty)] * env.reward_space.size
            self.penalty = np.array(penalty, dtype=np.float32)
        else:
            assert isinstance(penalty, (float, int))
            self.penalty = penalty

    def step(self, action: A):
        step = self.wrapped.step(action)
        step.reward = step.reward - self.penalty
        return step
