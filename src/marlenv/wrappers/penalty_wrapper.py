from dataclasses import dataclass
import numpy as np
from marlenv.models import Space
from .rlenv_wrapper import RLEnvWrapper, MARLEnv

from typing_extensions import TypeVar

AS = TypeVar("AS", bound=Space, default=Space)


@dataclass
class TimePenalty(RLEnvWrapper[AS]):
    penalty: float | np.ndarray

    def __init__(self, env: MARLEnv[AS], penalty: float | list[float]):
        super().__init__(env)

        if env.is_multi_objective:
            if isinstance(penalty, (float, int)):
                penalty = [float(penalty)] * env.reward_space.size
            self.penalty = np.array(penalty, dtype=np.float32)
        else:
            assert isinstance(penalty, (float, int))
            self.penalty = penalty

    def step(self, action):
        step = self.wrapped.step(action)
        step.reward = step.reward - self.penalty
        return step
