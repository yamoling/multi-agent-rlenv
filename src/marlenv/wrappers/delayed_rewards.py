from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeVar

from marlenv.models import Space

from .rlenv_wrapper import MARLEnv, RLEnvWrapper

AS = TypeVar("AS", bound=Space, default=Space)


@dataclass
class DelayedReward(RLEnvWrapper[AS]):
    delay: int

    def __init__(self, env: MARLEnv[AS], delay: int):
        super().__init__(env)
        self.delay = delay
        self.reward_queue = deque[npt.NDArray[np.float32]](maxlen=delay + 1)

    def reset(self):
        self.reward_queue.clear()
        for _ in range(self.delay):
            self.reward_queue.append(np.zeros(self.reward_space.shape, dtype=np.float32))
        return super().reset()

    def step(self, actions):
        step = super().step(actions)
        self.reward_queue.append(step.reward)
        # If the step is terminal, we sum all the remaining rewards
        if step.is_terminal:
            step.reward = np.sum(self.reward_queue, axis=0)
        else:
            step.reward = self.reward_queue.popleft()
        return step
