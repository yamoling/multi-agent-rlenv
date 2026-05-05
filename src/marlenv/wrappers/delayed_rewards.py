from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeVar

from .rlenv_wrapper import MARLEnv, RLEnvWrapper

A = TypeVar("A")


@dataclass
class DelayedReward(RLEnvWrapper[A]):
    delay: int

    def __init__(self, env: MARLEnv[A], delay: int):
        super().__init__(env)
        self.delay = delay
        self._reward_queue = deque[npt.NDArray[np.float32]](maxlen=delay + 1)

    def reset(self, *, seed: int | None = None):
        self._reward_queue.clear()
        for _ in range(self.delay):
            self._reward_queue.append(np.zeros(self.reward_space.shape, dtype=np.float32))
        return super().reset()

    def step(self, action):
        step = super().step(action)
        self._reward_queue.append(step.reward)
        # If the step is terminal, we sum all the remaining rewards
        if step.is_terminal:
            step.reward = np.sum(self._reward_queue, axis=0)
        else:
            step.reward = self._reward_queue.popleft()
        return step
