from .rlenv_wrapper import RLEnvWrapper, MARLEnv
from marlenv.models import ActionSpace
from typing_extensions import TypeVar
import numpy.typing as npt
import numpy as np
from dataclasses import dataclass
from collections import deque

A = TypeVar("A", default=npt.NDArray)
AS = TypeVar("AS", bound=ActionSpace, default=ActionSpace)


@dataclass
class DelayedReward(RLEnvWrapper[A, AS]):
    delay: int

    def __init__(self, env: MARLEnv[A, AS], delay: int):
        super().__init__(env)
        self.delay = delay
        self.reward_queue = deque[npt.NDArray[np.float32]](maxlen=delay + 1)

    def reset(self):
        self.reward_queue.clear()
        for _ in range(self.delay):
            self.reward_queue.append(np.zeros(self.reward_space.shape, dtype=np.float32))
        return super().reset()

    def step(self, actions: A):
        step = super().step(actions)
        self.reward_queue.append(step.reward)
        # If the step is terminal, we sum all the remaining rewards
        if step.is_terminal:
            step.reward = np.sum(self.reward_queue, axis=0)
        else:
            step.reward = self.reward_queue.popleft()
        return step
