import numpy as np
import numpy.typing as npt
from typing import TypeVar
from marlenv.models import ActionSpace, MARLEnv
from .rlenv_wrapper import RLEnvWrapper

A = TypeVar("A", bound=ActionSpace)
D = TypeVar("D")
S = TypeVar("S")
R = TypeVar("R", bound=float | npt.NDArray[np.float32])


class AvailableActions(RLEnvWrapper[A, D, S, R]):
    """Adds the available actions (one-hot) as an extra feature to the observation."""

    def __init__(self, env: MARLEnv[A, D, S, R]):
        super().__init__(env, extra_shape=(env.extra_shape[0] + env.n_actions,))

    def reset(self):
        obs, state = self.wrapped.reset()
        obs.add_extra(self.available_actions().astype(np.float32))
        return obs, state

    def step(self, actions: npt.NDArray[np.int32]):
        step = self.wrapped.step(actions)
        step.obs.add_extra(self.available_actions().astype(np.float32))
        return step
