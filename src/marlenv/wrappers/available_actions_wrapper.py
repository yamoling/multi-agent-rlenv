import numpy as np
import numpy.typing as npt
from typing_extensions import TypeVar
from marlenv.models import ActionSpace, MARLEnv
from .rlenv_wrapper import RLEnvWrapper
from dataclasses import dataclass


A = TypeVar("A", default=npt.NDArray)
AS = TypeVar("AS", bound=ActionSpace, default=ActionSpace)


@dataclass
class AvailableActions(RLEnvWrapper[A, AS]):
    """Adds the available actions (one-hot) as an extra feature to the observation."""

    def __init__(self, env: MARLEnv[A, AS]):
        super().__init__(env, extra_shape=(env.extra_shape[0] + env.n_actions,))

    def reset(self):
        obs, state = self.wrapped.reset()
        obs.add_extra(self.available_actions().astype(np.float32))
        return obs, state

    def step(self, actions: A):
        step = self.wrapped.step(actions)
        step.obs.add_extra(self.available_actions().astype(np.float32))
        return step
