import numpy as np
from typing import TypeVar
from rlenv.models import ActionSpace, Observation
from .rlenv_wrapper import RLEnvWrapper

A = TypeVar("A", bound=ActionSpace)


class AvailableActionsWrapper(RLEnvWrapper[A]):
    @property
    def extra_feature_shape(self):
        return (self.wrapped.extra_feature_shape[0] + self.n_actions,)

    def reset(self):
        obs = self.wrapped.reset()
        obs.extras = np.concatenate([obs.extras, self.get_avail_actions().astype(np.float32)], axis=-1)
        return obs

    def step(self, actions: np.ndarray[np.int32]) -> tuple[Observation, float, bool, dict]:
        obs, *rest = self.wrapped.step(actions)
        obs.extras = np.concatenate([obs.extras, self.get_avail_actions().astype(np.float32)], axis=-1)
        return obs, *rest
