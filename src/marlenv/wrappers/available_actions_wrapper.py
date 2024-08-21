import numpy as np
import numpy.typing as npt
from typing import TypeVar
from marlenv.models import ActionSpace, RLEnv
from .rlenv_wrapper import RLEnvWrapper

A = TypeVar("A", bound=ActionSpace)


class AvailableActions(RLEnvWrapper[A]):
    """Adds the available actions (one-hot) as an extra feature to the observation."""

    def __init__(self, env: RLEnv[A]):
        super().__init__(env, extra_feature_shape=(env.extra_feature_shape[0] + env.n_actions,))

    def reset(self):
        obs = self.wrapped.reset()
        obs.extras = np.concatenate([obs.extras, self.available_actions().astype(np.float32)], axis=-1)
        return obs

    def step(self, actions: npt.NDArray[np.int32]):
        obs, reward, done, truncated, info = self.wrapped.step(actions)
        obs.extras = np.concatenate([obs.extras, self.available_actions().astype(np.float32)], axis=-1)
        return obs, reward, done, truncated, info
