from typing import TypeVar
import numpy as np

from rlenv.models import ActionSpace, Observation
from .rlenv_wrapper import RLEnvWrapper, RLEnv

A = TypeVar("A", bound=ActionSpace)


class TimeLimit(RLEnvWrapper[A]):
    """
    Limits the number of time steps for an episode. When the number of steps is reached, then the episode is truncated.

    If the `add_extra` flag is set to True, then an extra signal is added to the observation, which is the ratio of the
    current step over the maximum number of steps. In this case, the done flag is also set to True when the maximum
    number of steps is reached.
    """

    def __init__(self, env: RLEnv[A], step_limit: int, add_extra: bool = False) -> None:
        super().__init__(env)
        self._step_limit = step_limit
        self._current_step = 0
        self._add_extra = add_extra

    @property
    def extra_feature_shape(self):
        assert len(self.wrapped.extra_feature_shape) == 1
        if not self._add_extra:
            return self.wrapped.extra_feature_shape
        dims = self.wrapped.extra_feature_shape[0]
        return (dims + 1,)

    def reset(self):
        self._current_step = 0
        obs = super().reset()
        return self.add_time_extra(obs)

    def step(self, actions):
        self._current_step += 1
        obs_, reward, done, truncated, info = super().step(actions)
        if self._add_extra:
            self.add_time_extra(obs_)
        if self._current_step >= self._step_limit:
            truncated = True
            # If we add an extra signal for the time, then we must set
            # the done flag to 1 for consistency:
            # -> since the state actually changes, then it is a terminal state.
            done = self._add_extra
        return obs_, reward, done, truncated, info

    def add_time_extra(self, obs: Observation):
        time_ratio = np.full(
            (self.n_agents, 1),
            self._current_step / self._step_limit,
            dtype=np.float32,
        )
        obs.extras = np.concatenate([obs.extras, time_ratio], axis=-1)
