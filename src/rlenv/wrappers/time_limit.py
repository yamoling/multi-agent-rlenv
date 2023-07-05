from typing import TypeVar
import numpy as np

from rlenv.models import ActionSpace, Observation
from .rlenv_wrapper import RLEnvWrapper, RLEnv

A = TypeVar("A", bound=ActionSpace)


class TimeLimitWrapper(RLEnvWrapper[A]):
    def __init__(self, env: RLEnv[A], step_limit: int, add_extra: bool = False) -> None:
        super().__init__(env)
        self._step_limit = step_limit
        self._current_step = 0
        self._add_extra = add_extra

    @property
    def extra_feature_shape(self):
        assert len(self.wrapped.extra_feature_shape) == 1
        dims = self.wrapped.extra_feature_shape[0]
        if self._add_extra:
            return (dims + 1,)
        return dims

    def reset(self):
        self._current_step = 0
        obs = super().reset()
        return self.add_time_extra(obs)

    def step(self, actions):
        self._current_step += 1
        obs_, reward, done, truncated, info = super().step(actions)
        obs_ = self.add_time_extra(obs_)
        truncated = truncated or (self._current_step >= self._step_limit)
        return obs_, reward, done, truncated, info

    def add_time_extra(self, obs: Observation) -> Observation:
        if self._add_extra:
            time_ratio = np.full((self.n_agents, 1), self._current_step / self._step_limit, dtype=np.float32)
            obs.extras = np.concatenate([obs.extras, time_ratio], axis=-1)
        return obs

    def kwargs(self) -> dict[str,]:
        return {"step_limit": self._step_limit}
