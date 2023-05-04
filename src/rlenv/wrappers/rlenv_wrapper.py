from typing import TypeVar
from abc import ABC
import numpy as np
from rlenv.models import RLEnv, Observation, ActionSpace

A = TypeVar("A", bound=ActionSpace)


class RLEnvWrapper(RLEnv[A], ABC):
    """Parent class for all RLEnv wrappers"""

    def __init__(self, env: RLEnv[A]) -> None:
        super().__init__(env.action_space)
        self.wrapped = env

    @property
    def state_shape(self):
        return self.wrapped.state_shape

    @property
    def observation_shape(self):
        return self.wrapped.observation_shape

    @property
    def extra_feature_shape(self):
        return self.wrapped.extra_feature_shape

    @property
    def name(self):
        return self.wrapped.name

    def kwargs(self) -> dict[str,]:
        return {}

    def step(self, actions: np.ndarray[np.int32]) -> tuple[Observation, float, bool, dict]:
        return self.wrapped.step(actions)

    def reset(self):
        return self.wrapped.reset()

    def get_state(self):
        return self.wrapped.get_state()

    def get_avail_actions(self):
        return self.wrapped.get_avail_actions()

    def render(self, mode: str = "human"):
        return self.wrapped.render(mode)

    def seed(self, seed_value: int):
        return self.wrapped.seed(seed_value)

    def summary(self) -> dict[str, str]:
        # Get the env summary, and add the wrapper to the wrappers list + the wrapper's kwargs
        summary = self.wrapped.summary()
        wrappers = summary.get("wrappers", [])
        wrappers.append(self.__class__.__name__)
        return {
            **summary,
            "n_actions": self.n_actions,
            "n_agents": self.n_agents,
            "obs_shape": self.observation_shape,
            "extras_shape": self.extra_feature_shape,
            "state_shape": self.state_shape,
            "wrappers": wrappers,
            self.__class__.__name__: self.kwargs(),
        }

    @classmethod
    def from_summary(cls, env: RLEnv, summary: dict[str,]) -> "RLEnvWrapper":
        kwargs = summary.pop(cls.__name__)
        return cls(env, **kwargs)
