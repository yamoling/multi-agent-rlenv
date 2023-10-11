from typing import TypeVar, Literal, overload
from abc import ABC
import numpy as np
from rlenv.models import RLEnv, Observation, ActionSpace
from dataclasses import dataclass

A = TypeVar("A", bound=ActionSpace)


@dataclass
class RLEnvWrapper(RLEnv[A], ABC):
    """Parent class for all RLEnv wrappers"""

    wrapped: RLEnv[A]

    def __init__(self, env: RLEnv[A]):
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

    def step(self, actions: np.ndarray[np.int32]) -> tuple[Observation, float, bool, bool, dict]:
        return self.wrapped.step(actions)

    def reset(self):
        return self.wrapped.reset()

    def get_state(self):
        return self.wrapped.get_state()

    def available_actions(self):
        return self.wrapped.available_actions()

    @overload
    def render(self, mode: Literal["human"]) -> None:
        """Render the environment in a window"""

    @overload
    def render(self, mode: Literal["rgb_array"]) -> np.ndarray:
        """Retrieve an image of the environment"""

    def render(self, mode):
        return self.wrapped.render(mode)

    @property
    def action_meanings(self) -> list[str]:
        return self.wrapped.action_meanings

    def seed(self, seed_value: int):
        return self.wrapped.seed(seed_value)
