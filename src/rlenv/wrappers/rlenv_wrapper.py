from typing import TypeVar, Literal, overload, Optional, Any
from dataclasses import dataclass
from serde import serde
from abc import ABC
import numpy as np
from rlenv.models import RLEnv, ActionSpace

A = TypeVar("A", bound=ActionSpace)


@serde
@dataclass
class RLEnvWrapper(RLEnv[A], ABC):
    """Parent class for all RLEnv wrappers"""

    wrapped: RLEnv[A]
    full_name: str
    """The full name of the wrapped environment, excluding the name of the nested wrappers."""

    def __init__(
        self,
        env: RLEnv[A],
        observation_shape: Optional[tuple[int, ...]] = None,
        state_shape: Optional[tuple[int, ...]] = None,
        extra_feature_shape: Optional[tuple[int, ...]] = None,
        action_space: Optional[A] = None,
    ):
        if observation_shape is None:
            observation_shape = env.observation_shape
        if state_shape is None:
            state_shape = env.state_shape
        if extra_feature_shape is None:
            extra_feature_shape = env.extra_feature_shape
        if action_space is None:
            action_space = env.action_space
        super().__init__(action_space, observation_shape, state_shape, extra_feature_shape)
        self.wrapped = env
        if isinstance(env, RLEnvWrapper):
            self.full_name = f"{self.__class__.__name__}({env.full_name})"
        else:
            self.full_name = f"{self.__class__.__name__}({env.name})"
        self.name = env.name

    def step(self, actions: list[int] | np.ndarray):
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
    def render(self, mode: Literal["rgb_array"]) -> np.ndarray[np.uint8, Any]:
        """Retrieve an image of the environment"""

    def render(self, mode):
        return self.wrapped.render(mode)

    def seed(self, seed_value: int):
        return self.wrapped.seed(seed_value)
