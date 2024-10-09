from typing import TypeVar, Literal, overload, Optional
from dataclasses import dataclass
from abc import ABC
import numpy as np
import numpy.typing as npt
from marlenv.models import MARLEnv, ActionSpace


A = TypeVar("A", bound=ActionSpace)
D = TypeVar("D")
S = TypeVar("S")
R = TypeVar("R", bound=float | npt.NDArray[np.float32])


@dataclass
class RLEnvWrapper(MARLEnv[A, D, S, R], ABC):
    """Parent class for all RLEnv wrappers"""

    wrapped: MARLEnv[A, D, S, R]
    full_name: str
    """The full name of the wrapped environment, excluding the name of the nested wrappers."""

    def __init__(
        self,
        env: MARLEnv[A, D, S, R],
        observation_shape: Optional[tuple[int, ...]] = None,
        state_shape: Optional[tuple[int, ...]] = None,
        extra_feature_shape: Optional[tuple[int, ...]] = None,
        action_space: Optional[A] = None,
    ):
        super().__init__(
            action_space=action_space or env.action_space,
            observation_shape=observation_shape or env.observation_shape,
            state_shape=state_shape or env.state_shape,
            extra_feature_shape=extra_feature_shape or env.extra_feature_shape,
        )
        self.wrapped = env
        if isinstance(env, RLEnvWrapper):
            self.full_name = f"{self.__class__.__name__}({env.full_name})"
            self.unwrapped = env.unwrapped
            """The base environment that is wrapped by all the nested wrappers."""
        else:
            self.full_name = f"{self.__class__.__name__}({env.name})"
            self.unwrapped = env
        self.name = env.name

    @property
    def agent_state_size(self):
        return self.wrapped.agent_state_size

    def step(self, actions):
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
    def render(self, mode: Literal["rgb_array"]) -> npt.NDArray[np.uint8]:
        """Retrieve an image of the environment"""

    def render(self, mode):
        return self.wrapped.render(mode)

    def seed(self, seed_value: int):
        return self.wrapped.seed(seed_value)
