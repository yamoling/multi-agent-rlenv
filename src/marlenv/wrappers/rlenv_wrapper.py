from typing import Optional
from typing_extensions import TypeVar
from dataclasses import dataclass
from abc import ABC
import numpy.typing as npt
from marlenv.models import MARLEnv, ActionSpace, DiscreteSpace, State


A = TypeVar("A", default=npt.NDArray)
AS = TypeVar("AS", bound=ActionSpace, default=ActionSpace)


@dataclass
class RLEnvWrapper(MARLEnv[A, AS], ABC):
    """Parent class for all RLEnv wrappers"""

    wrapped: MARLEnv[A, AS]
    full_name: str
    """The full name of the wrapped environment, excluding the name of the nested wrappers."""

    def __init__(
        self,
        env: MARLEnv[A, AS],
        observation_shape: Optional[tuple[int, ...]] = None,
        state_shape: Optional[tuple[int, ...]] = None,
        extra_shape: Optional[tuple[int, ...]] = None,
        state_extra_shape: Optional[tuple[int, ...]] = None,
        action_space: Optional[AS] = None,
        reward_space: Optional[DiscreteSpace] = None,
    ):
        super().__init__(
            action_space=action_space or env.action_space,
            observation_shape=observation_shape or env.observation_shape,
            state_shape=state_shape or env.state_shape,
            extra_shape=extra_shape or env.extra_shape,
            state_extra_shape=state_extra_shape or env.state_extra_shape,
            reward_space=reward_space or env.reward_space,
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

    def get_observation(self):
        return self.wrapped.get_observation()

    @property
    def agent_state_size(self):
        return self.wrapped.agent_state_size

    def step(self, actions: A):
        return self.wrapped.step(actions)

    def reset(self):
        return self.wrapped.reset()

    def get_state(self):
        return self.wrapped.get_state()

    def set_state(self, state: State):
        return self.wrapped.set_state(state)

    def available_actions(self):
        return self.wrapped.available_actions()

    def render(self):
        return self.wrapped.render()

    def get_image(self):
        return self.wrapped.get_image()

    def seed(self, seed_value: int):
        return self.wrapped.seed(seed_value)
