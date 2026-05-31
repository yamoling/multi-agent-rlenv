from dataclasses import dataclass, field

import numpy.typing as npt
from typing_extensions import TypeVar

from marlenv.models import ContinuousSpace, MARLEnv, Space, State

A = TypeVar("A")


@dataclass
class RLEnvWrapper(MARLEnv[A]):
    """Parent class for all RLEnv wrappers"""

    wrapped: MARLEnv[A]
    full_name: str = field(init=False)
    """The full name of the wrapped environment, excluding the name of the nested wrappers."""

    def __init__(
        self,
        wrapped: MARLEnv[A],
        *,
        n_agents: int | None = None,
        observation_shape: tuple[int, ...] | None = None,
        state_shape: tuple[int, ...] | None = None,
        extra_shape: tuple[int, ...] | None = None,
        state_extras_shape: tuple[int, ...] | None = None,
        action_space: Space[A] | None = None,
        reward_space: ContinuousSpace | None = None,
        extra_meanings: list[str] | None = None,
    ):
        if extra_meanings is not None:
            if extra_shape is None:
                extra_shape = wrapped.extras_shape
            if len(extra_meanings) != extra_shape[0]:
                raise ValueError(f"There are {len(extra_meanings)} extra_meanings but the announced extra_shape is {extra_shape} !")
        super().__init__(
            n_agents=n_agents or wrapped.n_agents,
            action_space=action_space or wrapped.action_space,
            observation_shape=observation_shape or wrapped.observation_shape,
            state_shape=state_shape or wrapped.state_shape,
            extras_shape=extra_shape or wrapped.extras_shape,
            state_extras_shape=state_extras_shape or wrapped.state_extras_shape,
            reward_space=reward_space or wrapped.reward_space,
            extras_meanings=extra_meanings or wrapped.extras_meanings,
        )
        self.wrapped = wrapped
        if isinstance(wrapped, RLEnvWrapper):
            self.full_name = f"{self.__class__.__name__}({wrapped.full_name})"
            self.unwrapped = wrapped.unwrapped
            """The base environment that is wrapped by all the nested wrappers."""
        else:
            self.full_name = f"{self.__class__.__name__}({wrapped.name})"
            self.unwrapped = wrapped

    def get_observation(self):
        return self.wrapped.get_observation()

    @property
    def name(self):
        return self.wrapped.name

    @property
    def agent_state_size(self):
        return self.wrapped.agent_state_size

    def step(self, action: A | npt.ArrayLike):
        return self.wrapped.step(action)

    def reset(self, *, seed: int | None = None):
        return self.wrapped.reset(seed=seed)

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
