from typing import Optional, Sequence
from typing_extensions import TypeVar
from dataclasses import dataclass
import numpy as np

from marlenv.models import MARLEnv, Space, DiscreteSpace, State


AS = TypeVar("AS", bound=Space, default=Space)


@dataclass
class RLEnvWrapper(MARLEnv[AS]):
    """Parent class for all RLEnv wrappers"""

    wrapped: MARLEnv[AS]
    full_name: str
    """The full name of the wrapped environment, excluding the name of the nested wrappers."""

    def __init__(
        self,
        env: MARLEnv[AS],
        *,
        n_agents: Optional[int] = None,
        observation_shape: Optional[tuple[int, ...]] = None,
        state_shape: Optional[tuple[int, ...]] = None,
        extra_shape: Optional[tuple[int, ...]] = None,
        state_extra_shape: Optional[tuple[int, ...]] = None,
        action_space: Optional[AS] = None,
        reward_space: Optional[DiscreteSpace] = None,
        extra_meanings: Optional[list[str]] = None,
    ):
        if extra_meanings is not None:
            if extra_shape is None:
                extra_shape = env.extras_shape
            if len(extra_meanings) != extra_shape[0]:
                raise ValueError(f"There are {len(extra_meanings)} extra_meanings but the announced extra_shape is {extra_shape} !")
        super().__init__(
            n_agents=n_agents or env.n_agents,
            action_space=action_space or env.action_space,
            observation_shape=observation_shape or env.observation_shape,
            state_shape=state_shape or env.state_shape,
            extras_shape=extra_shape or env.extras_shape,
            state_extra_shape=state_extra_shape or env.state_extra_shape,
            reward_space=reward_space or env.reward_space,
            extras_meanings=extra_meanings or env.extras_meanings,
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

    def step(self, actions: np.ndarray | Sequence):
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
