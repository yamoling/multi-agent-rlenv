from abc import ABC, abstractmethod
from typing import Union, Generic, TypeVar, overload, Any, Literal
import numpy as np
from serde import serde
from dataclasses import dataclass


from .spaces import ActionSpace
from .observation import Observation

A = TypeVar("A", bound=ActionSpace)


class StepData(tuple[Observation, float, bool, bool, dict[str, Any]]):
    """A tuple containing the data returned by the step method of an environment"""

    def __new__(cls, observation: Observation, reward: float, done: bool, truncated: bool, info: dict[str, Any]):
        return super().__new__(cls, (observation, reward, done, truncated, info))

    @property
    def observation(self) -> Observation:
        return self[0]

    @property
    def reward(self) -> float:
        return self[1]

    @property
    def done(self) -> bool:
        return self[2]

    @property
    def truncated(self) -> bool:
        return self[3]

    @property
    def info(self) -> dict[str, Any]:
        return self[4]


@serde
@dataclass
class RLEnv(ABC, Generic[A]):
    """This interface defines the attributes and methods that must be implemented to work with this framework"""

    action_space: A
    n_agents: int
    n_actions: int
    name: str

    def __init__(self, action_space: A):
        super().__init__()
        self.name = self.__class__.__name__
        self.action_space = action_space
        self.n_actions = action_space.n_actions
        self.n_agents = action_space.n_agents

    def available_actions(self) -> np.ndarray[np.int32]:
        """
        Get the currently available actions for each agent.

        The output array has shape (n_agents, n_actions) and contains 1 if the action is available and 0 otherwise.
        """
        return np.ones((self.n_agents, self.n_actions), dtype=np.int64)

    def seed(self, seed_value: int):
        """Set the environment seed"""
        raise NotImplementedError("Method not implemented")

    @property
    def extra_feature_shape(self) -> tuple[int, ...]:
        """The shape of the extra features."""
        return (0,)

    @property
    @abstractmethod
    def state_shape(self) -> tuple[int, ...]:
        """The state size."""

    @property
    @abstractmethod
    def observation_shape(self) -> tuple[int, ...]:
        """The shape of an observation for a single agent."""

    @abstractmethod
    def get_state(self) -> np.ndarray[np.float32]:
        """Retrieve the current state of the environment."""

    @abstractmethod
    def step(self, actions: np.ndarray[np.int32]) -> Union[StepData, tuple[Observation, float, bool, bool, dict[str, Any]]]:
        """Perform a step in the environment.

        Returns:
        - observations: The observations for each agent.
        - rewards: The team reward
        - done: Whether the episode is over
        - truncated: Whether the episode is truncated
        - info: Extra information
        """

    @abstractmethod
    def reset(self) -> Observation:
        """Reset the environment."""

    @overload
    def render(self, mode: Literal["human"]) -> None:
        """Render the environment in a window"""

    @overload
    def render(self, mode: Literal["rgb_array"]) -> np.ndarray[np.uint8]:
        """Retrieve an image of the environment"""

    @abstractmethod
    def render(self, mode):
        ...
