from abc import ABC, abstractmethod
from typing import Generic, TypeVar, overload, Any, Literal, Optional
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass


from .spaces import ActionSpace, DiscreteSpace
from .observation import Observation

A = TypeVar("A", bound=ActionSpace)


@dataclass
class RLEnv(ABC, Generic[A]):
    """This interface defines the attributes and methods that must be implemented to work with this framework"""

    action_space: A
    observation_shape: tuple[int, ...]
    """The shape of an observation for a single agent."""
    state_shape: tuple[int, ...]
    """The shape of the state."""
    extra_feature_shape: tuple[int, ...]
    reward_space: DiscreteSpace
    """Desription of the reward space. In general, this is a single scalar, but it can be multi-objective."""
    n_agents: int
    n_actions: int
    name: str

    def __init__(
        self,
        action_space: A,
        observation_shape: tuple[int, ...],
        state_shape: tuple[int, ...],
        extra_feature_shape: tuple[int, ...] = (0,),
        reward_space: Optional[DiscreteSpace] = None,
    ):
        super().__init__()
        self.name = self.__class__.__name__
        self.action_space = action_space
        self.n_actions = action_space.n_actions
        self.n_agents = action_space.n_agents
        self.observation_shape = observation_shape
        self.state_shape = state_shape
        self.extra_feature_shape = extra_feature_shape
        self.reward_space = reward_space or DiscreteSpace(1, ["default"])

    @property
    def agent_state_size(self) -> int:
        """The size of the state for a single agent."""
        raise NotImplementedError(f"{self.name} does not support unit_state_size")

    @property
    def reward_size(self) -> int:
        """The size of the reward signal. In general, this is 1, but it can be higher for multi-objective environments."""
        return self.reward_space.size

    def available_actions(self) -> npt.NDArray[np.bool_]:
        """
        Get the currently available actions for each agent.

        The output array has shape (n_agents, n_actions) and contains 1 if the action is available and 0 otherwise.
        """
        return np.full((self.n_agents, self.n_actions), True, dtype=bool)

    def seed(self, seed_value: int):
        """Set the environment seed"""
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def get_state(self) -> npt.NDArray[np.float32]:
        """Retrieve the current state of the environment."""

    @abstractmethod
    def step(self, actions: npt.ArrayLike) -> tuple[Observation, npt.NDArray[np.float32], bool, bool, dict[str, Any]]:
        """Perform a step in the environment.

        Returns:
        - observations: The observation resulting from the action.
        - rewards: The team reward (single item list in general, but can be multi-objective).
        - done: Whether the episode is over
        - truncated: Whether the episode is truncated
        - info: Extra information
        """

    @abstractmethod
    def reset(self) -> Observation:
        """Reset the environment."""

    @overload
    @abstractmethod
    def render(self, mode: Literal["human"]) -> None:
        """Render the environment in a window"""

    @overload
    @abstractmethod
    def render(self, mode: Literal["rgb_array"]) -> npt.NDArray[np.uint8]:
        """Retrieve an image of the environment"""

    @abstractmethod
    def render(self, mode) -> None | npt.NDArray[np.uint8]: ...

    @staticmethod
    def assert_same_inouts(env1: "RLEnv", env2: "RLEnv") -> None:
        """
        Raise a `ValueError` if the inputs and output spaces of the environments are different.
        """
        if env1.action_space != env2.action_space:
            raise ValueError(f"Action spaces are different: {env1.action_space} != {env2.action_space}")
        if env1.observation_shape != env2.observation_shape:
            raise ValueError(f"Observation shapes are different: {env1.observation_shape} != {env2.observation_shape}")
        if env1.state_shape != env2.state_shape:
            raise ValueError(f"State shapes are different: {env1.state_shape} != {env2.state_shape}")
        if env1.extra_feature_shape != env2.extra_feature_shape:
            raise ValueError(f"Extra feature shapes are different: {env1.extra_feature_shape} != {env2.extra_feature_shape}")
        if env1.reward_space != env2.reward_space:
            raise ValueError(f"Reward spaces are different: {env1.reward_space} != {env2.reward_space}")

    def has_same_inouts(self, other: "RLEnv") -> bool:
        """Alias for `have_same_inouts(self, other)`."""
        try:
            RLEnv.assert_same_inouts(self, other)
            return True
        except ValueError:
            return False

    @staticmethod
    def have_same_inouts(env1: "RLEnv", env2: "RLEnv") -> bool:
        """Check if two environments have the same input and output spaces."""
        try:
            RLEnv.assert_same_inouts(env1, env2)
            return True
        except ValueError:
            return False
