from abc import ABC, abstractmethod
from typing import Generic, Optional, overload, Any, Literal
from typing_extensions import TypeVar
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass


from .spaces import ActionSpace, DiscreteSpace
from .observation import Observation

A = TypeVar("A", bound=ActionSpace)
D = TypeVar("D")
S = TypeVar("S")
R = TypeVar("R", bound=float | npt.NDArray[np.float32], default=float)


@dataclass
class MARLEnv(ABC, Generic[A, D, S, R]):
    """
    Multi-Agent Reinforcement Learning environment.

    This type is generic on
        - A: the action space
        - D: the observation data type
        - S: the state data type
        - R: the reward data type (default is float)
    """

    action_space: A
    observation_shape: tuple[int, ...]
    """The shape of an observation for a single agent."""
    state_shape: tuple[int, ...]
    """The shape of the state."""
    extra_feature_shape: tuple[int, ...]
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
        self.reward_space = reward_space or DiscreteSpace(1, labels=["Reward"])
        """The reward space has shape (1, ) for single-objective environments."""
        print(R)

    @property
    def agent_state_size(self) -> int:
        """The size of the state for a single agent."""
        raise NotImplementedError(f"{self.name} does not support unit_state_size")

    @property
    def is_multi_objective(self) -> bool:
        """Whether the environment is multi-objective."""
        return self.reward_space.size > 1

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
    def step(self, actions) -> tuple[Observation[D, S], R, bool, bool, dict[str, Any]]:
        """Perform a step in the environment.

        Returns:
        - observations: The observation resulting from the action.
        - reward: The team reward.
        - done: Whether the episode is over
        - truncated: Whether the episode is truncated
        - info: Extra information
        """

    @abstractmethod
    def reset(self) -> Observation[D, S]:
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
    def assert_same_inouts(env1: "MARLEnv", env2: "MARLEnv") -> None:
        """
        Raise a `ValueError` if the inputs and output spaces of the environments are different.
        """

    def has_same_inouts(self, other) -> bool:
        """Alias for `have_same_inouts(self, other)`."""
        if not isinstance(other, MARLEnv):
            return False
        if self.action_space != other.action_space:
            return False
        if self.observation_shape != other.observation_shape:
            return False
        if self.state_shape != other.state_shape:
            return False
        if self.extra_feature_shape != other.extra_feature_shape:
            return False
        if self.reward_space != other.reward_space:
            return False
        return True

    @staticmethod
    def have_same_inouts(env1: "MARLEnv", env2: "MARLEnv") -> bool:
        """Check if two environments have the same input and output spaces."""
        try:
            MARLEnv.assert_same_inouts(env1, env2)
            return True
        except ValueError:
            return False


# @dataclass
# class MOMARLEnv(MARLEnv[A, D, S, npt.NDArray[np.float32]], ABC):
#     """Multi-Objective Multi-Agent Reinforcement Learning environment."""

#     reward_space: DiscreteSpace
#     """Desription of the reward space. In general, this is a single scalar, but it can be multi-objective."""

#     def __init__(
#         self,
#         action_space: A,
#         observation_shape: tuple[int, ...],
#         state_shape: tuple[int, ...],
#         reward_space: DiscreteSpace,
#         extra_feature_shape: tuple[int, ...] = (0,),
#     ):
#         super().__init__(action_space, observation_shape, state_shape, extra_feature_shape)
#         self.reward_space = reward_space

#     @property
#     def reward_size(self) -> int:
#         """The size of the reward signal. In general, this is 1, but it can be higher for multi-objective environments."""
#         return self.reward_space.size

#     @abstractmethod
#     def step(self, actions) -> tuple[Observation[D, S], npt.NDArray[np.float32], bool, bool, dict[str, Any]]:
#         """Perform a step in the environment.

#         Returns:
#         - observations: The observation resulting from the action.
#         - rewards: The 1D-array of rewards (one per objective).
#         - done: Whether the episode is over
#         - truncated: Whether the episode is truncated
#         - info: Extra information
#         """

#     def has_same_inouts(self, other) -> bool:
#         if not isinstance(other, MOMARLEnv):
#             return False
#         if self.reward_space != other.reward_space:
#             return False
#         return super().has_same_inouts(other)

#     @staticmethod
#     def assert_same_inouts(env1: "MOMARLEnv", env2: MARLEnv) -> None:
#         if not isinstance(env2, MOMARLEnv):
#             raise ValueError(f"Environment 2 is not a MultiObjectiveRLEnv: {env2}")
#         if env1.reward_space != env2.reward_space:
#             raise ValueError(f"Reward spaces are different: {env1.reward_space} != {env2.reward_space}")
#         return MARLEnv.assert_same_inouts(env1, env2)
