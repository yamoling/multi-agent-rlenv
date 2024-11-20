from abc import ABC, abstractmethod
from typing import Generic, Optional, Sequence, overload, Literal
from typing_extensions import TypeVar
import cv2
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from itertools import product


from .step import Step
from .state import State
from .spaces import ActionSpace, DiscreteSpace, DiscreteActionSpace, ContinuousActionSpace
from .observation import Observation

ActionSpaceType = TypeVar("ActionSpaceType", bound=ActionSpace, default=ActionSpace)
ObsType = TypeVar("ObsType", default=npt.NDArray[np.float32])
StateType = TypeVar("StateType", default=npt.NDArray[np.float32])
RewardType = TypeVar("RewardType", bound=float | npt.NDArray[np.float32], default=float)


@dataclass
class MARLEnv(ABC, Generic[ActionSpaceType, ObsType, StateType, RewardType]):
    """
    Multi-Agent Reinforcement Learning environment.

    This type is generic on
        - A: the action space
        - D: the observation data type
        - S: the state data type
        - R: the reward data type (default is float)
    """

    action_space: ActionSpaceType
    observation_shape: tuple[int, ...]
    """The shape of an observation for a single agent."""
    extra_shape: tuple[int, ...]
    """The shape of the extras features for a single agent (or the state)"""
    state_shape: tuple[int, ...]
    """The shape of the state."""
    n_agents: int
    n_actions: int
    name: str

    def __init__(
        self,
        action_space: ActionSpaceType,
        observation_shape: tuple[int, ...],
        state_shape: tuple[int, ...],
        extra_shape: tuple[int, ...] = (0,),
        state_extra_shape: tuple[int, ...] = (0,),
        reward_space: Optional[DiscreteSpace] = None,
    ):
        super().__init__()
        self.name = self.__class__.__name__
        self.action_space = action_space
        self.n_actions = action_space.n_actions
        self.n_agents = action_space.n_agents
        self.observation_shape = observation_shape
        self.state_shape = state_shape
        self.extra_shape = extra_shape
        self.state_extra_shape = state_extra_shape
        self.reward_space = reward_space or DiscreteSpace(1, labels=["Reward"])
        """The reward space has shape (1, ) for single-objective environments."""
        self.cv2_window_name = None

    @property
    def agent_state_size(self) -> int:
        """The size of the state for a single agent."""
        raise NotImplementedError(f"{self.name} does not support unit_state_size")

    @property
    def is_multi_objective(self) -> bool:
        """Whether the environment is multi-objective."""
        return self.reward_space.size > 1

    def available_actions(self) -> npt.NDArray[np.bool]:
        """
        Get the currently available actions for each agent.

        The output array has shape (n_agents, n_actions) and contains 1 if the action is available and 0 otherwise.
        """
        return np.full((self.n_agents, self.n_actions), True, dtype=np.bool)

    def available_joint_actions(self) -> list[tuple]:
        """Get the possible joint actions."""
        agents_available_actions = [np.nonzero(available)[0] for available in self.available_actions()]
        return list(product(*agents_available_actions))

    def seed(self, seed_value: int):
        """Set the environment seed"""
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def get_observation(self) -> Observation[ObsType]:
        """Retrieve the current observation of the environment."""

    @abstractmethod
    def get_state(self) -> State[StateType]:
        """Retrieve the current state of the environment."""

    def set_state(self, state: State[StateType]) -> None:
        """Set the state of the environment."""
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def step(self, actions: Sequence[int] | Sequence[float] | npt.NDArray) -> Step[ObsType, StateType, RewardType]:
        """Perform a step in the environment.

        Returns a Step object that can be unpacked as a 6-tuple containing:
        - observations: The observation resulting from the action.
        - state: The new state of the environment.
        - reward: The team reward.
        - done: Whether the episode is over
        - truncated: Whether the episode is truncated
        - info: Extra information
        """

    @abstractmethod
    def reset(self) -> tuple[Observation[ObsType], State[StateType]]:
        """Reset the environment."""

    def render(self):
        """Render the environment in a window (or in console)"""
        img = self.get_image()
        if self.cv2_window_name is None:
            self.cv2_window_name = self.name
            cv2.namedWindow(self.cv2_window_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.cv2_window_name, img)

    def get_image(self) -> npt.NDArray[np.uint8]:
        """Retrieve an image of the environment"""
        raise NotImplementedError("No image available for this environment")

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
        if self.extra_shape != other.extra_shape:
            return False
        if self.reward_space != other.reward_space:
            return False
        return True

    def __del__(self):
        if self.cv2_window_name is not None:
            cv2.destroyWindow(self.cv2_window_name)


class DiscreteMARLEnv(MARLEnv[DiscreteActionSpace, ObsType, StateType, RewardType]):
    def step(self, actions: Sequence[int] | npt.NDArray[np.int32]):
        return super().step(actions)


class ContinuousMARLEnv(MARLEnv[ContinuousActionSpace, ObsType, StateType, RewardType]):
    def step(self, actions: Sequence[float] | npt.NDArray[np.float32]):
        return super().step(actions)
