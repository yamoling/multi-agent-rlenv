from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product
from typing import Generic, Optional, Sequence, TypeVar

import cv2
import numpy as np
import numpy.typing as npt

from .observation import Observation
from .spaces import ContinuousSpace, Space, DiscreteSpace, MultiDiscreteSpace
from .state import State
from .step import Step

ActionSpaceType = TypeVar("ActionSpaceType", bound=Space)


@dataclass
class MARLEnv(ABC, Generic[ActionSpaceType]):
    """
    Multi-Agent Reinforcement Learning environment.

    This type is generic on
        - the action type
        - the action space

    You can inherit from this class to create your own environemnt:
    ```
    import numpy as np
    from marlenv import MARLEnv, DiscreteActionSpace, Observation

    N_AGENTS = 3
    N_ACTIONS = 5

    class CustomEnv(MARLEnv[DiscreteActionSpace]):
        def __init__(self, width: int, height: int):
            super().__init__(
                action_space=DiscreteActionSpace(N_AGENTS, N_ACTIONS),
                observation_shape=(height, width),
                state_shape=(1,),
            )
            self.time = 0

        def reset(self) -> Observation:
            self.time = 0
            ...
            return obs

        def get_state(self):
            return np.array([self.time])

        ...
    ```
    """

    action_space: ActionSpaceType
    observation_shape: tuple[int, ...]
    """The shape of an observation for a single agent."""
    extras_shape: tuple[int, ...]
    """The shape of the extras features for a single agent (or the state)"""
    extras_meanings: list[str]
    state_shape: tuple[int, ...]
    """The shape of the state."""
    n_agents: int
    n_actions: int
    name: str
    reward_space: Space

    def __init__(
        self,
        n_agents: int,
        action_space: ActionSpaceType,
        observation_shape: tuple[int, ...],
        state_shape: tuple[int, ...],
        extras_shape: tuple[int, ...] = (0,),
        state_extra_shape: tuple[int, ...] = (0,),
        reward_space: Optional[Space] = None,
        extras_meanings: Optional[list[str]] = None,
    ):
        super().__init__()
        self.name = self.__class__.__name__
        self.action_space = action_space
        self.n_actions = action_space.shape[-1]
        self.n_agents = n_agents
        self.observation_shape = observation_shape
        self.state_shape = state_shape
        self.extras_shape = extras_shape
        self.state_extra_shape = state_extra_shape
        if reward_space is None:
            reward_space = ContinuousSpace.from_shape(1, labels=["Reward"])
        self.reward_space = reward_space
        if extras_meanings is None:
            extras_meanings = [f"{self.name}-extra-{i}" for i in range(extras_shape[0])]
        elif len(extras_meanings) != extras_shape[0]:
            raise ValueError(f"extras_meanings has length {len(extras_meanings)} but expected {extras_shape[0]}")
        self.extras_meanings = extras_meanings
        """The reward space has shape (1, ) for single-objective environments."""
        self._cv2_window_name = None

    @property
    def agent_state_size(self) -> int:
        """The size of the state for a single agent."""
        raise NotImplementedError(f"{self.name} does not support unit_state_size")

    @property
    def is_multi_objective(self) -> bool:
        """Whether the environment is multi-objective."""
        return self.reward_space.size > 1

    @property
    def n_objectives(self) -> int:
        """The number of objectives in the environment."""
        return self.reward_space.size

    def sample_action(self):
        """Sample an available action from the action space."""
        match self.action_space:
            case MultiDiscreteSpace() as aspace:
                return aspace.sample(mask=self.available_actions())
            case ContinuousSpace() as aspace:
                return aspace.sample()
            case DiscreteSpace() as aspace:
                return np.array([aspace.sample(mask=self.available_actions())])
        raise NotImplementedError("Action space not supported")

    def available_actions(self) -> npt.NDArray[np.bool]:
        """
        Get the currently available actions for each agent.

        The output array has shape (n_agents, n_actions) and contains `True` if the action is available and `False` otherwise.
        """
        return np.full((self.n_agents, self.n_actions), True, dtype=np.bool)

    def available_joint_actions(self) -> list[tuple]:
        """Get the possible joint actions."""
        agents_available_actions = [np.nonzero(available)[0] for available in self.available_actions()]
        return list(product(*agents_available_actions))

    def seed(self, seed_value: int):
        """Set the environment seed"""
        return

    @abstractmethod
    def get_observation(self) -> Observation:
        """Retrieve the current observation of the environment."""

    @abstractmethod
    def get_state(self) -> State:
        """Retrieve the current state of the environment."""

    def set_state(self, state: State) -> None:
        """Set the state of the environment."""
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def step(self, action: Sequence | np.ndarray) -> Step:
        """Perform a step in the environment.

        Returns a Step object that can be unpacked as a 6-tuple containing:
        - observations: The observation resulting from the action.
        - state: The new state of the environment.
        - reward: The team reward.
        - done: Whether the episode is over
        - truncated: Whether the episode is truncated
        - info: Extra information
        """

    def random_step(self) -> Step:
        """Perform a random step in the environment."""
        return self.step(self.sample_action())

    @abstractmethod
    def reset(self) -> tuple[Observation, State]:
        """Reset the environment and return the initial observation and state."""

    def render(self):
        """Render the environment in a window (or in console)"""
        img = self.get_image()
        if self._cv2_window_name is None:
            self._cv2_window_name = self.name
            cv2.namedWindow(self._cv2_window_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self._cv2_window_name, img)
        cv2.waitKey(1)

    def get_image(self) -> npt.NDArray[np.uint8]:
        """Retrieve an image of the environment"""
        raise NotImplementedError("No image available for this environment")

    def replay(self, actions: Sequence, seed: Optional[int] = None):
        """Replay a sequence of actions."""
        from .episode import Episode  # Avoid circular import

        if seed is not None:
            self.seed(seed)
        obs, state = self.reset()
        episode = Episode.new(obs, state)
        for action in actions:
            step = self.step(action)
            episode.add(step, action)
        return episode

    def has_same_inouts(self, other: "MARLEnv[ActionSpaceType]") -> bool:
        """Alias for `have_same_inouts(self, other)`."""
        if not isinstance(other, MARLEnv):
            return False
        if self.action_space != other.action_space:
            return False
        if self.observation_shape != other.observation_shape:
            return False
        if self.state_shape != other.state_shape:
            return False
        if self.extras_shape != other.extras_shape:
            return False
        if self.reward_space != other.reward_space:
            return False
        return True

    def __del__(self):
        if not hasattr(self, "_cv2_window_name"):
            return
        if self._cv2_window_name is not None:
            cv2.destroyWindow(self._cv2_window_name)
