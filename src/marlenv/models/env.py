import math
from abc import ABC, abstractmethod
from dataclasses import KW_ONLY, dataclass, field
from itertools import product
from typing import Callable, Generic, Sequence, TypeVar

import cv2
import numpy as np
import numpy.typing as npt

from .episode import Episode
from .observation import Observation
from .spaces import ContinuousSpace, Space
from .state import State
from .step import Step

A = TypeVar("A")


@dataclass
class MARLEnv(ABC, Generic[A]):
    """
    Multi-agent reinforcement learning environment interface.

    You can inherit from this class to create your own environment:
    ```
    import numpy as np
    from marlenv import MARLEnv, DiscreteSpace, MultiDiscreteSpace, Observation, State, Step

    N_AGENTS = 3
    N_ACTIONS = 5

    class CustomEnv(MARLEnv[npt.NDArry[np.int64]]):
        def __init__(self, width: int, height: int):
            super().__init__(
                n_agents=N_AGENTS,
                action_space=DiscreteSpace.action(N_ACTIONS).repeat(N_AGENTS),
                observation_shape=(height, width),
                state_shape=(1,),
            )
            self.time = 0

        def reset(self) -> tuple[Observation, State]:
            self.time = 0
            return self.get_observation(), self.get_state()

        def step(self, action) -> Step:
            self.time += 1
            return Step(self.get_observation(), self.get_state(), reward=0.0, done=False)

        def get_state(self) -> State:
            return State(np.array([self.time], dtype=np.float32))

        def get_observation(self) -> Observation:
            return Observation(
                np.zeros((N_AGENTS, height, width), dtype=np.float32),
                self.available_actions(),
            )
    ```
    """

    n_agents: int
    action_space: Space[A]
    observation_shape: tuple[int, ...]
    """The shape of an observation for a single agent."""
    state_shape: tuple[int, ...]
    _: KW_ONLY
    extras_shape: tuple[int, ...] = ()
    """The shape of the extras features for a single agent"""
    state_extra_shape: tuple[int, ...] = (0,)
    """The shape of the state."""
    reward_space: Space[npt.NDArray[np.float32]] = field(default_factory=lambda: ContinuousSpace.from_shape(1, labels=["Reward"]))
    extras_meanings: list[str] = field(default_factory=list)

    def __post_init__(self):
        if len(self.extras_meanings) == 0:
            self.extras_meanings = [f"{self.name}-extra-{i}" for i in range(self.extras_size)]
        if len(self.extras_meanings) != self.extras_size:
            raise ValueError("There should either be no extra meaning provided, or all of then should be provided.")
        self._cv2_window_name = None

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def n_actions(self):
        return self.action_space.shape[-1]

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

    @property
    def extras_size(self) -> int:
        """The size of the flattened extras features for a single agent."""
        return math.prod(self.extras_shape)

    @property
    def state_extras_size(self) -> int:
        return math.prod(self.state_extra_shape)

    @property
    def observation_size(self) -> int:
        """The size of a flattened observation for a single agent."""
        return math.prod(self.observation_shape)

    @property
    def state_size(self) -> int:
        """The size of a flattened state."""
        return math.prod(self.state_shape)

    def sample_action(self):
        """Sample an available action from the action space."""
        return self.action_space.sample(self.available_actions())

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
    def step(self, action: A | npt.ArrayLike) -> Step:
        """Perform a step in the environment.

        Returns a Step object that can be unpacked as a 6-tuple containing:
        - observations: The observation resulting from the action.
        - state: The new state of the environment.
        - reward: The team reward.
        - done: Whether the episode has reached a terminal state
        - truncated: Whether the episode is truncated (not a terminal state)
        - info: Extra information
        """

    def random_step(self) -> Step:
        """Perform a random step in the environment."""
        return self.step(self.sample_action())

    @abstractmethod
    def reset(self, *, seed: int | None = None) -> tuple[Observation, State]:
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

    def replay(self, actions: Sequence, seed: int | None = None):
        """Replay a sequence of actions."""
        from .episode import Episode  # Avoid circular import

        if seed is not None:
            self.seed(seed)
        obs, state = self.reset()
        episode = Episode.new(obs, state)
        for action in actions:
            step = self.step(action)
            episode.add(step)
        return episode

    def rollout(self, agent: Callable[[Observation], A]):
        obs, state = self.reset()
        episode = Episode.new(obs, state)
        action = agent(obs)
        step = self.step(action)
        while not step.is_terminal:
            episode.add(step)
            action = agent(step.obs)
            step = self.step(action)
        episode.add(step)
        return episode

    def has_same_inouts(self, other: "MARLEnv[A]") -> bool:
        """
        Returns whether the environment has the same input and output shapes as another environment, which includes:
            - action space
            - observation shape
            - state shape
            - extras shape
            - reward space
        """
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


DiscreteMARLEnv = MARLEnv[npt.NDArray[np.int64]]
ContinuousMARLEnv = MARLEnv[npt.NDArray[np.float32]]
