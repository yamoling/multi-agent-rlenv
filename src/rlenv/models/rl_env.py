from abc import ABC, abstractmethod
from typing import Literal
import numpy as np


from .observation import Observation


class RLEnv(ABC):
    """This interface defines the attributes and methods that must be implemented to work with this framework"""

    @property
    def extra_feature_shape(self) -> tuple[int, ...]:
        """The shape of extra features"""
        return (0, )

    def get_avail_actions(self) -> np.ndarray[np.int32]:
        """Get the currently available actions"""
        return np.ones((self.n_agents, self.n_actions), dtype=np.int64)

    def seed(self, seed_value: int):
        """Set the environment seed"""
        raise NotImplementedError("Method not implemented")


    @property
    @abstractmethod
    def n_actions(self) -> int:
        """The number of actions that an agent can take."""

    @property
    @abstractmethod
    def n_agents(self) -> int:
        """The number of agents in the environment."""

    @property
    @abstractmethod
    def state_shape(self) -> tuple[int, ...]:
        """The state size."""

    @property
    @abstractmethod
    def observation_shape(self) -> tuple[int, ...]:
        """The shape of an observation for a single agent."""

    @property
    def name(self) -> str:
        """The environment name"""
        return self.__class__.__name__

    @abstractmethod
    def get_state(self) -> np.ndarray[np.float32]:
        """Retrieve the current state of the environment."""

    @abstractmethod
    def step(self, actions: np.ndarray[np.int32]) -> tuple[Observation, float, bool, dict]:
        """Perform a step in the environment."""

    @abstractmethod
    def reset(self) -> Observation:
        """Reset the environment."""

    @abstractmethod
    def render(self, mode: Literal["human", "rgb_array"]="human") -> None | np.ndarray:
        """
        Render the environment.
        When calling with 'rgb_array', returns the rgb_array but does not show anything on screen.
        """

    def summary(self) -> dict[str, str]:
        """Summary of the environment informations"""
        return {
            "name": self.name,
            "n_actions": int(self.n_actions),
            "n_agents": int(self.n_agents),
            "obs_shape": tuple(int(s) for s in self.observation_shape),
            "extras_shape": tuple(int(s) for s in self.extra_feature_shape),
            "state_shape": tuple(int(s) for s in self.state_shape)
        }