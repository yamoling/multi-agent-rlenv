from abc import ABC, abstractmethod
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
        """The shape of an observation."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The environment name"""

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
    def render(self, mode: str="human"):
        """Render the environment"""
