from typing import Generic, Optional, TypeVar
from abc import abstractmethod, ABC
import numpy as np
from dataclasses import dataclass

ActionType = TypeVar("ActionType")


@dataclass
class ActionSpace(ABC, Generic[ActionType]):
    n_agents: int
    """Number of agents."""
    n_actions: int
    """Number of actions that an agent can perform."""
    action_names: Optional[list[str]] = None
    """The meaning of each action."""

    def __post_init__(self):
        if self.action_names is None:
            self.action_names = [f"Action {i}" for i in range(self.n_actions)]

    @abstractmethod
    def sample(self) -> ActionType:
        """Sample actions from the action space for each agent."""


@dataclass
class DiscreteActionSpace(ActionSpace[np.ndarray[np.int32]]):
    def __post_init__(self):
        super().__post_init__()
        self._actions = np.array([range(self.n_actions) for _ in range(self.n_agents)])

    def sample(self, available_actions: Optional[np.ndarray[np.int32]] = None) -> np.ndarray[np.int32]:
        if available_actions is None:
            return np.random.randint(0, self.n_actions, self.n_agents)
        action_probs = available_actions / available_actions.sum(axis=1, keepdims=True)
        res = []
        for action, available in zip(self._actions, action_probs):
            res.append(np.random.choice(action, p=available))
        return np.array(res, dtype=np.int32)


@dataclass
class ContinuousActionSpace(ActionSpace[np.ndarray[float]]):
    low: float | list[float] = 0.0
    """Lower bound of the action space. If a float is provided, the same value is used for all actions."""
    high: float | list[float] = 1.0
    """Upper bound of the action space. If a float is provided, the same value is used for all actions."""

    def __post_init__(self):
        assert (
            isinstance(self.low, (float)) or len(self.low) == self.n_actions
        ), "'low' parameter must be a float or a list of floats with length equal to the number of actions."
        assert (
            isinstance(self.high, (float)) or len(self.high) == self.n_actions
        ), "'high' parameter must be a float or a list of floats with length equal to the number of actions."
        super().__post_init__()
        if isinstance(self.low, float):
            self.low = [self.low] * self.n_actions
        self.low = np.array(self.low, dtype=np.float32)
        if isinstance(self.high, float):
            self.high = [self.high] * self.n_actions
        self._high = np.array(self.high, dtype=np.float32)
        self.shape = (int(self.n_agents), int(self.n_actions))

    def sample(self) -> np.ndarray[np.float32]:
        return np.random.random(self.shape) * (self._high - self.low) + self.low
