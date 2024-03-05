from typing import Generic, Optional, TypeVar
from abc import abstractmethod, ABC
import numpy as np
import numpy.typing as npt
from serde import serde
from dataclasses import dataclass


ActionType = TypeVar("ActionType")


@serde
@dataclass
class ActionSpace(ABC, Generic[ActionType]):
    n_agents: int
    """Number of agents."""
    n_actions: int
    """Number of actions that an agent can perform."""
    action_names: list[str]
    """The meaning of each action."""

    def __init__(
        self, n_agents: int, n_actions: int, action_names: Optional[list[str]] = None
    ):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.shape = (n_agents, n_actions)
        if action_names is None:
            action_names = [f"Action {i}" for i in range(self.n_actions)]
        self.action_names = action_names

    @abstractmethod
    def sample(self) -> ActionType:
        """Sample actions from the action space for each agent."""


class DiscreteActionSpace(ActionSpace[npt.NDArray[np.int32]]):
    def __init__(
        self, n_agents: int, n_actions: int, action_names: Optional[list[str]] = None
    ):
        super().__init__(n_agents, n_actions, action_names)
        self._actions = [range(self.n_actions) for _ in range(self.n_agents)]

    def sample(self, available_actions: Optional[npt.NDArray[np.int32]] = None):
        if available_actions is None:
            return np.random.randint(0, self.n_actions, self.n_agents)
        action_probs = available_actions / available_actions.sum(axis=1, keepdims=True)
        res = []
        for action, available in zip(self._actions, action_probs):
            res.append(np.random.choice(action, p=available))
        return np.array(res, dtype=np.int32)


@dataclass
class ContinuousActionSpace(ActionSpace[npt.NDArray[np.float32]]):
    low: float | list[float] = 0.0
    """Lower bound of the action space. If a float is provided, the same value is used for all actions."""
    high: float | list[float] = 1.0
    """Upper bound of the action space. If a float is provided, the same value is used for all actions."""

    def __init__(
        self,
        n_agents: int,
        n_actions: int,
        low: float | list[float] = 0.0,
        high: float | list[float] = 1.0,
        action_names: Optional[list[str]] = None,
    ):
        assert isinstance(low, (float)) or (
            isinstance(low, list) and len(low) == n_actions
        ), "'low' parameter must be a float or a list of floats with length equal to the number of actions."
        assert (
            isinstance(high, (float))
            or isinstance(high, list)
            and len(high) == n_actions
        ), "'high' parameter must be a float or a list of floats with length equal to the number of actions."
        super().__init__(n_agents, n_actions, action_names)
        if isinstance(low, float):
            self.low = [low] * n_actions
        self.low = low
        if isinstance(high, float):
            self.high = [high] * self.n_actions
        self.high = high

    def sample(self):
        return (np.random.random(self.shape) * (
            np.array(self.high) - np.array(self.low)
        ) + np.array(self.low)).astype(np.float32)
