from typing import Generic, TypeVar
from abc import abstractmethod, ABC
import numpy as np

ActionType = TypeVar("ActionType", bound=np.ndarray)


class ActionSpace(ABC, Generic[ActionType]):
    def __init__(self, n_agents: int, n_actions: int):
        self._n_agents = n_agents
        self._n_actions = n_actions

    @abstractmethod
    def sample(self) -> ActionType:
        """Sample actions from the action space for each agent."""

    @property
    def n_actions(self) -> int:
        """Number of actions that an agent can perform."""
        return self._n_actions
    
    @property
    def n_agents(self) -> int:
        """Number of agents."""
        return self._n_agents



class DiscreteActionSpace(ActionSpace[np.ndarray[np.int32]]):
    def __init__(self, n_agents: int, n_actions: int):
        super().__init__(n_agents, n_actions)
        self._actions = np.array([range(n_actions) for _ in range(n_agents)])

    def sample(self, available_actions: np.ndarray[np.int32] = None) -> np.ndarray[np.int32]:
        if available_actions is None:
            return np.random.randint(0, self.n_actions, self.n_agents)
        action_probs = available_actions / available_actions.sum(axis=1, keepdims=True)
        res = []
        for action, available in zip(self._actions, action_probs):
            res.append(np.random.choice(action, p=available))
        return np.array(res, dtype=np.int32)
        

class ContinuousActionSpace(ActionSpace[np.ndarray[float]]):
    def __init__(
            self, 
            n_agents: int, 
            n_actions: int, 
            low: float|list[float]=0., 
            high: float|list[float]=1.
        ):
        """Continuous action space.
        
        Args:
            n_agents (int): Number of agents.
            n_actions (int): Number of actions per agent.
            low (float|list[float]): Lower bound of the action space. If a float is provided, the same value is used for all actions.
            high (float|list[float]): Upper bound of the action space. If a float is provided, the same value is used for all actions.
        """
        assert isinstance(low, (float)) or len(low) == n_actions, "'low' parameter must be a float or a list of floats with length equal to the number of actions."
        assert isinstance(high, (float)) or len(high) == n_actions, "'high' parameter must be a float or a list of floats with length equal to the number of actions."
        super().__init__(n_agents, n_actions)
        if isinstance(low, float):
            low = [low] * n_actions
        self._low = np.array(low, dtype=np.float32)
        if isinstance(high, float):
            high = [high] * n_actions
        self._high = np.array(high, dtype=np.float32)
        self.shape = (n_agents, n_actions)

    def sample(self) -> np.ndarray[np.float32]:
        return np.random.random(self.shape) * (self._high - self._low) + self._low

