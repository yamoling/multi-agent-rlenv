from typing import Any, Optional, TypeVar
from abc import abstractmethod, ABC
import numpy as np
import numpy.typing as npt
from serde import serde
from dataclasses import dataclass

ActionType = TypeVar("ActionType")


@serde
@dataclass
class Space(ABC):
    shape: tuple[int, ...]

    @abstractmethod
    def sample(self, mask: Optional[np.ndarray] = None) -> Any:
        """Sample a value from the space."""


@serde
@dataclass
class DiscreteSpace(Space):
    size: int
    """Number of categories"""
    labels: list[str]
    """The label of each category."""

    def __init__(self, size: int, labels: Optional[list[str]] = None):
        super().__init__((size,))
        self.size = size
        if labels is None:
            labels = [f"Label {i}" for i in range(size)]
        self.labels = labels
        self.space = np.arange(size)

    def sample(self, mask: Optional[np.ndarray[bool, Any]] = None) -> int:
        space = self.space
        if mask is not None:
            space = space[mask]
        return np.random.choice(space)


@serde
@dataclass
class MultiDiscreteSpace(Space):
    n_dims: int
    spaces: tuple[DiscreteSpace, ...]

    def __init__(self, *spaces: DiscreteSpace):
        Space.__init__(self, tuple(space.size for space in spaces))
        self.spaces = spaces
        self.n_dims = len(spaces)

    @classmethod
    def from_sizes(cls, *sizes: int):
        return cls(*(DiscreteSpace(size) for size in sizes))

    def sample(self, masks: Optional[np.ndarray[bool, Any] | list[np.ndarray[bool, Any]]] = None):
        if masks is None:
            return np.array([space.sample() for space in self.spaces], dtype=np.int32)
        return np.array([space.sample(mask) for mask, space in zip(masks, self.spaces)], dtype=np.int32)


@serde
@dataclass
class ContinuousSpace(Space):
    """A continuous space (box) in R^n."""

    low: npt.NDArray[np.float32]
    """Lower bound of the space for each dimension."""
    high: npt.NDArray[np.float32]
    """Upper bound of the space for each dimension."""

    def __init__(self, low: list | np.ndarray[np.float32, Any], high: list | np.ndarray[np.float32, Any]):
        if isinstance(low, list):
            low = np.array(low, dtype=np.float32)
        if isinstance(high, list):
            high = np.array(high, dtype=np.float32)
        assert low.shape == high.shape, "Low and high must have the same shape."
        assert np.all(low <= high), "All elements in low must be less than the corresponding elements in high."
        Space.__init__(self, low.shape)
        self.low = low
        self.high = high

    def sample(self):
        return np.random.random(self.shape) * (self.high - self.low) + self.low


@serde
@dataclass
class ActionSpace(Space):
    n_agents: int
    """Number of agents."""
    n_actions: int
    """Number of actions that an agent can perform."""
    action_names: list[str]
    """The meaning of each action."""

    def __init__(self, n_agents: int, n_actions: int, action_names: Optional[list[str]] = None):
        Space.__init__(self, (n_agents, n_actions))
        self.n_agents = n_agents
        self.n_actions = n_actions
        if action_names is None:
            action_names = [f"Action {i}" for i in range(self.n_actions)]
        assert len(action_names) == n_actions
        self.action_names = action_names


class DiscreteActionSpace(ActionSpace, MultiDiscreteSpace):
    def __init__(self, n_agents: int, n_actions: int, action_names: Optional[list[str]] = None):
        ActionSpace.__init__(self, n_agents, n_actions, action_names)
        MultiDiscreteSpace.__init__(
            self,
            *[DiscreteSpace(n_actions, labels=action_names) for _ in range(n_agents)],
        )
        self._actions = [list(range(self.n_actions)) for _ in range(self.n_agents)]

    def sample(self, available_actions: Optional[np.ndarray] = None):
        return MultiDiscreteSpace.sample(self, available_actions)


@serde
@dataclass
class ContinuousActionSpace(ActionSpace, ContinuousSpace):
    def __init__(
        self,
        n_agents: int,
        n_actions: int,
        low: Optional[float | list | npt.NDArray[np.float32]] = None,
        high: Optional[float | list | npt.NDArray[np.float32]] = None,
        action_names: Optional[list[str]] = None,
    ):
        if low is None:
            low = np.zeros((n_actions,), dtype=np.float32)
        elif isinstance(low, list):
            low = np.array(low, dtype=np.float32)
        elif isinstance(low, (float, int)):
            low = np.full((n_actions,), low, dtype=np.float32)
        if high is None:
            high = np.ones((n_actions,), dtype=np.float32)
        elif isinstance(high, list):
            high = np.array(high, dtype=np.float32)
        elif isinstance(high, (float, int)):
            high = np.full((n_actions,), high, dtype=np.float32)
        assert high.shape == (n_actions,)
        assert low.shape == (n_actions,)
        ContinuousSpace.__init__(self, low, high)
        ActionSpace.__init__(self, n_agents, n_actions, action_names)

    def sample(self):
        return ContinuousSpace.sample(self)
