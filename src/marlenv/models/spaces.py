import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt


@dataclass
class Space(ABC):
    shape: tuple[int, ...]
    size: int
    labels: list[str]

    def __init__(self, shape: tuple[int, ...], labels: Optional[list[str]] = None):
        self.shape = shape
        self.size = math.prod(shape)
        if labels is None:
            labels = [f"Dim {i}" for i in range(self.size)]
        self.labels = labels

    @abstractmethod
    def sample(self, mask: Optional[npt.NDArray[np.bool_]] = None) -> npt.NDArray[np.float32]:
        """Sample a value from the space."""

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Space):
            return False
        return self.shape == value.shape

    def __ne__(self, value: object) -> bool:
        return not self.__eq__(value)

    @property
    @abstractmethod
    def is_discrete(self) -> bool:
        """Whether the space is discrete."""

    @property
    def is_continuous(self) -> bool:
        """Whether the space is continuous."""
        return not self.is_discrete


@dataclass
class DiscreteSpace(Space):
    size: int
    """Number of categories"""

    def __init__(self, size: int, labels: Optional[list[str]] = None):
        super().__init__((size,), labels)
        self.size = size
        self.space = np.arange(size)

    def sample(self, mask: Optional[npt.NDArray[np.bool]] = None):
        space = self.space.copy()
        if mask is not None:
            space = space[mask]
        return int(np.random.choice(space))

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, DiscreteSpace):
            return False
        if self.size != value.size:
            return False
        return super().__eq__(value)

    @property
    def is_discrete(self) -> bool:
        return True

    @staticmethod
    def action(size, labels: Optional[list[str]] = None):
        """
        Create a discrete action space where the default labels are set to "Action-n".
        """
        if labels is None:
            labels = [f"Action {i}" for i in range(size)]
        return DiscreteSpace(size, labels)

    def repeat(self, n: int):
        """
        Repeat the discrete space n times.
        """
        return MultiDiscreteSpace(*([self] * n), labels=self.labels)


@dataclass
class MultiDiscreteSpace(Space):
    n_dims: int
    spaces: tuple[DiscreteSpace, ...]

    def __init__(self, *spaces: DiscreteSpace, labels: Optional[list[str]] = None):
        if labels is None:
            labels = [f"Discrete space {i}" for i in range(len(spaces))]
        Space.__init__(self, tuple(space.size for space in spaces), labels)
        self.spaces = spaces
        self.n_dims = len(spaces)

    @classmethod
    def from_sizes(cls, *sizes: int):
        return cls(*(DiscreteSpace(size) for size in sizes))

    def sample(self, mask: Optional[npt.NDArray[np.bool] | list[npt.NDArray[np.bool]]] = None):
        if mask is None:
            return np.array([space.sample() for space in self.spaces], dtype=np.int32)
        return np.array([space.sample(mask=mask) for mask, space in zip(mask, self.spaces)], dtype=np.int32)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, MultiDiscreteSpace):
            return False
        if len(self.spaces) != len(value.spaces):
            return False
        for s1, s2 in zip(self.spaces, value.spaces):
            if s1.size != s2.size:
                return False
        return super().__eq__(value)

    @property
    def is_discrete(self) -> bool:
        return True


@dataclass
class ContinuousSpace(Space):
    """A continuous space (box) in R^n."""

    low: npt.NDArray[np.float32]
    """Lower bound of the space for each dimension."""
    high: npt.NDArray[np.float32]
    """Upper bound of the space for each dimension."""

    def __init__(
        self,
        low: int | float | list | npt.NDArray[np.float32] | None,
        high: int | float | list | npt.NDArray[np.float32] | None,
        labels: Optional[list[str]] = None,
    ):
        match low:
            case None:
                assert high is not None, "If low is None, high must be set to infer the shape."
                shape = ContinuousSpace.get_shape(high)
                low = np.full(shape, -np.inf, dtype=np.float32)
            case list():
                low = np.array(low, dtype=np.float32)
            case float() | int():
                low = np.array([low], dtype=np.float32)
        match high:
            case None:
                assert low is not None, "If high is None, low must be set to infer the shape."
                shape = ContinuousSpace.get_shape(low)
                high = np.full(shape, np.inf, dtype=np.float32)
            case list():
                high = np.array(high, dtype=np.float32)
            case float() | int():
                high = np.array([high], dtype=np.float32)
        assert low.shape == high.shape, f"Low and high must have the same shape. Low shape: {low.shape}, high shape: {high.shape}"
        assert np.all(low <= high), "All elements in low must be less than the corresponding elements in high."
        Space.__init__(self, low.shape, labels)
        self.low = low
        self.high = high

    @staticmethod
    def from_shape(
        shape: int | tuple[int, ...],
        low: Optional[int | float | list | npt.NDArray[np.float32]] = None,
        high: Optional[int | float | list | npt.NDArray[np.float32]] = None,
        labels: Optional[list[str]] = None,
    ):
        if isinstance(shape, int):
            shape = (shape,)
        match low:
            case None:
                low = np.full(shape, -np.inf, dtype=np.float32)
            case float() | int():
                low = np.full(shape, low, dtype=np.float32)
            case list():
                low = np.array(low, dtype=np.float32)
        match high:
            case None:
                high = np.full(shape, np.inf, dtype=np.float32)
            case float() | int():
                high = np.full(shape, high, dtype=np.float32)
            case list():
                high = np.array(high, dtype=np.float32)
        return ContinuousSpace(low, high, labels)

    def clamp(self, action: np.ndarray | list):
        """Clamp the action to the bounds of the space."""
        if isinstance(action, list):
            action = np.array(action)
        return np.clip(action, self.low, self.high)

    def sample(self) -> npt.NDArray[np.float32]:
        r = np.random.random(self.shape) * (self.high - self.low) + self.low
        return r.astype(np.float32)

    @staticmethod
    def get_shape(item: float | int | list | npt.NDArray[np.float32]) -> tuple[int, ...]:
        """Get the shape of the item."""
        if isinstance(item, list):
            item = np.array(item)
        if isinstance(item, np.ndarray):
            return item.shape
        return (1,)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, ContinuousSpace):
            return False
        if not np.all(self.low == value.low):
            return False
        if not np.all(self.high == value.high):
            return False
        return super().__eq__(value)

    def repeat(self, n: int):
        """
        Repeat the continuous space n times to become of shape (n, *shape).
        """
        low = np.tile(self.low, (n, 1))
        high = np.tile(self.high, (n, 1))
        return ContinuousSpace.from_shape(
            (n, *self.shape),
            low=low,
            high=high,
            labels=self.labels,
        )

    @property
    def is_discrete(self) -> bool:
        return False
