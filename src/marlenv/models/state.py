from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Literal, Optional, overload

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeVar

if TYPE_CHECKING:
    from torch import Tensor  # pyright: ignore[reportMissingImports]

StateType = TypeVar("StateType", default=npt.NDArray[np.float32])


@dataclass
class State(Generic[StateType]):
    data: StateType
    extras: npt.NDArray[np.float32]

    def __init__(self, data: StateType, extras: Optional[npt.NDArray[np.float32]] = None):
        self.data = data
        if extras is None:
            extras = np.empty(0, dtype=np.float32)
        self.extras = extras

    def add_extra(self, extra: int | float | npt.NDArray[np.float32]):
        if isinstance(extra, (float, int)):
            extra = np.array([extra], dtype=np.float32)
        self.extras = np.concatenate((self.extras, extra))

    @property
    def shape(self) -> tuple[int, ...]:
        if isinstance(self.data, np.ndarray):
            return self.data.shape
        raise ValueError("State data is not a numpy array")

    @property
    def extras_shape(self) -> tuple[int, ...]:
        return self.extras.shape

    def __hash__(self) -> int:
        if isinstance(self.data, np.ndarray):
            d = hash(self.data.tobytes())
        else:
            d = hash(self.data)
        return hash((d, self.extras.tobytes()))

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, State):
            return False
        if isinstance(self.data, np.ndarray):
            if not isinstance(value.data, np.ndarray):
                return False
            if not np.array_equal(self.data, value.data):
                return False
        if not np.array_equal(self.extras, value.extras):
            return False
        return True

    @overload
    def as_tensors(self, device=None, *, batch_dim: Literal[True]) -> "tuple[Tensor, Tensor]":
        """Convert the state to a tuple of tensors of shape `(1, *self.shape)`."""

    @overload
    def as_tensors(self, device=None, *, batch_dim: bool = False) -> "tuple[Tensor, Tensor]":
        """Convert the state to a tuple of tensors of shape `self.shape`."""

    def as_tensors(self, device=None, *, batch_dim=False):
        import torch  # pyright: ignore[reportMissingImports]

        data = torch.from_numpy(self.data).to(device, non_blocking=True)
        extras = torch.from_numpy(self.extras).to(device, non_blocking=True)
        if batch_dim:
            data = data.unsqueeze(0)
            extras = extras.unsqueeze(0)
        return data, extras
