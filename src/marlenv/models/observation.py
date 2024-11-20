from typing import Optional, Sequence, Generic
from typing_extensions import TypeVar
import numpy as np
import numpy.typing as npt

ObsType = TypeVar("ObsType", default=npt.NDArray[np.float32])


class Observation(Generic[ObsType]):
    """
    Container class for policy input arguments.
    """

    data: ObsType
    """The actual environment observation. The shape is [n_agents, *obs_shape]"""
    extras: npt.NDArray[np.float32]
    """The extra information to provide to the dqn alongisde the features (agent ID, last action, ...)"""
    available_actions: npt.NDArray[np.bool_]
    """The available actions at the time of the observation"""
    n_agents: int

    def __init__(
        self,
        data: ObsType,
        available_actions: Sequence[bool] | npt.NDArray[np.bool],
        extras: Optional[npt.NDArray[np.float32]] = None,
    ):
        self.data = data
        if not isinstance(available_actions, np.ndarray):
            available_actions = np.array(available_actions)
        self.available_actions = available_actions
        self.n_agents = len(available_actions)
        if extras is not None:
            self.extras = extras
        else:
            self.extras = np.zeros((self.n_agents, 0), dtype=np.float32)

    def add_extra(self, extra: list[list[float]] | npt.NDArray[np.float32]):
        """Append an extra feature to the observation"""
        self.extras = np.concatenate((self.extras, extra), axis=1)

    @property
    def extras_shape(self) -> tuple[int, ...]:
        """The shape of the observation extras"""
        return self.extras.shape

    def __hash__(self):
        if isinstance(self.data, np.ndarray):
            d = hash(self.data.tobytes())
        else:
            d = hash(self.data)
        return hash((d, self.extras.tobytes()))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        if not isinstance(other, Observation):
            return False
        if isinstance(self.data, np.ndarray):
            if not isinstance(other.data, np.ndarray):
                return False
            if not np.array_equal(self.data, other.data):
                return False
        return np.array_equal(self.extras, other.extras) and np.array_equal(self.available_actions, other.available_actions)
