from typing import Optional, Generic, Sequence, TypeVar
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


D = TypeVar("D")
S = TypeVar("S")


@dataclass
class Observation(Generic[D, S]):
    """
    Container class for policy input arguments.
    """

    data: D
    """The actual environment observation. The shape is [n_agents, *obs_shape]"""
    available_actions: npt.NDArray[np.bool_]
    """The available actions at the time of the observation"""
    state: S
    """The environment state at the time of the observation"""
    extras: npt.NDArray[np.float32]
    """The extra information to provide to the dqn alongisde the features (agent ID, last action, ...)"""
    n_agents: int

    def __init__(
        self,
        data: D,
        available_actions: Sequence[bool] | np.ndarray,
        state: S,
        extras: Optional[npt.NDArray[np.float32]] = None,
    ):
        self.data = data
        if not isinstance(available_actions, np.ndarray):
            available_actions = np.array(available_actions)
        self.available_actions = available_actions
        self.state = state
        self.n_agents = len(available_actions)
        if extras is not None:
            self.extras = extras
        else:
            self.extras = np.zeros((self.n_agents, 0), dtype=np.float32)

    @property
    def extras_shape(self) -> tuple[int, ...]:
        """The shape of the observation extras"""
        return self.extras.shape

    def __hash__(self):
        if isinstance(self.data, np.ndarray):
            d = hash(self.data.tobytes())
        else:
            d = hash(self.data)
        if isinstance(self.state, np.ndarray):
            s = hash(self.state.tobytes())
        else:
            s = hash(self.state)
        return hash((d, s, self.extras.tobytes()))

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
        if isinstance(self.state, np.ndarray):
            if not isinstance(other.state, np.ndarray):
                return False
            if not np.array_equal(self.state, other.state):
                return False
        return np.array_equal(self.extras, other.extras) and np.array_equal(self.available_actions, other.available_actions)
