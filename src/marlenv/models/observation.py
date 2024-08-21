from typing import Optional
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


@dataclass
class Observation:
    """
    Container class for policy input arguments.
    """

    data: npt.NDArray[np.float32]
    """The actual environment observation. The shape is [n_agents, *obs_shape]"""
    available_actions: npt.NDArray[np.bool_]
    """The available actions at the time of the observation"""
    state: npt.NDArray[np.float32]
    """The environment state at the time of the observation"""
    extras: npt.NDArray[np.float32]
    """The extra information to provide to the dqn alongisde the features (agent ID, last action, ...)"""

    def __init__(
        self,
        data: npt.NDArray[np.float32],
        available_actions: npt.NDArray[np.bool_],
        state: npt.NDArray[np.float32],
        extras: Optional[npt.NDArray[np.float32]] = None,
    ):
        self.data = data
        self.available_actions = available_actions
        self.state = state
        if extras is not None:
            self.extras = extras
        else:
            self.extras = np.zeros((len(data), 0), dtype=np.float32)

    @property
    def n_agents(self) -> int:
        """The number of agents in the observation"""
        return self.data.shape[0]

    @property
    def data_shape(self) -> tuple[int, ...]:
        """The shape of the observation data"""
        return self.data.shape

    @property
    def extras_shape(self) -> tuple[int, ...]:
        """The shape of the observation extras"""
        return self.extras.shape

    def __hash__(self):
        return hash((self.data.tobytes(), self.state.tobytes(), self.extras.tobytes()))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        if not isinstance(other, Observation):
            return False

        return (
            np.array_equal(self.data, other.data)
            and np.array_equal(self.state, other.state)
            and np.array_equal(self.extras, other.extras)
            and np.array_equal(self.available_actions, other.available_actions)
        )
