from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt


@dataclass
class Observation:
    """
    Container class for policy input arguments.
    """

    data: npt.NDArray[np.float32]
    """The actual environment observation. The shape is [n_agents, *obs_shape]"""
    extras: npt.NDArray[np.float32]
    """The extra information to provide to the dqn alongisde the features (agent ID, last action, ...)"""
    available_actions: npt.NDArray[np.bool_]
    """The available actions at the time of the observation"""
    n_agents: int

    def __init__(
        self,
        data: npt.NDArray[np.float32],
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

    def agent(self, agent_id: int, keep_dim: bool = True) -> "Observation":
        """
        Return the observation of the given agent.

        If `keep_dim` is True, the resulting shape is [1, *obs_shape].
        Otherwise, the resulting shape is [*obs_shape].
        """
        if keep_dim:
            return Observation(
                data=self.data[agent_id : agent_id + 1],
                extras=self.extras[agent_id : agent_id + 1],
                available_actions=self.available_actions[agent_id : agent_id + 1],
            )
        return Observation(
            data=self.data[agent_id],
            extras=self.extras[agent_id],
            available_actions=self.available_actions[agent_id],
        )

    @property
    def extras_shape(self) -> tuple[int, ...]:
        """The shape of the observation extras"""
        return self.extras[0].shape

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
