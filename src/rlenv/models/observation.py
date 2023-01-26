from dataclasses import dataclass
import numpy as np


@dataclass
class Observation:
    """
    Container class for policy input arguments.
    """
    data: np.ndarray[np.float32]
    """The actual environment observation. The shape is [n_agents, *obs_shape]"""
    available_actions: np.ndarray[np.int32]
    """The available actions at the time of the observation"""
    state: np.ndarray[np.float32]
    """The environment state at the time of the observation"""
    extras: np.ndarray[np.float32]
    """The extra information to provide to the dqn alongisde the features (agent ID, last action, ...)"""

    def __init__(
        self,
        data: np.ndarray[np.float32],
        available_actions: np.ndarray[np.int32],
        state: np.ndarray[np.float32],
        extras: np.ndarray[np.float32] | None=None
    ) -> None:
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
