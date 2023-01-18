from dataclasses import dataclass
from typing import Optional
import numpy as np
import numpy.typing as npt


@dataclass
class Observation:
    """
    Container class for policy input arguments.
    """
    data: npt.NDArray[np.float32]
    """The actual environment observation. The shape is [n_agents, *obs_shape]"""
    available_actions: npt.NDArray[np.int32]
    """The available actions at the time of the observation"""
    state: npt.NDArray[np.float32]
    """The environment state at the time of the observation"""
    extras: npt.NDArray[np.float32]
    """The extra information to provide to the dqn alongisde the features (agent ID, last action, ...)"""

    def __init__(
        self,
        data: npt.NDArray[np.float32],
        available_actions: npt.NDArray[np.int32],
        state: npt.NDArray[np.float32],
        extras: Optional[npt.NDArray[np.float32]]=None
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
