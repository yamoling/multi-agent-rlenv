from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import numpy.typing as npt

from .observation import Observation

@dataclass
class Transition:
    """Transition model"""
    obs: Observation
    action: npt.NDArray[np.int32]
    reward: float
    done: bool
    info: Dict[str, Any]
    obs_: Observation

    @property
    def is_done(self) -> bool:
        """Whether the transition is the last one"""
        return self.done

    @property
    def n_agents(self) -> int:
        """The number of agents"""
        return len(self.action)
