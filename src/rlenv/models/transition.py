from dataclasses import dataclass
from typing import Any
import numpy as np

from .observation import Observation

@dataclass
class Transition:
    """Transition model"""
    obs: Observation
    action: np.ndarray[np.int32]
    reward: float
    done: bool
    info: dict[str, Any]
    obs_: Observation

    @property
    def is_done(self) -> bool:
        """Whether the transition is the last one"""
        return self.done

    @property
    def n_agents(self) -> int:
        """The number of agents"""
        return len(self.action)
