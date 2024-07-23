from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
import numpy.typing as npt

from .observation import Observation


@dataclass
class Transition:
    """Transition model"""

    obs: Observation
    action: np.ndarray
    reward: np.ndarray
    done: bool
    info: dict[str, Any]
    obs_: Observation
    truncated: bool
    probs: np.ndarray | None

    def __init__(
        self,
        obs: Observation,
        action: npt.ArrayLike,
        reward: npt.ArrayLike,
        done: bool,
        info: dict[str, Any],
        obs_: Observation,
        truncated: bool,
        action_probs: Optional[np.ndarray] = None,
    ):
        self.obs = obs
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        self.action = action
        if not isinstance(reward, np.ndarray):
            reward = np.array(reward)
        self.reward = reward
        self.done = done
        self.info = info
        self.obs_ = obs_
        self.truncated = truncated
        self.probs = action_probs

    @property
    def is_terminal(self) -> bool:
        """Whether the transition is the last one"""
        return self.done or self.truncated

    @property
    def n_agents(self) -> int:
        """The number of agents"""
        return len(self.action)

    @property
    def n_actions(self) -> int:
        return int(self.obs.available_actions.shape[-1])

    def __hash__(self) -> int:
        return hash((self.obs, self.action.tobytes(), self.reward.tobytes(), self.done, self.obs_))

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transition):
            return False
        return (
            self.obs == other.obs
            and np.array_equal(self.action, other.action)
            and self.reward == other.reward
            and self.done == other.done
            and self.obs_ == other.obs_
        )
