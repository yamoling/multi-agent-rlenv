from typing import Any, Optional, Generic, TypeVar
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

from .observation import Observation, D, S


R = TypeVar("R")


@dataclass
class Transition(Generic[R, S, D]):
    """Transition model"""

    obs: Observation[D, S]
    action: np.ndarray
    reward: R
    done: bool
    info: dict[str, Any]
    obs_: Observation[D, S]
    truncated: bool
    probs: Optional[np.ndarray] = None

    def __init__(
        self,
        obs: Observation,
        action: npt.ArrayLike,
        reward: R,
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
        ho = hash(self.obs)
        ho_ = hash(self.obs_)
        if isinstance(self.reward, np.ndarray):
            hr = hash(self.reward.tobytes())
        else:
            hr = self.reward
        return hash((ho, self.action.tobytes(), hr, self.done, ho_, self.truncated))

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
