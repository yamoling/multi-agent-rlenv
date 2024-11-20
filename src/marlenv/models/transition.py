from typing import Any, Optional, Generic
import numpy as np
import numpy.typing as npt

from .observation import Observation, ObsType
from .state import State, StateType
from .step import Step, RewardType


class Transition(Generic[RewardType, StateType, ObsType]):
    """Transition model"""

    obs: Observation[ObsType]
    state: State[StateType]
    action: np.ndarray
    reward: RewardType
    done: bool
    info: dict[str, Any]
    next_obs: Observation[ObsType]
    next_state: State[StateType]
    truncated: bool
    action_probs: Optional[np.ndarray] = None

    def __init__(
        self,
        obs: Observation[ObsType],
        state: State[StateType],
        action: npt.ArrayLike,
        reward: RewardType,
        done: bool,
        info: dict[str, Any],
        next_obs: Observation[ObsType],
        next_state: State[StateType],
        truncated: bool,
        action_probs: Optional[np.ndarray] = None,
    ):
        self.obs = obs
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        self.action = action
        self.reward = reward
        self.done = done
        self.info = info
        self.next_obs = next_obs
        self.truncated = truncated
        self.action_probs = action_probs
        self.state = state
        self.next_state = next_state

    @staticmethod
    def from_step(
        prev_obs: Observation[ObsType],
        prev_state: State[StateType],
        actions: np.ndarray,
        step: Step[ObsType, StateType, RewardType],
        probs: Optional[np.ndarray] = None,
    ):
        return Transition(
            obs=prev_obs,
            state=prev_state,
            action=actions,
            reward=step.reward,
            done=step.done,
            info=step.info,
            next_obs=step.obs,
            next_state=step.state,
            truncated=step.truncated,
            action_probs=probs,
        )

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
        ho_ = hash(self.next_obs)
        hs = hash(self.state)
        hs_ = hash(self.next_state)
        if isinstance(self.reward, np.ndarray):
            hr = hash(self.reward.tobytes())
        else:
            hr = self.reward
        return hash((ho, self.action.tobytes(), hr, self.done, ho_, self.truncated, hs, hs_))

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transition):
            return False
        return (
            self.obs == other.obs
            and self.state == other.state
            and np.array_equal(self.action, other.action)
            and self.reward == other.reward
            and self.done == other.done
            and self.next_obs == other.next_obs
            and self.truncated == other.truncated
            and self.state == other.state
        )
