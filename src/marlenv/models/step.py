from dataclasses import dataclass
from typing import Generic, Any, Optional
from typing_extensions import TypeVar
import numpy as np
import numpy.typing as npt
from .observation import Observation, ObsType
from .state import State, StateType

RewardType = TypeVar("RewardType", bound=float | npt.NDArray, default=float)


@dataclass
class Step(Generic[ObsType, StateType, RewardType]):
    obs: Observation[ObsType]
    """The new observation (1 per agent) of the environment resulting from the agent's action."""
    state: State[StateType]
    """The new state of the environment."""
    reward: RewardType
    """The reward obtained after the agents' joint action."""
    done: bool
    """Whether the episode is done."""
    truncated: bool
    """Whether the episode has been truncated, i.e. is not done but has been cut for some reason (e.g. max steps)."""
    info: dict[str, Any]
    """Additional information that the environment might provide."""

    def __init__(
        self,
        obs: Observation[ObsType],
        state: State[StateType],
        reward: RewardType,
        done: bool,
        truncated: bool = False,
        info: Optional[dict[str, Any]] = None,
    ):
        self.obs = obs
        self.state = state
        self.reward = reward
        self.done = done
        self.truncated = truncated
        self.info = info or {}

    def astuple(self):
        return self.obs, self.state, self.reward, self.done, self.truncated, self.info

    def __iter__(self):
        return iter(self.astuple())

    @property
    def is_terminal(self):
        return self.truncated or self.done
