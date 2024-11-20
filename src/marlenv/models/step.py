from typing import Generic, Any, NamedTuple, Optional
from typing_extensions import TypeVar
import numpy.typing as npt
from .observation import Observation, ObsType
from .state import State, StateType


RewardType = TypeVar("RewardType", bound=float | npt.NDArray, default=float)


# It is not possible to override the __new__ of a NamedTuple. Therefore, we can not give a default value
# to the `info` attribute. We have to use a NamedTuple instead of a dataclass because we want to
# be able to unpack the Step object in a tuple without losing type information.
class _Step(NamedTuple, Generic[ObsType, StateType, RewardType]):
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

    @property
    def is_terminal(self):
        return self.truncated or self.done

    def with_attrs(
        self,
        obs: Optional[Observation[ObsType]] = None,
        state: Optional[State[StateType]] = None,
        reward: Optional[RewardType] = None,
        done: Optional[bool] = None,
        truncated: Optional[bool] = None,
        info: Optional[dict[str, Any]] = None,
    ):
        """
        Return a new Step object with the given attributes replaced.

        Note that the new object shares the same references as the original one for the attributes that are not replaced.
        """
        return _Step(
            obs if obs is not None else self.obs,
            state if state is not None else self.state,
            reward if reward is not None else self.reward,
            done if done is not None else self.done,
            truncated if truncated is not None else self.truncated,
            info if info is not None else self.info,
        )


class Step(Generic[ObsType, StateType, RewardType], _Step[ObsType, StateType, RewardType]):
    """
    Named Tuple for a step in the environment.

    - obs: The observation resulting from the action.
    - state: The new state of the environment.
    - reward: The team reward.
    - done: Whether the episode is over
    - truncated: Whether the episode is truncated
    - info: Extra information
    """

    def __new__(
        cls,
        obs: Observation[ObsType],
        state: State[StateType],
        reward: RewardType,
        done: bool,
        truncated: bool = False,
        info: Optional[dict[str, Any]] = None,
    ):
        if info is None:
            info = {}
        return super().__new__(cls, obs, state, reward, done, truncated, info)
