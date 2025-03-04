from typing import Any, Optional, Sequence
from typing_extensions import deprecated
import numpy.typing as npt
import numpy as np
from dataclasses import dataclass
from .observation import Observation
from .state import State


@dataclass
class Step:
    obs: Observation
    """The new observation (1 per agent) of the environment resulting from the agent's action."""
    state: State
    """The new state of the environment."""
    reward: npt.NDArray[np.float32]
    """The reward obtained after the agents' joint action."""
    done: bool
    """Whether the episode is done."""
    truncated: bool
    """Whether the episode has been truncated, i.e. is not done but has been cut for some reason (e.g. max steps)."""
    info: dict[str, Any]
    """Additional information that the environment might provide."""

    def __init__(
        self,
        obs: Observation,
        state: State,
        reward: npt.NDArray[np.float32] | float | Sequence[float],
        done: bool,
        truncated: bool = False,
        info: Optional[dict[str, Any]] = None,
    ):
        if info is None:
            info = {}
        self.obs = obs
        self.state = state
        match reward:
            case int() | float():
                self.reward = np.array([reward], dtype=np.float32)
            case np.ndarray():
                self.reward = reward
            case other:
                # We assume this is a sequence of some sort
                self.reward = np.array(other, dtype=np.float32)
        self.done = done
        self.truncated = truncated
        self.info = info

    @property
    def is_terminal(self):
        """
        Whether the episode is done or truncated, i.e. the step was the last of an episode.

        Typically used in a `while not step.is_terminal` loop.
        """
        return self.truncated or self.done

    def __iter__(self):
        yield self.obs
        yield self.state
        yield self.reward
        yield self.done
        yield self.truncated
        yield self.info

    @deprecated("Step is no longer a `NamedTuple` and should be modified by setting the attributes directly.")
    def with_attrs(
        self,
        obs: Optional[Observation] = None,
        state: Optional[State] = None,
        reward: Optional[npt.NDArray[np.float32]] = None,
        done: Optional[bool] = None,
        truncated: Optional[bool] = None,
        info: Optional[dict[str, Any]] = None,
    ):
        """
        Return a new Step object with the given attributes replaced.

        Note that the new object shares the same references as the original one for the attributes that are not replaced.
        """
        return Step(
            obs if obs is not None else self.obs,
            state if state is not None else self.state,
            reward if reward is not None else self.reward,
            done if done is not None else self.done,
            truncated if truncated is not None else self.truncated,
            info if info is not None else self.info,
        )
