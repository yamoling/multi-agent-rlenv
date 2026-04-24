from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from .observation import Observation
from .state import State


@dataclass
class Step:
    """
    A step contains the action and the result of performing that action in the environment:
        - the action
        - the new observation
        - the new state
        - the reward received for the step performed
        - whether the episode is done or truncated
        - some info (mainly for logging purposes)
    """

    action: npt.NDArray
    """The action performed by the agents to obtain this step."""
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
        action: npt.ArrayLike | Sequence,
        obs: Observation,
        state: State,
        reward: npt.NDArray[np.float32] | float | Sequence[float],
        done: bool,
        truncated: bool = False,
        info: dict[str, Any] | None = None,
    ):
        if info is None:
            info = {}
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        self.action = action
        self.obs = obs
        self.state = state
        match reward:
            case int() | float():
                self.reward = np.array([reward], dtype=np.float32)
            case np.ndarray():
                self.reward = reward.astype(np.float32)
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
        return iter((self.obs, self.state, self.reward, self.done, self.truncated, self.info))
