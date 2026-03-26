from dataclasses import dataclass
from typing import Generic, Literal, Optional, TypeVar, overload

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from . import wrappers
from .models import MARLEnv, Space

AS = TypeVar("AS", bound=Space)


@dataclass
class Builder(Generic[AS]):
    """Builder for environments"""

    _env: MARLEnv[AS]

    def __init__(self, env: MARLEnv[AS]):
        self._env = env

    def time_limit(self, n_steps: int, add_extra: bool = True, truncation_penalty: Optional[float] = None):
        """
        Limits the number of time steps for an episode. When the number of steps is reached, then the episode is truncated.

        - If the `add_extra` flag is set to True, then an extra signal is added to the observation, which is the ratio of the
        current step over the maximum number of steps. In this case, the done flag is also set to True when the maximum
        number of steps is reached.
        - The `truncated` flag is only set to `True` when the maximum number of steps is reached and the episode is not done.
        - The `truncation_penalty` is subtracted from the reward when the episode is truncated. This is only possible when
        the `add_extra` flag is set to True, otherwise the agent is not able to anticipate this penalty.
        """
        self._env = wrappers.TimeLimit(self._env, n_steps, add_extra, truncation_penalty)
        return self

    def delay_rewards(self, delay: int):
        """Delays the rewards by `delay` steps"""
        self._env = wrappers.DelayedReward(self._env, delay)
        return self

    def pad(self, to_pad: Literal["obs", "extra"], n: int):
        match to_pad:
            case "obs":
                self._env = wrappers.PadObservations(self._env, n)
            case "extra":
                self._env = wrappers.PadExtras(self._env, n)
            case other:
                raise ValueError(f"Unknown padding type: {other}")
        return self

    def agent_id(self):
        """Adds agent ID to the observations"""
        self._env = wrappers.AgentId(self._env)
        return self

    def last_action(self):
        """Adds the last action to the observations"""
        self._env = wrappers.LastAction(self._env)  # type: ignore
        return self

    @overload
    def mask_actions(self, mask: npt.NDArray[np.bool] | list[bool]) -> Self:
        """
        Mask the actions whose indices are set to `False`. For instance, with [True, False, True, False, True], only actions 0, 2 and 4 are available.
        """

    @overload
    def mask_actions(self, mask: int | list[int]) -> Self:
        """Mask the actions whose index (indices) is (are) given as parameter. For instance, with `[1, 3]`, actions 1 and 3 are made unavailable for all agents. With `1`, then action `1` is made unavailable for all agents."""

    def mask_actions(self, mask: npt.NDArray[np.bool] | list[bool] | int | list[int]):
        mask_array = np.full((self._env.n_agents, self._env.n_actions), True, dtype=np.bool)
        if isinstance(mask, int):
            assert self._env.n_actions > mask, f"Action {mask} does not exist."
            mask_array[:, mask] = False
        elif isinstance(mask, list):
            is_int = isinstance(mask[0], int)
            for i, m in enumerate(mask):
                if isinstance(m, int):
                    assert is_int, f"Mask {mask} is a list of integers, but element {m} is not an integer."
                    assert m < self._env.n_actions, f"Action {m} does not exist."
                    mask_array[:, m] = False
                else:
                    assert not is_int, f"Mask {mask} is a list of booleans, but element {m} is not a boolean."
                    mask_array[:, i] = m
        else:
            mask_array = mask
        self._env = wrappers.AvailableActionsMask(self._env, mask_array)
        return self

    def centralised(self):
        """Centralises the observations and actions"""
        from marlenv.models import MultiDiscreteSpace

        assert isinstance(self._env.action_space, MultiDiscreteSpace)
        self._env = wrappers.Centralized(self._env)  # type: ignore
        return self

    def record(
        self,
        folder: str,
        encoding: Literal["mp4", "avi"] = "mp4",
    ):
        """Add video recording to the environment"""
        self._env = wrappers.VideoRecorder(self._env, folder, video_encoding=encoding)
        return self

    def available_actions(self):
        """Adds the available actions to the observations extras"""
        self._env = wrappers.AvailableActions(self._env)
        return self

    def available_actions_mask(self, mask: npt.NDArray[np.bool]):
        """Masks a subset of the available actions.
        The mask must have shape (n_agents, n_actions), where any False value will be masked."""
        self._env = wrappers.AvailableActionsMask(self._env, mask)
        return self

    def blind(self, p: float):
        """Blinds (replaces with zeros) the observations with probability p"""
        self._env = wrappers.Blind(self._env, p)  # type: ignore
        return self

    def randomize_actions(self, p: list[float] | float | npt.NDArray[np.float32]):
        """Randomizes the actions with probability 0 <= p <= 1. The length of p must be equal to the number of agents."""
        self._env = wrappers.ActionRandomizer(self._env, p)  # type: ignore
        return self

    def time_penalty(self, penalty: float):
        self._env = wrappers.TimePenalty(self._env, penalty)
        return self

    def build(self) -> MARLEnv[AS]:
        """Build and return the environment"""
        return self._env
