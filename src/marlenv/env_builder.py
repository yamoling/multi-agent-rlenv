from dataclasses import dataclass
from typing import Generic, Literal, Optional, TypeVar
import numpy as np
import numpy.typing as npt

from . import wrappers
from .models import Space, MARLEnv

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

    def mask_actions(self, mask: npt.NDArray[np.bool]):
        self._env = wrappers.AvailableActionsMask(self._env, mask)
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

    def time_penalty(self, penalty: float):
        self._env = wrappers.TimePenalty(self._env, penalty)
        return self

    def build(self) -> MARLEnv[AS]:
        """Build and return the environment"""
        return self._env
