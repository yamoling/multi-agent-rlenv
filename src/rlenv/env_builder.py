import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, TypeVar, Generic, overload, Any


from .models import RLEnv, ActionSpace
from . import wrappers

A = TypeVar("A", bound=ActionSpace, covariant=True)

try:
    from pettingzoo import ParallelEnv

    @overload
    def make(env: ParallelEnv) -> RLEnv[ActionSpace]:
        ...
except ImportError:
    ParralelEnv = Any


@overload
def make(env: str) -> RLEnv[ActionSpace]:
    ...


@overload
def make(env: RLEnv[A]) -> RLEnv[A]:
    ...


def make(env):
    """Make an RLEnv from Gym, SMAC or PettingZoo"""
    return Builder(env).build()


@dataclass
class Builder(Generic[A]):
    """Builder for environments"""

    _env: RLEnv[A]
    _test_env: RLEnv[A]

    def __init__(self, env: RLEnv[A] | str | ParallelEnv):
        match env:
            case str():
                self._env = self._init_env(env)
            case RLEnv():
                self._env = env
            case ParallelEnv():  # type: ignore
                try:
                    from rlenv.adapters import PettingZoo
                except ImportError:
                    raise ImportError("PettingZoo is not installed")
                self._env = PettingZoo(env)  # type: ignore
            case _:
                raise NotImplementedError()
        self._test_env = deepcopy(self._env)

    def _init_env(self, env: str) -> RLEnv:
        if env.lower().startswith("smac"):
            return self._get_smac_env(env)

        try:
            import gymnasium as gym
            from rlenv.adapters import Gym

            return Gym(gym.make(env, render_mode="rgb_array"))
        except ImportError:
            raise ImportError("Gymnasium is not installed")

    def _get_smac_env(self, env_name: str) -> RLEnv:
        try:
            from rlenv.adapters import SMAC
        except ImportError:
            raise ImportError("SMAC is not installed")
        env_name = env_name.lower()
        map_name = env_name[len("smac:") :]
        if len(map_name) == 0:
            map_name = "3m"
        return SMAC(map_name=map_name)

    def time_limit(self, n_steps: int, add_extra: bool = False):
        """Set the time limit (horizon) of the environment (train & test)"""
        self._env = wrappers.TimeLimit(self._env, n_steps, add_extra)
        self._test_env = wrappers.TimeLimit(self._test_env, n_steps, add_extra)
        return self

    def pad(self, to_pad: Literal["obs", "extra"], n: int):
        match to_pad:
            case "obs":
                self._env = wrappers.PadObservations(self._env, n)
                self._test_env = wrappers.PadObservations(self._test_env, n)
            case "extra":
                self._env = wrappers.PadExtras(self._env, n)
                self._test_env = wrappers.PadExtras(self._test_env, n)
            case other:
                raise ValueError(f"Unknown padding type: {other}")
        return self

    def agent_id(self):
        """Adds agent ID to the observations"""
        self._env = wrappers.AgentId(self._env)
        self._test_env = wrappers.AgentId(self._test_env)
        return self

    def last_action(self):
        """Adds the last action to the observations"""
        self._env = wrappers.LastAction(self._env)
        self._test_env = wrappers.LastAction(self._test_env)
        return self

    def centralised(self):
        """Centralises the observations and actions"""
        self._env = wrappers.Centralised(self._env)
        self._test_env = wrappers.Centralised(self._test_env)
        return self

    def record(
        self,
        folder: str,
        record_training=False,
        encoding: Literal["mp4", "avi"] = "mp4",
    ):
        """Add video recording of runs. Onnly records tests by default."""
        if record_training:
            self._env = wrappers.VideoRecorder(self._env, os.path.join(folder, "training"), video_encoding=encoding)
            folder = os.path.join(folder, "test")
        self._test_env = wrappers.VideoRecorder(self._test_env, folder, video_encoding=encoding)
        return self

    def available_actions(self):
        """Adds the available actions to the observations extras"""
        self._env = wrappers.AvailableActions(self._env)
        self._test_env = wrappers.AvailableActions(self._test_env)
        return self

    def blind(self, p: float):
        """Blinds the observations with probability p"""
        self._env = wrappers.Blind(self._env, p)
        self._test_env = wrappers.Blind(self._test_env, p)
        return self

    def time_penalty(self, penalty: float):
        self._env = wrappers.TimePenalty(self._env, penalty)
        self._test_env = wrappers.TimePenalty(self._test_env, penalty)
        return self

    def build(self) -> RLEnv[A]:
        """Build and return the environment"""
        return self._env

    def build_all(self) -> tuple[RLEnv[A], RLEnv[A]]:
        """Build and return the training and testing environments"""
        return self._env, self._test_env
