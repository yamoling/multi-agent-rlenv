import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, TypeVar, Generic, TypeAlias


from .models import RLEnv, ActionSpace
from . import wrappers

A = TypeVar("A", bound=ActionSpace)

try:
    from pettingzoo import ParallelEnv

    EnvType: TypeAlias = str | RLEnv | ParallelEnv
except ImportError:
    # EnvType: TypeAlias = str | RLEnv
    pass


def make(env: str | RLEnv) -> RLEnv[ActionSpace]:
    """Make an RLEnv from Gym, SMAC or PettingZoo"""
    return Builder(env).build()


@dataclass
class Builder(Generic[A]):
    """Builder for environments"""

    _env: RLEnv[A]
    _test_env: RLEnv[A]

    def __init__(self, env: str | RLEnv[A]):
        match env:
            case str():
                self._env = self._init_env(env)
            case RLEnv():
                self._env = env
            case ParallelEnv():
                try:
                    from rlenv.adapters import PettingZooAdapter
                except ImportError:
                    raise ImportError("PettingZoo is not installed")
                self._env = PettingZooAdapter(env)  # type: ignore
            case _:
                raise NotImplementedError()
        self._test_env = deepcopy(self._env)

    def _init_env(self, env: str) -> RLEnv:
        if env.lower().startswith("smac"):
            return self._get_smac_env(env)

        try:
            import gymnasium as gym
            from rlenv.adapters import GymAdapter

            return GymAdapter(gym.make(env, render_mode="rgb_array"))
        except ImportError:
            raise ImportError("Gymnasium is not installed")

    def _get_smac_env(self, env_name: str) -> RLEnv:
        try:
            from rlenv.adapters import SMACAdapter
        except ImportError:
            raise ImportError("SMAC is not installed")
        env_name = env_name.lower()
        map_name = env_name[len("smac:") :]
        if len(map_name) == 0:
            map_name = "3m"
        return SMACAdapter(map_name=map_name)

    def time_limit(self, n_steps: int, add_extra: bool = False):
        """Set the time limit (horizon) of the environment (train & test)"""
        self._env = wrappers.TimeLimitWrapper(self._env, n_steps, add_extra)
        self._test_env = wrappers.TimeLimitWrapper(self._test_env, n_steps, add_extra)
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
        self._env = wrappers.AgentIdWrapper(self._env)
        self._test_env = wrappers.AgentIdWrapper(self._test_env)
        return self

    def last_action(self):
        """Adds the last action to the observations"""
        self._env = wrappers.LastActionWrapper(self._env)
        self._test_env = wrappers.LastActionWrapper(self._test_env)
        return self

    def record(
        self,
        folder: str,
        record_training=False,
        encoding: Literal["mp4", "avi"] = "mp4",
    ):
        """Add video recording of runs. Onnly records tests by default."""
        if record_training:
            self._env = wrappers.VideoRecorder(
                self._env, os.path.join(folder, "training"), video_encoding=encoding
            )
            folder = os.path.join(folder, "test")
        self._test_env = wrappers.VideoRecorder(
            self._test_env, folder, video_encoding=encoding
        )
        return self

    def available_actions(self):
        """Adds the available actions to the observations extras"""
        self._env = wrappers.AvailableActionsWrapper(self._env)
        self._test_env = wrappers.AvailableActionsWrapper(self._test_env)
        return self

    def blind(self, p: float):
        """Blinds the observations with probability p"""
        self._env = wrappers.BlindWrapper(self._env, p)
        self._test_env = wrappers.BlindWrapper(self._test_env, p)
        return self

    def intrinsic_reward(
        self,
        method: Literal["linear", "exp"],
        initial_reward: float,
        anneal: int,
        also_for_testing=False,
    ):
        match method:
            case "linear":
                self._env = wrappers.LinearStateCount(self._env, initial_reward, anneal)
                if also_for_testing:
                    self._test_env = wrappers.LinearStateCount(
                        self._test_env, initial_reward, anneal
                    )
            case "exp":
                self._env = wrappers.DecreasingExpStateCount(
                    self._env, initial_reward, anneal
                )
                if also_for_testing:
                    self._test_env = wrappers.DecreasingExpStateCount(
                        self._test_env, initial_reward, anneal
                    )
            case other:
                raise ValueError(f"'{other}' is not a known extrinsic reward wrapper")
        return self

    def time_penalty(self, penalty: float):
        self._env = wrappers.TimePenaltyWrapper(self._env, penalty)
        self._test_env = wrappers.TimePenaltyWrapper(self._test_env, penalty)
        return self

    def force_actions(self, agent_actions: dict[int, int]):
        self._env = wrappers.ForceActionWrapper(self._env, agent_actions)
        self._test_env = wrappers.ForceActionWrapper(self._test_env, agent_actions)
        return self

    def build(self) -> RLEnv[A]:
        """Build and return the environment"""
        return self._env

    def build_all(self) -> tuple[RLEnv[A], RLEnv[A]]:
        """Build and return the training and testing environments"""
        return self._env, self._test_env
