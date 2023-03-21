import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal
from pettingzoo import ParallelEnv

from .models import RLEnv
from . import adapters
from . import wrappers



def make(env: str | ParallelEnv) -> RLEnv:
    """Make an RLEnv from Gym, SMAC or PettingZoo"""
    return Builder(env).build()


@dataclass
class Builder:
    """Builder for environments"""
    _env: RLEnv
    _test_env: RLEnv

    def __init__(self, env: str|RLEnv|ParallelEnv) -> None:
        match env:
            case str():
                self._env = self._init_env(env)
            case RLEnv():
                self._env = env
            case ParallelEnv():
                self._env = adapters.PettingZooAdapter(env)
            case _:
                raise NotImplementedError()
        self._test_env = deepcopy(self._env)

    def _init_env(self, env: str) -> RLEnv:
        if env.lower().startswith("smac"):
            return self._get_smac_env(env)
        else:
            import gymnasium as gym
            return adapters.GymAdapter(gym.make(env, render_mode="rgb_array"))

    def _get_smac_env(self, env_name: str) -> RLEnv:
        env_name = env_name.lower()
        map_name = env_name[len("smac:"):]
        if len(map_name) == 0:
            map_name = "3m"
        return adapters.SMACAdapter(map_name=map_name)

    def time_limit(self, n_steps: int):
        """Set the time limit (horizon) of the environment (train & test)"""
        if hasattr(self._env, "horizon"):
            setattr(self._env, "horizon", n_steps)
            setattr(self._test_env, "horizon", n_steps)
        elif hasattr(self._env, "time_limit"):
            setattr(self._env, "time_limit", n_steps)
            setattr(self._test_env, "time_limit", n_steps)
        else:
            self._env = wrappers.TimeLimitWrapper(self._env, n_steps)
            self._test_env = wrappers.TimeLimitWrapper(self._test_env, n_steps)
        return self
    
    def pad(self, to_pad: Literal["obs", "extra"], n: int):
        match to_pad:
            case "obs": 
                self._env = wrappers.PadObservations(self._env, n)
                self._test_env = wrappers.PadObservations(self._test_env, n)
            case "extra": 
                self._env = wrappers.PadExtras(self._env, n)
                self._test_env = wrappers.PadExtras(self._test_env, n)
            case other: raise ValueError(f"Unknown padding type: {other}")
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

    def record(self, folder: str, record_training=False, encoding: Literal["mp4", "avi"]="mp4"):
        """Add video recording of runs. Onnly records tests by default."""
        if record_training:
            self._env = wrappers.VideoRecorder(self._env, os.path.join(folder, "training"), video_encoding=encoding)
            folder = os.path.join(folder, "test")
        self._test_env = wrappers.VideoRecorder(self._test_env, folder, video_encoding=encoding)
        return self

    def intrinsic_reward(self, method: Literal["linear", "exp"], initial_reward: float, anneal: float, also_for_testing=False):
        match method:
            case "linear":
                self._env = wrappers.LinearStateCount(self._env, initial_reward, anneal)
                if also_for_testing:
                    self._test_env = wrappers.LinearStateCount(self._test_env, initial_reward, anneal)
            case "exp":
                self._env = wrappers.DecreasingExpStateCount(self._env, initial_reward, anneal)
                if also_for_testing:
                    self._test_env = wrappers.DecreasingExpStateCount(self._test_env, initial_reward, anneal)
            case other:
                raise ValueError(f"'{other}' is not a known extrinsic reward wrapper")
        return self
    
    def penalty(self, penalty: float):
        self.env = wrappers.PenaltyWrapper(self._env, penalty)
        self.test_env = wrappers.PenaltyWrapper(self._test_env, penalty)

    def build(self) -> RLEnv:
        """Build and return the environment"""
        return self._env

    def build_all(self) -> tuple[RLEnv, RLEnv]:
        """Build and return the training and testing environments"""
        return self._env, self._test_env
        