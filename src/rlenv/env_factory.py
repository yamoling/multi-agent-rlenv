from dataclasses import dataclass
from typing import Union
from pettingzoo import ParallelEnv

from .models import RLEnv
from .config import EnvConfig
from .wrappers import SMACWrapper, GymWrapper, AgentIdWrapper, LastActionWrapper


def make(env_name: str) -> RLEnv:
    """Make an RLEnv from Gym, SMAC or Laser"""
    config = EnvConfig(env_name, None, False, False)
    return make_from_config(config)


def make_from_config(config: EnvConfig) -> RLEnv:
    """Make the environment from the given configuration"""
    builder = EnvBuilder(config.env)
    if config.with_last_action:
        builder.with_last_action()
    if config.with_agent_id:
        builder.with_agent_id()
    if config.horizon and config.horizon > 0:
        builder.horizon(config.horizon)
    return builder.build()


@dataclass
class EnvBuilder:
    """Builder for environments"""
    _env: RLEnv

    def __init__(self, env: str|RLEnv|ParallelEnv) -> None:
        match env:
            case str():
                self._env = self._init_env(env)
            case RLEnv():
                self._env = env
            case ParallelEnv():
                from .wrappers import PettingZooWrapper
                self._env = PettingZooWrapper(env)
            case _:
                raise NotImplementedError()

    def _init_env(self, env: str) -> RLEnv:
        if env.lower().startswith("smac"):
            return self._get_smac_env(env)
        else:
            import gymnasium as gym
            return GymWrapper(gym.make(env))

    def _get_smac_env(self, env: str) -> RLEnv:
        env = env.lower()
        map_name = env[len("smac:"):]
        if len(map_name) == 0:
            map_name = "3m"
        return SMACWrapper(map_name=map_name)

    # def _get_laser_env(self, env_name: str) -> RLEnv:
    #     from laser_env import LaserEnv, ObservationType
    #     map_name = "lvl1"
    #     obs_dtype = ObservationType.RELATIVE_POSITIONS
    #     splits = env_name.split(":")
    #     if len(splits) > 1:
    #         map_name = splits[1]
    #     if len(splits) > 2:
    #         obs_dtype_str = splits[2]
    #         if obs_dtype_str == "rgb":
    #             obs_dtype = ObservationType.RGB_IMAGE
    #         elif obs_dtype_str == "relative":
    #             obs_dtype = ObservationType.RELATIVE_POSITIONS
    #         elif obs_dtype_str.startswith("layer"):
    #             obs_dtype = ObservationType.LAYERED
    #         else:
    #             raise ValueError(f"Unknown observation type: {obs_dtype_str}")
    #     if len(splits) > 3:
    #         print(f"WARNING: Unknown arguments for laser env: f{splits[2:]}")
    #     return LaserEnv(map_name, obs_type=obs_dtype)

    def horizon(self, horizon: int):
        """Set the horizon (time limit) of the environment"""
        if hasattr(self._env, "horizon"):
            setattr(self._env, "horizon", horizon)
        elif hasattr(self._env, "time_limit"):
            setattr(self._env, "time_limit", horizon)
        else:
            raise NotImplementedError(f"{self._env.name} does not support horizon !")
        return self

    def with_agent_id(self):
        """Adds agent ID to the observations"""
        self._env = AgentIdWrapper(self._env)
        return self

    def with_last_action(self):
        """Adds the last action to the observations"""
        self._env = LastActionWrapper(self._env)
        return self

    def build(self) -> RLEnv:
        """Build and return the RLEnv"""
        return self._env
