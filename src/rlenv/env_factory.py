from dataclasses import dataclass
from pettingzoo import ParallelEnv

from .models import RLEnv
from .wrappers import SMACWrapper, GymWrapper, AgentIdWrapper, LastActionWrapper


def make(env: str | ParallelEnv) -> RLEnv:
    """Make an RLEnv from Gym, SMAC or Laser"""
    return Builder(env).build()


@dataclass
class Builder:
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
