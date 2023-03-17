from abc import ABC
import numpy as np
from rlenv.models import RLEnv, Observation


class RLEnvWrapper(RLEnv, ABC):
    """Parent class for all RLEnv wrappers"""
    def __init__(self, env: RLEnv) -> None:
        super().__init__()
        self.env = env

    @property
    def n_actions(self):
        return self.env.n_actions

    @property
    def n_agents(self):
        return self.env.n_agents

    @property
    def state_shape(self):
        return self.env.state_shape

    @property
    def observation_shape(self):
        return self.env.observation_shape

    @property
    def extra_feature_shape(self):
        return self.env.extra_feature_shape

    @property
    def name(self):
        return self.env.name
    
    def kwargs(self) -> dict[str,]:
        return {}

    def step(self, actions: np.ndarray[np.int32]) -> tuple[Observation, float, bool, dict]:
        return self.env.step(actions)
    
    def reset(self):
        return self.env.reset()

    def get_state(self):
        return self.env.get_state()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def render(self, mode: str = "human"):
        return self.env.render(mode)

    def seed(self, seed_value: int):
        return self.env.seed(seed_value)

    def summary(self) -> dict[str, str]:
        # Get the env summary, and add the wrapper to the wrappers list + the wrapper's kwargs
        summary = self.env.summary()
        wrappers = summary.get("wrappers", [])
        wrappers.append(self.__class__.__name__)
        return {
            **summary,
            "n_actions": self.n_actions,
            "n_agents": self.n_agents,
            "obs_shape": self.observation_shape,
            "extras_shape": self.extra_feature_shape,
            "state_shape": self.state_shape,
            "wrappers": wrappers,
            self.__class__.__name__: self.kwargs()
        }
    
    @classmethod
    def from_summary(cls, env: RLEnv, summary: dict[str,]) -> "RLEnvWrapper":
        kwargs = summary.pop(cls.__name__)
        return cls(env, **kwargs)
