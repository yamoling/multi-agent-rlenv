from typing import TypeVar
from gymnasium import Env, spaces
import numpy as np
from rlenv.models import RLEnv, Observation, ActionSpace, DiscreteActionSpace, ContinuousActionSpace

A = TypeVar("A", bound=ActionSpace)


class GymAdapter(RLEnv[A]):
    """Wraps a gym envronment in an RLEnv"""

    def __init__(self, env: Env):
        match env.action_space:
            case spaces.Discrete() as s:
                space = DiscreteActionSpace(1, s.n)
            case spaces.Box() as s:
                space = ContinuousActionSpace(1, s.shape[0], low=s.low, high=s.high)
            case other:
                raise NotImplementedError(f"Action space {other} not supported")
        super().__init__(space)
        self.env = env

    @property
    def n_agents(self) -> int:
        return 1

    @property
    def state_shape(self):
        return (1,)

    @property
    def observation_shape(self):
        return self.env.observation_space.shape

    @property
    def name(self) -> str:
        return self.env.unwrapped.spec.id

    def step(self, actions) -> tuple[Observation, float, bool, bool, dict]:
        obs_, reward, done, truncated, info = self.env.step(actions[0])
        obs_ = Observation(np.array([obs_], dtype=np.float32), self.available_actions(), self.get_state())
        return obs_, reward, done, truncated, info

    def get_state(self):
        return np.zeros(1, dtype=np.float32)

    def reset(self):
        obs_data, _info = self.env.reset()
        obs = Observation(np.array([obs_data], dtype=np.float32), self.available_actions(), self.get_state())
        return obs

    def render(self, mode: str = "human"):
        return self.env.render()

    def seed(self, seed_value: int):
        self.env.reset(seed=seed_value)
        self.env.action_space.seed(seed_value)
