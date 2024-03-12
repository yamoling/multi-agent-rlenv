from gymnasium import Env, spaces
import numpy as np
from serde import serde

from rlenv.models import (
    RLEnv,
    Observation,
    ActionSpace,
    DiscreteActionSpace,
    ContinuousActionSpace,
)


@serde
class Gym(RLEnv[ActionSpace]):
    """Wraps a gym envronment in an RLEnv"""

    def __init__(self, env: Env):
        if env.observation_space.shape is None:
            raise NotImplementedError("Observation space must have a shape")
        match env.action_space:
            case spaces.Discrete() as s:
                space = DiscreteActionSpace(1, int(s.n))
            case spaces.Box() as s:
                if len(s.low.shape) > 1 or len(s.high.shape) > 1:
                    raise NotImplementedError("Multi-dimensional action spaces not supported")
                space = ContinuousActionSpace(1, s.shape[0], low=float(s.low[0]), high=float(s.high[0]))
            case other:
                raise NotImplementedError(f"Action space {other} not supported")
        super().__init__(space, env.observation_space.shape, (1,))
        self.env = env
        if self.env.unwrapped.spec is not None:
            self.name = self.env.unwrapped.spec.id
        else:
            self.name = "gym-no-id"

    def step(self, actions):
        obs_, reward, done, truncated, info = self.env.step(list(actions)[0])
        obs_ = Observation(
            np.array([obs_], dtype=np.float32),
            self.available_actions(),
            self.get_state(),
        )
        return obs_, float(reward), done, truncated, info

    def get_state(self):
        return np.zeros(1, dtype=np.float32)

    def reset(self):
        obs_data, _info = self.env.reset()
        obs = Observation(
            np.array([obs_data], dtype=np.float32),
            self.available_actions(),
            self.get_state(),
        )
        return obs

    def render(self, mode: str = "human"):
        return self.env.render()

    def seed(self, seed_value: int):
        self.env.reset(seed=seed_value)
