from typing import Tuple
from gymnasium.core import Env
import numpy as np
from rlenv.models import RLEnv, Observation


class GymWrapper(RLEnv):
    """Wraps a gym envronment in an RLEnv"""

    def __init__(self, env: Env) -> None:
        super().__init__()
        # if "atari" in env.spec.entry_point.lower():
        #     env = gym.wrappers.AtariPreprocessing(env, frame_skip=1)
        #     env = gym.wrappers.FrameStack(env, 4)
        self.env = env

    @property
    def n_actions(self) -> int:
        return self.env.action_space.n

    @property
    def n_agents(self) -> int:
        return 1

    @property
    def state_shape(self):
        return (1, )

    @property
    def observation_shape(self):
        return self.env.observation_space.shape

    @property
    def name(self) -> str:
        return self.env.spec.id

    def step(self, actions) -> Tuple[Observation, float, bool, dict]:
        obs_, reward, done, truncated, info = self.env.step(actions[0])
        done = done or truncated
        obs_ = Observation(np.array([obs_], dtype=np.float32), self.get_avail_actions(), self.get_state())
        return obs_, reward, done, info

    def get_state(self):
        return np.zeros(1, dtype=np.float32)

    def reset(self):
        obs_data, _info = self.env.reset()
        obs = Observation(np.array([obs_data], dtype=np.float32), self.get_avail_actions(), self.get_state())
        return obs

    def render(self, mode: str = "human"):
        return self.env.render(mode)

    def seed(self, seed_value: int):
        self.env.reset(seed=seed_value)
        self.env.action_space.seed(seed_value)
