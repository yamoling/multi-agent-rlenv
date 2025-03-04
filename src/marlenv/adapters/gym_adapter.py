from dataclasses import dataclass
from typing import Sequence

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import Env, spaces

from marlenv.models import (
    ActionSpace,
    ContinuousActionSpace,
    DiscreteActionSpace,
    MARLEnv,
    Observation,
    State,
    Step,
)


@dataclass
class Gym(MARLEnv[Sequence | npt.NDArray, ActionSpace]):
    """Wraps a gym envronment in an RLEnv"""

    def __init__(self, env: Env | str, **kwargs):
        if isinstance(env, str):
            env = gym.make(env, render_mode="rgb_array", **kwargs)
        if env.observation_space.shape is None:
            raise NotImplementedError("Observation space must have a shape")
        match env.action_space:
            case spaces.Discrete() as s:
                space = DiscreteActionSpace(1, int(s.n))
            case spaces.Box() as s:
                low = s.low.astype(np.float32)
                high = s.high.astype(np.float32)
                if not isinstance(low, np.ndarray):
                    low = np.full(s.shape, s.low, dtype=np.float32)
                if not isinstance(high, np.ndarray):
                    high = np.full(s.shape, s.high, dtype=np.float32)
                space = ContinuousActionSpace(1, low, high)
            case other:
                raise NotImplementedError(f"Action space {other} not supported")
        super().__init__(space, env.observation_space.shape, (1,))
        self.env = env
        if self.env.unwrapped.spec is not None:
            self.name = self.env.unwrapped.spec.id
        else:
            self.name = "gym-no-id"
        self.last_obs = None

    def get_observation(self):
        if self.last_obs is None:
            raise ValueError("No observation available. Call reset() first.")
        return self.last_obs

    def step(self, actions):
        obs, reward, done, truncated, info = self.env.step(list(actions)[0])
        self.last_obs = Observation(
            np.array([obs], dtype=np.float32),
            self.available_actions(),
        )
        return Step(
            self.last_obs,
            self.get_state(),
            np.array([reward]),
            done,
            truncated,
            info,
        )

    def get_state(self):
        return State(np.zeros(1, dtype=np.float32))

    def reset(self):
        obs_data, _info = self.env.reset()
        self.last_obs = Observation(
            np.array([obs_data], dtype=np.float32),
            self.available_actions(),
        )
        return self.last_obs, self.get_state()

    def get_image(self):
        return self.env.render()

    def seed(self, seed_value: int):
        self.env.reset(seed=seed_value)
