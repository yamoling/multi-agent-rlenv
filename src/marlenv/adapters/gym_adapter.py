import sys
import cv2
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
        self._gym_env = env
        if self._gym_env.unwrapped.spec is not None:
            self.name = self._gym_env.unwrapped.spec.id
        else:
            self.name = "gym-no-id"
        self._last_obs = None

    def get_observation(self):
        if self._last_obs is None:
            raise ValueError("No observation available. Call reset() first.")
        return self._last_obs

    def step(self, actions):
        obs, reward, done, truncated, info = self._gym_env.step(list(actions)[0])
        self._last_obs = Observation(
            np.array([obs], dtype=np.float32),
            self.available_actions(),
        )
        return Step(
            self._last_obs,
            self.get_state(),
            np.array([reward]),
            done,
            truncated,
            info,
        )

    def get_state(self):
        return State(np.zeros(1, dtype=np.float32))

    def reset(self):
        obs_data, _info = self._gym_env.reset()
        self._last_obs = Observation(
            np.array([obs_data], dtype=np.float32),
            self.available_actions(),
        )
        return self._last_obs, self.get_state()

    def get_image(self):
        image = np.array(self._gym_env.render())
        if sys.platform in ("linux", "linux2"):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def seed(self, seed_value: int):
        self._gym_env.reset(seed=seed_value)
