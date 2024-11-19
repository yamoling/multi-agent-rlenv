from gymnasium import Env, spaces
import numpy as np
import numpy.typing as npt

from marlenv.models import (
    MARLEnv,
    Observation,
    ActionSpace,
    DiscreteSpace,
    ContinuousSpace,
)


class Gym(MARLEnv[ActionSpace, npt.NDArray[np.float32], npt.NDArray[np.float32], float]):
    """Wraps a gym envronment in an RLEnv"""

    def __init__(self, env: Env):
        if env.observation_space.shape is None:
            raise NotImplementedError("Observation space must have a shape")
        match env.action_space:
            case spaces.Discrete() as s:
                space = ActionSpace(1, DiscreteSpace(int(s.n)))
            case spaces.Box() as s:
                low = s.low.astype(np.float32)
                high = s.high.astype(np.float32)
                if not isinstance(low, np.ndarray):
                    low = np.full(s.shape, s.low, dtype=np.float32)
                if not isinstance(high, np.ndarray):
                    high = np.full(s.shape, s.high, dtype=np.float32)
                space = ActionSpace(1, ContinuousSpace(low=low, high=high))
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
            self.get_state(),
        )
        return self.last_obs, reward, done, truncated, info

    def get_state(self):
        return np.zeros(1, dtype=np.float32)

    def reset(self):
        obs_data, _info = self.env.reset()
        self.last_obs = Observation(
            np.array([obs_data], dtype=np.float32),
            self.available_actions(),
            self.get_state(),
        )
        return self.last_obs

    def render(self, mode: str = "human"):
        return self.env.render()

    def seed(self, seed_value: int):
        self.env.reset(seed=seed_value)
