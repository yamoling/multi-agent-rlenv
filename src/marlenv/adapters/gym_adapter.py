import sys
from dataclasses import dataclass
from typing import Any, Literal

import cv2
import gymnasium as gym  # pyright: ignore[reportMissingImports]
import numpy as np
import numpy.typing as npt
from gymnasium import Env, spaces  # pyright: ignore[reportMissingImports]
from typing_extensions import override

from marlenv import ContinuousSpace, DiscreteSpace, MARLEnv, MultiDiscreteSpace, Observation, State, Step


@dataclass
class Gym(MARLEnv[npt.NDArray]):
    """Wrap a Gymnasium environment in a `MARLEnv` interface."""

    def __init__(self, env: Env[npt.NDArray, Any] | str, **kwargs):
        if isinstance(env, str):
            env = gym.make(env, render_mode="rgb_array", **kwargs)
        if env.observation_space.shape is None:
            raise NotImplementedError("Observation space must have a shape")
        match env.action_space:
            case spaces.Discrete() as s:
                space = DiscreteSpace.action(int(s.n)).repeat(1)
            case spaces.Box() as s:
                low = s.low.astype(np.float32)
                high = s.high.astype(np.float32)
                if not isinstance(low, np.ndarray):
                    low = np.full(s.shape, s.low, dtype=np.float32)
                if not isinstance(high, np.ndarray):
                    high = np.full(s.shape, s.high, dtype=np.float32)
                space = ContinuousSpace(low, high, labels=[f"Action {i}" for i in range(s.shape[0])]).repeat(1)
            case other:
                raise NotImplementedError(f"Action space {other} not supported")
        super().__init__(1, space, env.observation_space.shape, (1,))
        self._gym_env = env
        self._last_obs = None

    @property
    @override
    def name(self):
        if self._gym_env.unwrapped.spec is not None:
            return self._gym_env.unwrapped.spec.id
        return "gym-no-id"

    def get_observation(self):
        if self._last_obs is None:
            raise ValueError("No observation available. Call reset() first.")
        return self._last_obs

    def step(self, action):
        action = np.array(action)
        obs, reward, done, truncated, info = self._gym_env.step(action[0])
        self._last_obs = Observation(
            np.array([obs], dtype=np.float32),
            self.available_actions(),
        )
        return Step(
            np.array(action),
            self._last_obs,
            self.get_state(),
            np.array([reward]),
            done,
            truncated,
            info,
        )

    def get_state(self):
        return State(np.zeros(1, dtype=np.float32))

    def reset(self, *, seed: int | None = None):
        obs_data, _info = self._gym_env.reset(seed=seed)
        self._last_obs = Observation(
            np.array([obs_data], dtype=np.float32),
            self.available_actions(),
        )
        return self._last_obs, self.get_state()

    def get_image(self):
        image = np.array(self._gym_env.render())
        if sys.platform in ("linux", "linux2"):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return np.array(image, dtype=np.uint8)

    def seed(self, seed_value: int):
        self._gym_env.reset(seed=seed_value)


def make(env_id: str, **kwargs):
    """Create a Gymnasium-based `MARLEnv` from an environment id."""
    gym_env = gym.make(env_id, render_mode="rgb_array", **kwargs)
    return Gym(gym_env)


@dataclass
class ToGym(gym.Env[npt.NDArray[np.float32], Any]):
    """
    Helper to turn a single-agent `MARLEnv` into a `gymnasium.Env`.

    Note:
    -----
    ToGym can not be serialized as JSON because gymnasium's Box type (among others) is not serializable.
    """

    on_unavailable_action: Literal["error", "random"]

    def __init__(
        self,
        env: MARLEnv[npt.NDArray],
        render_mode: Literal["human", "rgb_array", "ansi"] | None = "human",
        on_unavailable_action: Literal["error", "random"] = "random",
    ):
        assert env.n_agents == 1, "Only single-agent environments can be tunred into gymnasium environments"
        assert env.reward_space.size == 1, "Only single-objective environments can be turned into gymnasium environments"
        if len(env.observation_shape) > 1 and env.extras_size > 0:
            raise NotImplementedError("Environments with >1 dim and extras are not supported")
        gym.Env.__init__(self)
        self._env = env
        self.render_mode = render_mode
        self.on_unavailable_action = on_unavailable_action
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=env.observation_shape, dtype=np.float32)
        match env.action_space:
            case DiscreteSpace() | MultiDiscreteSpace() as s:
                self.action_space = spaces.Discrete(s.size)
            case ContinuousSpace() as s:
                self.action_space = spaces.Box(
                    low=s.low.astype(np.float32),
                    high=s.high.astype(np.float32),
                    shape=s.shape,
                    dtype=np.float32,
                )
            case other:
                raise NotImplementedError(f"Action space {other} not supported")

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if isinstance(action, int):
            available = self._env.available_actions()[0]
            if not available[action]:
                if self.on_unavailable_action == "random":
                    action = self._env.sample_action()
                else:
                    raise NotImplementedError("Unavailable action selected. To allow this, set on_unavailable_action to 'random'.")
        action = np.array(action)
        action = np.expand_dims(action, axis=0)
        step = self._env.step(action)
        data = np.concat([step.obs.data[0], step.obs.extras[0]])
        return (
            data,
            step.reward.item(),
            step.done,
            step.truncated,
            step.info,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._env.seed(seed)
        obs, _ = self._env.reset(seed=seed)
        data = obs.data.squeeze(0)
        extra = obs.extras.squeeze(0)
        return np.concat([data, extra]), {}

    def render(self):
        if self.render_mode is None:
            return
        if self.render_mode in ("human", "rgb_array"):
            return self._env.get_image()
        if self.render_mode == "ansi":
            return str(self._env)
        raise NotImplementedError(f"Render mode {self.render_mode} not supported")
