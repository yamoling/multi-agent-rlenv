from dataclasses import dataclass

import numpy as np
from typing_extensions import TypeVar

from marlenv.models import Observation

from .rlenv_wrapper import MARLEnv, RLEnvWrapper

A = TypeVar("A")


@dataclass
class PadExtras(RLEnvWrapper[A]):
    """RLEnv wrapper that adds extra zeros at the end of the observation extras."""

    n: int

    def __init__(self, env: MARLEnv[A], n_added: int, label: str = "Padding"):
        assert len(env.extras_shape) == 1, "PadExtras only accepts 1D extras"
        super().__init__(
            env,
            extra_shape=(env.extras_shape[0] + n_added,),
            extra_meanings=env.extras_meanings + [f"{label}-{i}" for i in range(n_added)],
        )
        self.n = n_added
        self.padding = np.zeros((self.n_agents, self.n), dtype=np.float32)

    def step(self, action):
        step = super().step(action)
        step.obs.add_extra(self.padding)
        return step

    def reset(self, *, seed: int | None = None):
        obs, state = super().reset()
        obs.add_extra(self.padding)
        return obs, state


@dataclass
class PadObservations(RLEnvWrapper[A]):
    """RLEnv wrapper that adds extra zeros at the end of the observation data."""

    def __init__(self, env: MARLEnv[A], n_added: int) -> None:
        assert len(env.observation_shape) == 1, "PadObservations only accepts 1D observations"
        super().__init__(env, observation_shape=(env.observation_shape[0] + n_added,))
        self.n = n_added

    def step(self, action):
        step = super().step(action)
        step.obs = self._add_obs(step.obs)
        return step

    def reset(self, *, seed: int | None = None):
        obs, state = super().reset()
        obs = self._add_obs(obs)
        return obs, state

    def _add_obs(self, obs: Observation):
        obs.data = np.concatenate([obs.data, np.zeros((obs.n_agents, self.n), dtype=np.float32)], axis=-1)
        return obs
