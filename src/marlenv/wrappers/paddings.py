import numpy as np
from dataclasses import dataclass
from marlenv.models import Observation, ActionSpace
from .rlenv_wrapper import RLEnvWrapper, MARLEnv
from typing import TypeVar

A = TypeVar("A", bound=ActionSpace)
D = TypeVar("D")
S = TypeVar("S")
R = TypeVar("R", bound=float | np.ndarray)


@dataclass
class PadExtras(RLEnvWrapper[A, D, S, R]):
    """RLEnv wrapper that adds extra zeros at the end of the observation extras."""

    n: int

    def __init__(self, env: MARLEnv[A, D, S, R], n_added: int):
        assert len(env.extra_shape) == 1, "PadExtras only accepts 1D extras"
        super().__init__(env, extra_shape=(env.extra_shape[0] + n_added,))
        self.n = n_added

    def step(self, actions):
        obs, *data = super().step(actions)
        return self._add_extras(obs), *data

    def reset(self):
        return self._add_extras(super().reset())

    def _add_extras(self, obs: Observation):
        obs.extras = np.concatenate([obs.extras, np.zeros((obs.n_agents, self.n), dtype=np.float32)], axis=-1)
        return obs


class PadObservations(RLEnvWrapper[A, D, S, R]):
    """RLEnv wrapper that adds extra zeros at the end of the observation data."""

    def __init__(self, env: MARLEnv[A, D, S, R], n_added: int) -> None:
        assert len(env.observation_shape) == 1, "PadObservations only accepts 1D observations"
        super().__init__(env, observation_shape=(env.observation_shape[0] + n_added,))
        self.n = n_added

    def step(self, actions):
        obs, *data = super().step(actions)
        return self._add_obs(obs), *data

    def reset(self):
        return self._add_obs(super().reset())

    def _add_obs(self, obs: Observation):
        obs.data = np.concatenate([obs.data, np.zeros((obs.n_agents, self.n), dtype=np.float32)], axis=-1)
        return obs
