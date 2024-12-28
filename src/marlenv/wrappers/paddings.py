import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from marlenv.models import Observation, ActionSpace
from .rlenv_wrapper import RLEnvWrapper, MARLEnv
from typing_extensions import TypeVar


A = TypeVar("A", default=npt.NDArray)
AS = TypeVar("AS", bound=ActionSpace, default=ActionSpace)


@dataclass
class PadExtras(RLEnvWrapper[A, AS]):
    """RLEnv wrapper that adds extra zeros at the end of the observation extras."""

    n: int

    def __init__(self, env: MARLEnv[A, AS], n_added: int):
        assert len(env.extra_shape) == 1, "PadExtras only accepts 1D extras"
        meanings = env.extras_meanings + [f"Padding-{i}" for i in range(n_added)]
        super().__init__(
            env,
            extra_shape=(env.extra_shape[0] + n_added,),
            extra_meanings=meanings,
        )
        self.n = n_added

    def step(self, actions):
        step = super().step(actions)
        step.obs = self._add_extras(step.obs)
        return step

    def reset(self):
        obs, state = super().reset()
        obs = self._add_extras(obs)
        return obs, state

    def _add_extras(self, obs: Observation):
        obs.extras = np.concatenate([obs.extras, np.zeros((obs.n_agents, self.n), dtype=np.float32)], axis=-1)
        return obs


@dataclass
class PadObservations(RLEnvWrapper[A, AS]):
    """RLEnv wrapper that adds extra zeros at the end of the observation data."""

    def __init__(self, env: MARLEnv[A, AS], n_added: int) -> None:
        assert len(env.observation_shape) == 1, "PadObservations only accepts 1D observations"
        super().__init__(env, observation_shape=(env.observation_shape[0] + n_added,))
        self.n = n_added

    def step(self, actions):
        step = super().step(actions)
        step.obs = self._add_obs(step.obs)
        return step

    def reset(self):
        obs, state = super().reset()
        obs = self._add_obs(obs)
        return obs, state

    def _add_obs(self, obs: Observation):
        obs.data = np.concatenate([obs.data, np.zeros((obs.n_agents, self.n), dtype=np.float32)], axis=-1)
        return obs
