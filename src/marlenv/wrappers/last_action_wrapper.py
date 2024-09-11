import numpy as np
import numpy.typing as npt
from marlenv.models import Observation, ActionSpace
from .rlenv_wrapper import RLEnvWrapper, MARLEnv
from typing import TypeVar

A = TypeVar("A", bound=ActionSpace)
D = TypeVar("D")
S = TypeVar("S")
R = TypeVar("R", bound=float | np.ndarray)


class LastAction(RLEnvWrapper[A, D, S, R]):
    """Env wrapper that adds the last action taken by the agents to the extra features."""

    def __init__(self, env: MARLEnv[A, D, S, R]):
        assert len(env.extra_feature_shape) == 1, "Adding last action is only possible with 1D extras"
        super().__init__(
            env,
            extra_feature_shape=(env.extra_feature_shape[0] + env.n_actions,),
        )

    def reset(self):
        obs = super().reset()
        return self._add_last_action(obs, None)

    def step(self, actions: npt.NDArray[np.int32]):
        obs_, r, done, truncated, info = super().step(actions)
        obs_ = self._add_last_action(obs_, actions)
        return obs_, r, done, truncated, info

    def _add_last_action(self, obs: Observation, last_actions: npt.NDArray[np.int32] | None):
        one_hot_actions = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
        if last_actions is not None:
            index = np.arange(self.n_agents)
            one_hot_actions[index, last_actions] = 1.0
        obs.extras = np.concatenate([obs.extras, one_hot_actions], axis=-1)
        return obs
