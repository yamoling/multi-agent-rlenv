import numpy as np
from marlenv.models import MARLEnv, ActionSpace
from dataclasses import dataclass
from .rlenv_wrapper import RLEnvWrapper

from typing_extensions import TypeVar

A = TypeVar("A", default=np.ndarray)
AS = TypeVar("AS", bound=ActionSpace, default=ActionSpace)


@dataclass
class AgentId(RLEnvWrapper[A, AS]):
    """RLEnv wrapper that adds a one-hot encoding of the agent id."""

    def __init__(self, env: MARLEnv[A, AS]):
        assert len(env.extra_shape) == 1, "AgentIdWrapper only works with single dimension extras"
        meanings = env.extras_meanings + [f"Agent ID-{i}" for i in range(env.n_agents)]
        super().__init__(env, extra_shape=(env.n_agents + env.extra_shape[0],), extra_meanings=meanings)
        self._identity = np.identity(env.n_agents, dtype=np.float32)

    def step(self, actions):
        step = super().step(actions)
        step.obs.add_extra(self._identity)
        return step

    def reset(self):
        obs, state = super().reset()
        obs.add_extra(self._identity)
        return obs, state
