import numpy as np
from rlenv.models import RLEnv, Observation
from .rlenv_wrapper import RLEnvWrapper


class AgentIdWrapper(RLEnvWrapper):
    """RLEnv wrapper that adds a one-hot encoding of the agent id."""

    def __init__(self, env: RLEnv) -> None:
        assert len(env.extra_feature_shape) == 1, "AgentIdWrapper only works with single dimension extras"
        super().__init__(env)
        self._identity = np.identity(env.n_agents, dtype=np.float32)

    @property
    def extra_feature_shape(self):
        return (self.n_agents + self.wrapped.extra_feature_shape[0],)

    def step(self, actions):
        obs, *data = super().step(actions)
        return self._add_one_hot(obs), *data

    def reset(self):
        return self._add_one_hot(super().reset())

    def _add_one_hot(self, observation: Observation) -> Observation:
        observation.extras = np.concatenate([observation.extras, self._identity], axis=-1)
        return observation
