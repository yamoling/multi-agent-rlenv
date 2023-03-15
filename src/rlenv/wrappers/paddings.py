import numpy as np
from .rlenv_wrapper import RLEnvWrapper, RLEnv, Observation

class PadExtras(RLEnvWrapper):
    """RLEnv wrapper that adds extra zeros at the end of the observation extras."""
    def __init__(self, env: RLEnv, n_added: int) -> None:
        assert len(env.extra_feature_shape) == 1, "PadExtras only accepts 1D extras"
        super().__init__(env)
        self.n = n_added
        self._extras_shape = (env.extra_feature_shape[0] + n_added, )

    @property
    def extra_feature_shape(self):
        return self._extras_shape

    def step(self, actions):
        obs, *data = super().step(actions)
        return self._add_extras(obs), *data

    def reset(self):
        return self._add_extras(super().reset())
    
    def _add_extras(self, obs: Observation) -> Observation:
        obs.extras = np.concatenate([obs.extras, np.zeros((obs.n_agents, self.n))], axis=-1)
        return obs

class PadObservations(RLEnvWrapper):
    """RLEnv wrapper that adds extra zeros at the end of the observation data."""
    def __init__(self, env: RLEnv, n_added: int) -> None:
        assert len(env.observation_shape) == 1, "PadObservations only accepts 1D observations"
        super().__init__(env)
        self.n = n_added
        self._obs_shape = (env.observation_shape[0] + n_added, )

    @property
    def observation_shape(self):
        return self._obs_shape

    def step(self, actions):
        obs, *data = super().step(actions)
        return self._add_obs(obs), *data

    def reset(self):
        return self._add_obs(super().reset())
    
    def _add_obs(self, obs: Observation) -> Observation:
        obs.data = np.concatenate([obs.data, np.zeros((obs.n_agents, self.n))], axis=-1)
        return obs