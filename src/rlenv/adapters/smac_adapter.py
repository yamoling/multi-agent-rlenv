import numpy as np
from smac.env import StarCraft2Env

from rlenv.models import RLEnv, Observation


class SMACAdapter(RLEnv):
    """Wrapper for the SMAC environment to work with this framework"""

    def __init__(self, map_name: str, time_limit=150) -> None:
        super().__init__()
        self._env = StarCraft2Env(map_name=map_name)
        self._env_info = self._env.get_env_info()
        self._time_limit = time_limit
        self._t = 0
        self._seed = self._env.seed()

    @property
    def n_actions(self) -> int:
        return self._env.n_actions

    @property
    def n_agents(self) -> int:
        return self._env.n_agents

    @property
    def state_shape(self):
        return (self._env_info["state_shape"], )

    @property
    def observation_shape(self):
        return (self._env_info["obs_shape"], )

    @property
    def name(self) -> str:
        return f"smac-{self._env.map_name}"

    def reset(self):
        obs, state = self._env.reset()
        self._t = 0
        obs = Observation(np.array(obs), self.get_avail_actions(), state)
        return obs

    def get_state(self):
        return self._env.get_state()

    def step(self, actions):
        reward, done, info = self._env.step(actions)
        obs = Observation(np.array(self._env.get_obs()), self.get_avail_actions(), self.get_state())
        self._t += 1
        if not done and self._t >= self._time_limit:
            done = True
            info["battle_won"] = False
        return obs, reward, done, info

    def get_avail_actions(self):
        return np.array(self._env.get_avail_actions())

    def render(self, mode: str="human"):
        return self._env.render(mode)

    def seed(self, seed_value: int):
        self._env = StarCraft2Env(map_name=self._env.map_name, seed=seed_value)
