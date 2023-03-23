import numpy as np
from .rlenv_wrapper import RLEnvWrapper, RLEnv

class ForceActionWrapper(RLEnvWrapper):
    def __init__(self, env: RLEnv, forced_actions: dict[int, int]) -> None:
        super().__init__(env)
        self._forced_actions = forced_actions
        self._available_actions_mask = np.ones((self.n_agents, self.n_actions), dtype=np.int32)
        for index, action in forced_actions.items():
            self._available_actions_mask[index, :] = 0
            self._available_actions_mask[index, action] = 1

    def get_avail_actions(self):
        available = super().get_avail_actions()
        return available * self._available_actions_mask

    def step(self, actions):
        obs, *data = super().step(actions)
        obs.available_actions = obs.available_actions * self._available_actions_mask
        return obs, *data
    
    def reset(self):
        obs = super().reset()
        obs.available_actions = obs.available_actions * self._available_actions_mask
        return obs

    def kwargs(self) -> dict[str,]:
        return { "forced_actions": self._forced_actions }