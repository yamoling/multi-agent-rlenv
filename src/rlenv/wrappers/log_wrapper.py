import os
import json
from .rlenv_wrapper import RLEnvWrapper, RLEnv

class ActionLogWrapper(RLEnvWrapper):
    def __init__(self, env: RLEnv, directory=""):
        super().__init__(env)
        self._actions = []
        self._episode_num = -1
        self._directory = directory

    def reset(self):
        if self._episode_num >= 0:
            log_path = os.path.join(self._directory, f"{self._episode_num}.json")
            with open(log_path, "w") as f:
                json.dump(self._actions, f)
        self._actions = []
        self._episode_num += 1
        return super().reset()
    
    def step(self, actions):
        self._actions.append(actions)
        return super().step(actions)