import os
import json
import numpy as np
from .rlenv_wrapper import RLEnvWrapper, RLEnv, Observation

class LogWrapper(RLEnvWrapper):
    """This class is a base class for other loggers and should not be used without subclassing."""
    def __init__(self, env: RLEnv, directory: str, file_name: str) -> None:
        super().__init__(env)
        self._directory = directory
        self._file_name = file_name
        self._episode_num = -1
        self._logs = []

    def _save(self):
        log_path = os.path.join(self._directory, f"{self._episode_num}", self._file_name)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(self._logs, f)
        self._logs = []

    def reset(self) -> Observation:
        self._episode_num += 1
        return super().reset()

    def step(self, actions: np.ndarray[np.int32]) -> tuple[Observation, float, bool, dict]:
        obs, reward, done, info = super().step(actions)
        if done:
            self._save()
        return obs, reward, done, info
    

class LogActionWrapper(LogWrapper):
    def __init__(self, env: RLEnv, directory: str):
        super().__init__(env, directory, "actions.json")

    def step(self, actions):
        match actions:
            case np.ndarray(): actions = actions.tolist()
            case list(): pass
            case other: raise TypeError(f"actions must be a list or numpy array, not {type(other)}")
        self._logs.append(actions)
        return super().step(actions)
    
    def summary(self) -> dict[str, str]:
        return {
            **super().summary(),
            "actions_logdir": self._directory
        }
    

class LogObservationWrapper(LogWrapper):
    def __init__(self, env: RLEnv, directory: str):
        super().__init__(env, directory, "observations.json")

    def reset(self):
        obs = super().reset()
        self._logs.append(obs)
        return obs

    def step(self, actions):
        obs, reward, done, info = super().step(actions)
        self._logs.append(obs.to_json())
        return obs, reward, done, info

    def summary(self) -> dict[str, str]:
        return {
            **super().summary(),
            "obs_logdir": self._directory
        }