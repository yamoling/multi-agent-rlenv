import numpy as np
import numpy.typing as npt
from typing import Literal, overload
from smac.env import StarCraft2Env

from marlenv.models import RLEnv, Observation, DiscreteActionSpace


class SMAC(RLEnv[DiscreteActionSpace]):
    """Wrapper for the SMAC environment to work with this framework"""

    @overload
    def __init__(self, map_name: str) -> None: ...

    @overload
    def __init__(self, env: StarCraft2Env) -> None: ...

    def __init__(self, env_or_map_name):  # type: ignore
        match env_or_map_name:
            case StarCraft2Env():
                self._env = env_or_map_name
                map_name = env_or_map_name.map_name
            case str():
                map_name = env_or_map_name
                self._env = StarCraft2Env(map_name=map_name)
            case other:
                raise ValueError(f"Invalid argument type: {type(other)}")
        self._env = StarCraft2Env(map_name=map_name)
        action_space = DiscreteActionSpace(self._env.n_agents, self._env.n_actions)
        self._env_info = self._env.get_env_info()
        super().__init__(
            action_space=action_space,
            observation_shape=(self._env_info["obs_shape"],),
            state_shape=(self._env_info["state_shape"],),
        )
        self._seed = self._env.seed()
        self.name = f"smac-{self._env.map_name}"

    def reset(self):
        obs, state = self._env.reset()
        obs = Observation(np.array(obs), self.available_actions(), state)
        return obs

    def get_state(self):
        return self._env.get_state()

    def step(self, actions):
        reward, done, info = self._env.step(actions)
        obs = Observation(np.array(self._env.get_obs()), self.available_actions(), self.get_state())
        return obs, np.array([reward], np.float32), done, False, info

    def available_actions(self) -> npt.NDArray[np.bool_]:
        return np.array(self._env.get_avail_actions()) == 1

    def render(self, mode: Literal["human", "rgb_array"] = "human"):
        return self._env.render(mode)

    def seed(self, seed_value: int):
        self._env = StarCraft2Env(map_name=self._env.map_name, seed=seed_value)
