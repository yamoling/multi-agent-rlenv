from pettingzoo import ParallelEnv
from rlenv.models import RLEnv, Observation
import numpy as np


class PettingZooAdapter(RLEnv):
    def __init__(self, env: ParallelEnv) -> None:
        super().__init__()
        self._env = env
        self.agents = env.possible_agents

    @property
    def n_actions(self) -> int:
        return self._env.action_space(self.agents[0]).n

    @property
    def n_agents(self) -> int:
        return len(self.agents)

    @property
    def state_shape(self):
        return (0,)

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return self._env.observation_space(self.agents[0])._shape

    def get_state(self):
        try:
            return self._env.state()
        except NotImplementedError:
            return np.array([])

    def step(self, actions: np.ndarray[np.int32]) -> tuple[Observation, float, bool, dict]:
        action_dict = dict(zip(self.agents, actions))
        obs, reward, term, trunc, info = self._env.step(action_dict)
        obs_data = np.array([v for v in obs.values()])
        reward = sum(reward.values())
        observation = Observation(obs_data, self.available_actions(), self.get_state())
        done = all(t1 or t2 for t1, t2 in zip(term, trunc))
        return observation, reward, done, info

    def reset(self) -> Observation:
        obs = self._env.reset()
        obs_data = np.array([v for v in obs.values()])
        return Observation(obs_data, self.available_actions(), self.get_state())

    def available_actions(self):
        return np.ones(self.n_actions)

    def seed(self, seed_value: int):
        self._env.seed(seed_value)

    def render(self):
        return self._env.render()
