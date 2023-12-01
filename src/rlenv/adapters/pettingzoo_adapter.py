from pettingzoo import ParallelEnv
from rlenv.models import RLEnv, Observation, ActionSpace, DiscreteActionSpace
import numpy as np
import numpy.typing as npt


class PettingZoo(RLEnv[ActionSpace]):
    def __init__(self, env: ParallelEnv):
        aspace = env.action_space(0)
        if aspace.shape is None:
            raise NotImplementedError("Only discrete action spaces are supported")
        obs_space = env.observation_space(env.possible_agents[0])
        if obs_space.shape is None:
            raise NotImplementedError("Only discrete observation spaces are supported")
        super().__init__(DiscreteActionSpace(env.num_agents, aspace.shape[0]), obs_space.shape, env.state().shape)
        self._env = env
        self.agents = env.possible_agents

    def get_state(self):
        try:
            return self._env.state()
        except NotImplementedError:
            return np.array([])

    def step(self, actions: npt.NDArray[np.int32]) -> tuple[Observation, float, bool, dict]:
        action_dict = dict(zip(self.agents, actions))
        obs, reward, term, trunc, info = self._env.step(action_dict)
        obs_data = np.array([v for v in obs.values()])
        reward = sum(reward.values())
        observation = Observation(obs_data, self.available_actions(), self.get_state())
        done = all(t1 or t2 for t1, t2 in zip(term, trunc))
        return observation, reward, done, info

    def reset(self) -> Observation:
        obs = self._env.reset()
        obs_data = np.array([v.values() for v in obs], dtype=np.float32)
        return Observation(obs_data, self.available_actions(), self.get_state())

    def available_actions(self):
        return np.ones(self.n_actions, dtype=np.float32)

    def seed(self, seed_value: int):
        self._env.reset(seed=seed_value)

    def render(self):
        return self._env.render()
