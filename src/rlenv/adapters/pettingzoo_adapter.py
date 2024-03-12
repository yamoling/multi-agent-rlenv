from pettingzoo import ParallelEnv
from gymnasium import spaces  # pettingzoo uses gymnasium spaces
from rlenv.models import RLEnv, Observation, ActionSpace, DiscreteActionSpace, ContinuousActionSpace
import numpy as np
import numpy.typing as npt


class PettingZoo(RLEnv[ActionSpace]):
    def __init__(self, env: ParallelEnv):
        env.reset()
        aspace = env.action_space(env.possible_agents[0])

        match aspace:
            case spaces.Discrete() as s:
                space = DiscreteActionSpace(env.num_agents, int(s.n))

            case spaces.Box() as s:
                if len(s.shape) > 1:
                    raise NotImplementedError("Multi-dimensional action spaces not supported")

                space = ContinuousActionSpace(env.num_agents, s.shape[0], low=s.low.tolist(), high=s.high.tolist())

            case other:
                raise NotImplementedError(f"Action space {other} not supported")

        obs_space = env.observation_space(env.possible_agents[0])
        if obs_space.shape is None:
            raise NotImplementedError("Only discrete observation spaces are supported")
        self._env = env
        super().__init__(space, obs_space.shape, self.get_state().shape)
        self.agents = env.possible_agents

    def get_state(self):
        try:
            return self._env.state()
        except NotImplementedError:
            return np.array([])

    def step(self, actions: npt.NDArray[np.int32]):
        action_dict = dict(zip(self.agents, actions))
        obs, reward, term, trunc, info = self._env.step(action_dict)
        obs_data = np.array([v for v in obs.values()])
        reward = sum(reward.values())
        observation = Observation(obs_data, self.available_actions(), self.get_state())
        return observation, reward, term, trunc, info

    def reset(self) -> Observation:
        obs = self._env.reset()[0]
        obs_data = np.array([v for v in obs.values()])
        return Observation(obs_data, self.available_actions(), self.get_state())

    def available_actions(self):
        return np.ones(self.n_actions, dtype=np.float32)

    def seed(self, seed_value: int):
        self._env.reset(seed=seed_value)

    def render(self):
        return self._env.render()
