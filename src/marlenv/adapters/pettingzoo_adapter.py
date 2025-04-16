from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt
from gymnasium import spaces  # pettingzoo uses gymnasium spaces
from pettingzoo import ParallelEnv

from marlenv.models import MARLEnv, Observation, State, Step, DiscreteSpace, ContinuousSpace, Space


@dataclass
class PettingZoo(MARLEnv[Space]):
    def __init__(self, env: ParallelEnv):
        aspace = env.action_space(env.possible_agents[0])
        n_agents = len(env.possible_agents)
        match aspace:
            case spaces.Discrete() as s:
                space = DiscreteSpace.action(int(s.n)).repeat(n_agents)

            case spaces.Box() as s:
                low = s.low.astype(np.float32)
                high = s.high.astype(np.float32)
                if not isinstance(low, np.ndarray):
                    low = np.full(s.shape, s.low, dtype=np.float32)
                if not isinstance(high, np.ndarray):
                    high = np.full(s.shape, s.high, dtype=np.float32)
                space = ContinuousSpace(low, high=high).repeat(n_agents)
            case other:
                raise NotImplementedError(f"Action space {other} not supported")

        obs_space = env.observation_space(env.possible_agents[0])
        if obs_space.shape is None:
            raise NotImplementedError("Only discrete observation spaces are supported")
        self._pz_env = env
        env.reset()
        super().__init__(n_agents, space, obs_space.shape, self.get_state().shape)
        self.agents = env.possible_agents
        self.last_observation = None

    def get_state(self):
        try:
            return self._pz_env.state()
        except NotImplementedError:
            return np.array([0])

    def step(self, actions: npt.NDArray | Sequence):
        action_dict = dict(zip(self.agents, actions))
        obs, reward, term, trunc, info = self._pz_env.step(action_dict)
        obs_data = np.array([v for v in obs.values()])
        reward = np.sum([r for r in reward.values()], keepdims=True)
        self.last_observation = Observation(obs_data, self.available_actions())
        state = State(self.get_state())
        return Step(self.last_observation, state, reward, any(term.values()), any(trunc.values()), info)

    def reset(self):
        obs = self._pz_env.reset()[0]
        obs_data = np.array([v for v in obs.values()])
        self.last_observation = Observation(obs_data, self.available_actions(), self.get_state())
        return self.last_observation

    def get_observation(self):
        if self.last_observation is None:
            raise ValueError("No observation available. Call reset() first.")
        return self.last_observation

    def seed(self, seed_value: int):
        self._pz_env.reset(seed=seed_value)

    def render(self, *_):
        return self._pz_env.render()
