import numpy as np
from marlenv import MARLEnv, Observation, DiscreteActionSpace, DiscreteSpace
# from .models.rl_env import MOMARLEnv


class MockEnv(MARLEnv[DiscreteActionSpace, np.ndarray, np.ndarray]):
    def __init__(
        self,
        n_agents: int = 4,
        obs_size: int = 42,
        n_actions: int = 5,
        end_game: int = 30,
        reward_step: int = 1,
        agent_state_size: int = 1,
        extras_size: int = 0,
    ) -> None:
        super().__init__(
            DiscreteActionSpace(n_agents, n_actions),
            (obs_size,),
            (n_agents * agent_state_size,),
            extra_feature_shape=(extras_size,),
        )
        self.obs_size = obs_size
        self.extra_size = extras_size
        self._agent_state_size = agent_state_size
        self.end_game = end_game
        self.reward_step = reward_step
        self.t = 0
        self.actions_history = []

    @property
    def agent_state_size(self):
        return self._agent_state_size

    def reset(self):
        self.t = 0
        return self.observation()

    def observation(self):
        obs_data = np.array(
            [np.arange(self.t + agent, self.t + agent + self.obs_size) for agent in range(self.n_agents)],
            dtype=np.float32,
        )
        extras = np.arange(self.n_agents * self.extra_size, dtype=np.float32).reshape((self.n_agents, self.extra_size))
        return Observation(obs_data, self.available_actions(), self.get_state(), extras)

    def get_state(self):
        return np.full((self.n_agents * self.agent_state_size,), self.t, dtype=np.float32)

    def render(self, mode: str = "human"):
        return

    def step(self, action):
        self.t += 1
        self.actions_history.append(action)
        return (
            self.observation(),
            self.reward_step,
            self.t >= self.end_game,
            False,
            {},
        )


class MOMockEnv(MARLEnv[DiscreteActionSpace, np.ndarray, np.ndarray]):
    """Multi-Objective Mock Environment"""

    def __init__(
        self,
        n_agents: int = 4,
        n_objectives: int = 2,
        obs_size: int = 42,
        n_actions: int = 5,
        end_game: int = 30,
        reward_step: int = 1,
        agent_state_size: int = 1,
        extras_size: int = 0,
    ) -> None:
        super().__init__(
            DiscreteActionSpace(n_agents, n_actions),
            (obs_size,),
            (n_agents * agent_state_size,),
            extra_feature_shape=(extras_size,),
            reward_space=DiscreteSpace(n_objectives),
        )
        self.obs_size = obs_size
        self.extra_size = extras_size
        self._agent_state_size = agent_state_size
        self.end_game = end_game
        self.reward_step = reward_step
        self.t = 0
        self.actions_history = []

    @property
    def agent_state_size(self):
        return self._agent_state_size

    def reset(self):
        self.t = 0
        return self.observation()

    def observation(self):
        obs_data = np.array(
            [np.arange(self.t + agent, self.t + agent + self.obs_size) for agent in range(self.n_agents)],
            dtype=np.float32,
        )
        extras = np.arange(self.n_agents * self.extra_size, dtype=np.float32).reshape((self.n_agents, self.extra_size))
        return Observation(obs_data, self.available_actions(), self.get_state(), extras)

    def get_state(self):
        return np.full((self.n_agents * self.agent_state_size,), self.t, dtype=np.float32)

    def render(self, mode: str = "human"):
        return

    def step(self, action):
        self.t += 1
        self.actions_history.append(action)
        return (
            self.observation(),
            np.full(self.reward_space.shape, self.reward_step, dtype=np.float32),
            self.t >= self.end_game,
            False,
            {},
        )
