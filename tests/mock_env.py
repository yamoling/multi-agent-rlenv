import numpy as np
from rlenv import RLEnv, Observation


class MockEnv(RLEnv):
    OBS_SIZE = 10
    N_ACTIONS = 5
    END_GAME = 10

    def __init__(self, n_agents) -> None:
        super().__init__()
        self._n_agents = n_agents
        self.t = 0

    @property
    def name(self) -> str:
        return "mock-env"

    @property
    def n_actions(self) -> int:
        return 5

    @property
    def n_agents(self) -> int:
        return self._n_agents

    @property
    def observation_shape(self):
        return (MockEnv.OBS_SIZE, )

    @property
    def state_shape(self):
        return (0, )

    def reset(self):
        self.t = 0
        return self.observation()

    def observation(self):
        obs_data = np.array([np.arange(self.t + agent, self.t + agent + MockEnv.OBS_SIZE) for agent in range(self.n_agents)])
        return Observation(obs_data, self.get_avail_actions(), self.get_state())

    def get_state(self):
        return np.array([])

    def render(self, mode: str = "human"):
        return

    def step(self, action):
        self.t += 1
        return self.observation(), 1, self.t >= MockEnv.END_GAME, {}