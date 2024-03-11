import numpy as np
from rlenv import RLEnv, Observation, DiscreteActionSpace, MultiObjectiveRLEnv as MORLenv


class MockEnv(RLEnv[DiscreteActionSpace]):
    OBS_SIZE = 42
    N_ACTIONS = 5
    END_GAME = 30
    REWARD_STEP = 1
    UNIT_STATE_SIZE = 1

    def __init__(self, n_agents: int, n_objectives: int = 1) -> None:
        super().__init__(
            DiscreteActionSpace(n_agents, MockEnv.N_ACTIONS),
            (MockEnv.OBS_SIZE,),
            (n_agents * MockEnv.UNIT_STATE_SIZE,),
            reward_size=n_objectives,
        )
        self.t = 0
        self.actions_history = []

    @property
    def unit_state_size(self):
        return MockEnv.UNIT_STATE_SIZE

    def reset(self):
        self.t = 0
        return self.observation()

    def observation(self):
        obs_data = np.array(
            [np.arange(self.t + agent, self.t + agent + MockEnv.OBS_SIZE) for agent in range(self.n_agents)],
            dtype=np.float32,
        )
        return Observation(obs_data, self.available_actions(), self.get_state())

    def get_state(self):
        return np.full((self.n_agents * MockEnv.UNIT_STATE_SIZE,), self.t, dtype=np.float32)

    def render(self, mode: str = "human"):
        return

    def step(self, action):
        self.t += 1
        self.actions_history.append(action)
        return (
            self.observation(),
            [MockEnv.REWARD_STEP] * self.reward_size,
            self.t >= MockEnv.END_GAME,
            False,
            {},
        )


class MockMOEnv(MORLenv, MockEnv):
    def __init__(self, n_agents, n_objectives):
        MockEnv.__init__(self, n_agents)
        MORLenv.__init__(
            self,
            n_objectives,
            DiscreteActionSpace(n_agents, MockEnv.N_ACTIONS),
            (MockEnv.OBS_SIZE,),
            (n_agents * MockEnv.UNIT_STATE_SIZE,),
        )

    def step(self, action):
        obs, reward, done, truncated, info = MockEnv.step(self, action)
        return obs, [reward] * self.n_objectives, done, truncated, info
