import numpy as np
from marlenv import MARLEnv, Observation, DiscreteSpace, State, Step


class MatrixGame(MARLEnv):
    """Single step matrix game used in QTRAN, Qatten and QPLEX papers."""

    N_AGENTS = 2
    UNIT_DIM = 1
    OBS_SHAPE = (1,)
    STATE_SIZE = UNIT_DIM * N_AGENTS

    QPLEX_PAYOFF_MATRIX = [
        [8.0, -12.0, -12.0],
        [-12.0, 0.0, 0.0],
        [-12.0, 0.0, 0.0],
    ]

    def __init__(self, payoff_matrix: list[list[float]]):
        action_names = [chr(ord("A") + i) for i in range(len(payoff_matrix[0]))]
        super().__init__(
            2,
            action_space=DiscreteSpace(len(payoff_matrix[0]), action_names).repeat(2),
            observation_shape=MatrixGame.OBS_SHAPE,
            state_shape=(MatrixGame.STATE_SIZE,),
        )
        self.current_step = 0
        self.payoffs = payoff_matrix

    def reset(self):
        self.current_step = 0
        return self.get_observation(), self.get_state()

    def get_observation(self):
        return Observation(
            np.array([[self.current_step]] * MatrixGame.N_AGENTS, np.float32),
            self.available_actions(),
        )

    def step(self, action):
        action = list(action)
        self.current_step += 1
        return Step(self.get_observation(), self.get_state(), self.payoffs[action[0]][action[1]], True)

    def render(self):
        return

    def get_state(self):
        return State(np.zeros((MatrixGame.STATE_SIZE,), np.float32))

    def seed(self, seed_value):
        return
