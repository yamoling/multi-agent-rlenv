from enum import IntEnum
import cv2
import marlenv
import numpy as np
import numpy.typing as npt
from typing import Sequence
from marlenv import Observation, State, DiscreteSpace, Step

PAYOFF_INITIAL = [[0, 0], [0, 0]]
PAYOFF_2A = [[7, 7], [7, 7]]
PAYOFF_2B = [[0, 1], [1, 8]]


class TwoStepsState(IntEnum):
    INITIAL = 0
    STATE_2A = 1
    STATE_2B = 2
    END = 3

    def one_hot(self):
        res = np.zeros((4,), dtype=np.float32)
        res[self.value] = 1
        return res

    @staticmethod
    def from_one_hot(x: np.ndarray):
        for s in TwoStepsState:
            if x[s.value] == 1:
                return s
        raise ValueError()


class TwoStepsGame(marlenv.MARLEnv):
    """
    Two-steps game used in QMix paper (https://arxiv.org/pdf/1803.11485.pdf, section 5)
    to demonstrate its superior representationability compared to VDN.
    """

    def __init__(self):
        self.state = TwoStepsState.INITIAL
        self._identity = np.identity(2, dtype=np.float32)
        super().__init__(
            2,
            DiscreteSpace(2).repeat(2),
            observation_shape=(self.state.one_hot().shape[0] + 2,),
            state_shape=self.state.one_hot().shape,
        )

    def reset(self):
        self.state = TwoStepsState.INITIAL
        return self.observation(), self.get_state()

    def step(self, action: npt.NDArray[np.int32] | Sequence):
        match self.state:
            case TwoStepsState.INITIAL:
                # In the initial step, only agent 0's actions have an influence on the state
                payoffs = PAYOFF_INITIAL
                if action[0] == 0:
                    self.state = TwoStepsState.STATE_2A
                elif action[0] == 1:
                    self.state = TwoStepsState.STATE_2B
                else:
                    raise ValueError(f"Invalid action: {action[0]}")
            case TwoStepsState.STATE_2A:
                payoffs = PAYOFF_2A
                self.state = TwoStepsState.END
            case TwoStepsState.STATE_2B:
                payoffs = PAYOFF_2B
                self.state = TwoStepsState.END
            case TwoStepsState.END:
                raise ValueError("Episode is already over")
        reward = payoffs[action[0]][action[1]]
        done = self.state == TwoStepsState.END
        return Step(self.observation(), self.get_state(), reward, done, False)

    def get_state(self):
        return State(self.state.one_hot())

    def observation(self):
        obs_data = np.array([self.state.one_hot(), self.state.one_hot()])
        extras = self._identity
        return Observation(obs_data, self.available_actions(), extras)

    def render(self):
        print(self.state)

    def get_image(self):
        state = self.state.one_hot()
        img = cv2.cvtColor(state, cv2.COLOR_GRAY2BGR)
        return np.array(img, dtype=np.uint8)

    def set_state(self, state: State):
        self.state = TwoStepsState.from_one_hot(state.data)
