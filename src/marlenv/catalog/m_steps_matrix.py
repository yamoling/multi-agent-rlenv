from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np

from marlenv.models import DiscreteSpace, MARLEnv, MultiDiscreteSpace, Observation, State, Step


class Action(Enum):
    TOP_LEFT = "top left"
    TOP_RIGHT = "top right"
    BOTTOM_LEFT = "bottom left"
    BOTTOM_RIGHT = "bottom right"

    @staticmethod
    def from_actions(actions: tuple[int, int]):
        match actions:
            case (0, 0):
                return Action.TOP_LEFT
            case (0, 1):
                return Action.TOP_RIGHT
            case (1, 0):
                return Action.BOTTOM_LEFT
            case (1, 1):
                return Action.BOTTOM_RIGHT
        raise ValueError(f"Invalid actions: {actions}")

    def to_tuple(self):
        match self:
            case Action.TOP_LEFT:
                return (0, 0)
            case Action.TOP_RIGHT:
                return (0, 1)
            case Action.BOTTOM_LEFT:
                return (1, 0)
            case Action.BOTTOM_RIGHT:
                return (1, 1)


@dataclass
class MStepsMatrix(MARLEnv[MultiDiscreteSpace]):
    """
    Implementation of the `m`-steps matrix game used in the MAVEN paper (https://proceedings.neurips.cc/paper_files/paper/2019/file/f816dc0acface7498e10496222e9db10-Paper.pdf).
    """

    n_steps: int

    def __init__(self, n_steps: int):
        assert n_steps > 1, "n_steps must be greater than 1, otherwise the initial payoff matrix would already be the final one"
        super().__init__(2, DiscreteSpace(2).repeat(2), (n_steps + 1,), (n_steps + 1,))
        self.n_steps = n_steps
        self._current_step = 0
        self._path: None | Literal["left", "right"] = None

    def reset(self, *, seed: int | None = None):
        self._current_step = 0
        self._path = None
        return self.get_observation(), self.get_state()

    def get_observation(self) -> Observation:
        data = np.zeros((self.n_agents, self.n_steps + 1), dtype=np.float32)
        data[:, self._current_step] = 1.0
        return Observation(data, self.available_actions())

    def get_state(self):
        data = np.zeros((self.n_steps + 1,), dtype=np.float32)
        data[self._current_step] = 1.0
        return State(data)

    def step(self, action):
        self._current_step += 1
        action_enum = Action.from_actions(tuple(np.array(action)))
        # Case 1) first step after reset
        if self._path is None:
            match action_enum:
                case Action.TOP_RIGHT | Action.BOTTOM_LEFT:
                    return Step(action, self.get_observation(), self.get_state(), 0.0, done=True)
                case Action.TOP_LEFT:
                    self._path = "left"
                case Action.BOTTOM_RIGHT:
                    self._path = "right"
            return Step(action, self.get_observation(), self.get_state(), 1.0, False)
        # Case 2) we already went to the right
        if self._path == "right":
            # At the last step, always give +1
            if self._current_step >= self.n_steps:
                return Step(action, self.get_observation(), self.get_state(), 1.0, True)
            # At any other step, only BOTTOM_RIGHT provides a reward, other actions terminate the episode
            if action_enum == Action.BOTTOM_RIGHT:
                return Step(action, self.get_observation(), self.get_state(), 1.0, False)
            return Step(action, self.get_observation(), self.get_state(), 0.0, True)
        # Case 3) we took the left path
        if self._current_step == self.n_steps:
            # Optimal path -> +4 reward
            if action_enum == Action.BOTTOM_RIGHT:
                return Step(action, self.get_observation(), self.get_state(), 4, True)
            return Step(action, self.get_observation(), self.get_state(), 1, True)
        # Any action other than TOP_LEFT terminates the episode with 0 reward
        if action_enum == Action.TOP_LEFT:
            return Step(action, self.get_observation(), self.get_state(), 1.0, False)
        return Step(action, self.get_observation(), self.get_state(), 0.0, True)
