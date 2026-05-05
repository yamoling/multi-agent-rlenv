from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from marlenv import MARLEnv, Observation, Space

from .rlenv_wrapper import RLEnvWrapper

A = TypeVar("A", bound=Space)


@dataclass
class PotentialShaping(RLEnvWrapper[A], ABC):
    """
    Potential shaping for the Laser Learning Environment (LLE).

    https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
    """

    gamma: float

    def __init__(
        self,
        env: MARLEnv,
        gamma: float = 1.0,
        extra_shape: tuple[int] | None = None,
    ):
        super().__init__(env, extra_shape=extra_shape)
        self.gamma = gamma
        self._current_potential = self.compute_potential()

    def add_extras(self, obs: Observation) -> Observation:
        """Add the extras related to potential shaping. Does nothing by default."""
        return obs

    def reset(self, *, seed: int | None = None):
        obs, state = super().reset()
        self._current_potential = self.compute_potential()
        return self.add_extras(obs), state

    def step(self, action):
        prev_potential = self._current_potential
        step = super().step(action)

        self._current_potential = self.compute_potential()
        shaped_reward = self.gamma * self._current_potential - prev_potential
        step.obs = self.add_extras(step.obs)
        step.reward += shaped_reward
        return step

    @abstractmethod
    def compute_potential(self) -> float | npt.NDArray[np.float32]:
        """Compute the potential of the current state of the environment."""
