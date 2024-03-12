from abc import abstractmethod
from dataclasses import dataclass
from typing import Any
import numpy as np
from .observation import Observation
from .rl_env import RLEnv, A


@dataclass
class MultiObjectiveRLEnv(RLEnv[A]):
    n_objectives: int
    reward_shape: tuple[int, ...]

    def __init__(
        self,
        n_objectives: int,
        action_space: A,
        observation_shape: tuple[int, ...],
        state_shape: tuple[int, ...],
        extra_feature_shape: tuple[int, ...] = (0,),
    ):
        RLEnv.__init__(
            self,
            action_space,
            observation_shape,
            state_shape,
            extra_feature_shape,
        )
        self.n_objectives = n_objectives
        self.reward_shape = (n_objectives,)

    @abstractmethod
    def step(self, actions: list[int] | np.ndarray) -> tuple[Observation, list[float], bool, bool, dict[str, Any]]:
        """Perform a step in the environment.

        Returns:
        - observation: The new observation of the environment.
        - rewards: the list of rewards for each objective
        - done: Whether the episode is over
        - truncated: Whether the episode is truncated
        - info: Extra information
        """
