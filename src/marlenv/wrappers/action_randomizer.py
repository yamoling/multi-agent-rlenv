from dataclasses import dataclass
from typing import TypeVar, cast

import numpy as np
import numpy.typing as npt

from .rlenv_wrapper import MARLEnv, RLEnvWrapper

A = TypeVar("A", bound=npt.NDArray)


@dataclass
class ActionRandomizer(RLEnvWrapper[A]):
    p: npt.NDArray[np.float32]

    def __init__(self, env: MARLEnv[A], p: float | list[float] | npt.NDArray[np.float32]):
        super().__init__(env)
        if isinstance(p, (float, int)):
            p = np.full(self.n_agents, p, dtype=np.float32)
        elif isinstance(p, list):
            p = np.array(p, dtype=np.float32)
        assert p.shape == (self.n_agents,), "p must be a float or a list/array of floats with length equal to n_agents"
        assert np.all((0.0 <= p) & (p <= 1.0)), "Probabilities must be between 0 and 1"
        self.p = p

    def step(self, action):
        replacements = self.action_space.sample(self.available_actions())
        randomize_mask = np.random.rand(self.n_agents) < self.p
        action = np.where(randomize_mask, replacements, action)
        return super().step(cast(A, action))

    def seed(self, seed_value: int):
        np.random.seed(seed_value)
        super().seed(seed_value)
