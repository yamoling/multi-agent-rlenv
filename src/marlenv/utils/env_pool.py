import random
from dataclasses import dataclass
from typing import Collection, TypeVar

from marlenv.wrappers.rlenv_wrapper import MARLEnv, RLEnvWrapper

A = TypeVar("A")


@dataclass
class EnvPool(RLEnvWrapper[A]):
    """Randomly selects an environment from the pool on reset."""

    envs: list[MARLEnv[A]]

    def __init__(self, envs: Collection[MARLEnv[A]]):
        assert len(self.envs) > 0, "EnvPool must contain at least one environment"
        for env in self.envs[1:]:
            assert env.has_same_inouts(self.envs[0]), "All environments must have the same inputs and outputs"
        super().__init__(self.envs[0])

    def seed(self, seed_value: int):
        random.seed(seed_value)
        for env in self.envs:
            env.seed(seed_value)

    def reset(self, *, seed: int | None = None):
        self.wrapped = random.choice(self.envs)
        return super().reset()
