import random
from dataclasses import dataclass
from typing import Collection, TypeVar

import numpy as np
import numpy.typing as npt

from marlenv.models import MARLEnv, Observation, State, Step

A = TypeVar("A")


@dataclass
class EnvPool(MARLEnv[A]):
    """Randomly selects an environment from the pool on reset."""

    envs: list[MARLEnv[A]]

    def __init__(self, envs: Collection[MARLEnv[A]]):
        self.envs = list(envs)
        assert len(self.envs) > 0, "EnvPool must contain at least one environment"
        for env in self.envs[1:]:
            assert env.has_same_inouts(self.envs[0]), "All environments must have the same inputs and outputs"
        self.current = self.envs[0]
        super().__init__(
            self.current.n_agents,
            self.current.action_space,
            self.current.observation_shape,
            self.current.state_shape,
            extras_shape=self.current.extras_shape,
            state_extra_shape=self.current.state_extra_shape,
            reward_space=self.current.reward_space,
            extras_meanings=self.current.extras_meanings,
        )

    def step(self, action: A | npt.ArrayLike) -> Step:
        return self.current.step(action)

    def seed(self, seed_value: int):
        random.seed(seed_value)
        for env in self.envs:
            env.seed(seed_value)

    def reset(self, *, seed: int | None = None):
        if seed is not None:
            self.seed(seed)
        self.current = random.choice(self.envs)
        return self.current.reset(seed=seed)

    @property
    def name(self):
        return f"EnvPool-#{self.current.name}"

    def get_observation(self) -> Observation:
        return self.current.get_observation()

    def get_state(self) -> State:
        return self.current.get_state()

    def get_image(self) -> npt.NDArray[np.uint8]:
        return self.current.get_image()

    def render(self):
        return self.current.render()

    @property
    def agent_state_size(self) -> int:
        return self.current.agent_state_size

    def set_state(self, state: State) -> None:
        return self.current.set_state(state)

    def available_actions(self) -> npt.NDArray[np.bool]:
        return self.current.available_actions()
