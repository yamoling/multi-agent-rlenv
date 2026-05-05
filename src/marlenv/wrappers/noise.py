import random
from dataclasses import dataclass
from typing import Literal, Sequence, TypeVar

import numpy as np
import numpy.typing as npt

from marlenv import MARLEnv
from marlenv.models.spaces import Space

from .rlenv_wrapper import RLEnvWrapper

A = TypeVar("A", bound=Space)


@dataclass
class NoiseWrapper(RLEnvWrapper[A]):
    noise_size: int
    noise_type: Literal["one-hot", "continuous"]
    with_state_extras: bool

    def __init__(
        self,
        wrapped: MARLEnv[A],
        noise_size: int,
        noise_type: Literal["one-hot", "continuous"] = "one-hot",
        with_state_extras: bool = True,
        same_for_all_agents: bool = True,
    ):
        assert len(wrapped.extras_shape) == 1, "NoiseWrapper only supports 1D extras"
        if not same_for_all_agents:
            raise NotImplementedError("NoiseWrapper with same_for_all_agents=False is not implemented yet")
        self.noise_type = noise_type
        self.noise_size = noise_size
        self.with_state_extras = with_state_extras
        extra_meanings = wrapped.extras_meanings + [f"Noise-{i}" for i in range(noise_size)]
        state_extra_shape = wrapped.state_extra_shape
        if with_state_extras:
            assert len(state_extra_shape) == 1, "NoiseWrapper only supports 1D state extras"
            state_extra_shape = (state_extra_shape[0] + noise_size,)
        super().__init__(
            wrapped,
            extra_meanings=extra_meanings,
            extra_shape=(wrapped.extras_size + noise_size,),
            state_extra_shape=state_extra_shape,
        )
        self._episode_noise = np.zeros((self.n_agents, noise_size), dtype=np.float32)

    def reset(self, *, seed: int | None = None):
        match self.noise_type:
            case "one-hot":
                self._episode_noise = np.zeros((self.n_agents, self.noise_size), dtype=np.float32)
                self._episode_noise[:, random.randint(0, self.noise_size - 1)] = 1.0
            case "continuous":
                self._episode_noise = np.random.rand(self.noise_size).astype(np.float32)
            case other:
                raise ValueError(f"Invalid noise type: {other}")
        obs, state = super().reset(seed=seed)
        obs.add_extra(self._episode_noise)
        if self.with_state_extras:
            state.add_extra(self._episode_noise[0])
        return obs, state

    def step(self, action: npt.ArrayLike | Sequence):
        step = super().step(action)
        step.obs.add_extra(self._episode_noise)
        if self.with_state_extras:
            step.state.add_extra(self._episode_noise[0])
        return step
