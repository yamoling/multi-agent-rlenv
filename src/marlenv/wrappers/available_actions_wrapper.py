import numpy as np
from typing_extensions import TypeVar
from marlenv.models import Space, MARLEnv
from .rlenv_wrapper import RLEnvWrapper
from dataclasses import dataclass


AS = TypeVar("AS", bound=Space, default=Space)


@dataclass
class AvailableActions(RLEnvWrapper[AS]):
    """Adds the available actions (one-hot) as an extra feature to the observation."""

    def __init__(self, env: MARLEnv[AS]):
        meanings = env.extras_meanings + [f"{a} available" for a in env.action_space.labels]
        super().__init__(env, extra_shape=(env.extras_shape[0] + env.n_actions,), extra_meanings=meanings)

    def reset(self):
        obs, state = self.wrapped.reset()
        obs.add_extra(self.available_actions().astype(np.float32))
        return obs, state

    def step(self, action):
        step = self.wrapped.step(action)
        step.obs.add_extra(self.available_actions().astype(np.float32))
        return step
