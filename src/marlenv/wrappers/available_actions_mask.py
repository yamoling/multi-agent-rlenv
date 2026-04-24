from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeVar

from marlenv.models import Space

from .rlenv_wrapper import MARLEnv, RLEnvWrapper

AS = TypeVar("AS", bound=Space, default=Space)


@dataclass
class AvailableActionsMask(RLEnvWrapper[AS]):
    """Permanently masks a subset of the available actions."""

    action_mask: npt.NDArray[np.bool]

    def __init__(self, env: MARLEnv[AS], action_mask: npt.NDArray[np.bool] | Sequence[bool] | Sequence[Sequence[bool]]):
        super().__init__(env)
        if not isinstance(action_mask, np.ndarray):
            action_mask = np.array(action_mask, dtype=np.bool)
        assert len(action_mask.shape) <= 2, "Action mask must be a 1D (actions) or 2D (agent-wise actions) array."
        assert action_mask.shape[-1] == env.n_actions, "Action mask must have the same number of actions as the environment."
        if action_mask.ndim == 1:
            action_mask = np.tile(action_mask, (env.n_agents, 1))
        assert action_mask.shape[0] == env.n_agents, "Action mask must have the same number of agents as the environment."
        n_available_action_per_agent = action_mask.sum(axis=-1)
        assert np.all(n_available_action_per_agent >= 1), "At least one action must be available for each agent."
        self.action_mask = action_mask

    def reset(self, *, seed: int | None = None):
        obs, state = self.wrapped.reset()
        obs.available_actions = self.available_actions()
        return obs, state

    def step(self, action):
        step = self.wrapped.step(action)
        step.obs.available_actions = self.available_actions()
        return step

    def available_actions(self):
        return self.action_mask & self.wrapped.available_actions()

    def get_observation(self):
        obs = super().get_observation()
        obs.available_actions = self.available_actions()
        return obs
