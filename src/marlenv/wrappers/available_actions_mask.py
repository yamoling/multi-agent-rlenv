import numpy as np
import numpy.typing as npt
from typing_extensions import TypeVar
from .rlenv_wrapper import MARLEnv, RLEnvWrapper
from marlenv.models import ActionSpace
from dataclasses import dataclass

A = TypeVar("A", default=npt.NDArray)
AS = TypeVar("AS", bound=ActionSpace, default=ActionSpace)


@dataclass
class AvailableActionsMask(RLEnvWrapper[A, AS]):
    """Permanently masks a subset of the available actions."""

    action_mask: npt.NDArray[np.bool_]

    def __init__(self, env: MARLEnv[A, AS], action_mask: npt.NDArray[np.bool_]):
        super().__init__(env)
        assert action_mask.shape == (env.n_agents, env.n_actions), "Action mask must have shape (n_agents, n_actions)."
        n_available_action_per_agent = action_mask.sum(axis=-1)
        assert np.all(n_available_action_per_agent >= 1), "At least one action must be available for each agent."
        self.action_mask = action_mask

    def reset(self):
        obs, state = self.wrapped.reset()
        obs.available_actions = self.available_actions()
        return obs, state

    def step(self, actions):
        step = self.wrapped.step(actions)
        step.obs.available_actions = self.available_actions()
        return step

    def available_actions(self):
        return self.action_mask & self.wrapped.available_actions()
