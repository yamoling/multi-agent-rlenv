from dataclasses import dataclass
from typing_extensions import TypeVar
from typing import Sequence

import numpy as np
import numpy.typing as npt

from marlenv.models import State, ActionSpace, ContinuousActionSpace, DiscreteActionSpace

from .rlenv_wrapper import MARLEnv, RLEnvWrapper

AS = TypeVar("AS", bound=ActionSpace, default=ActionSpace)
DiscreteActionType = npt.NDArray[np.int64 | np.int32] | Sequence[int]
ContinuousActionType = npt.NDArray[np.float32] | Sequence[Sequence[float]]
A = TypeVar("A", bound=DiscreteActionType | ContinuousActionType)


@dataclass
class LastAction(RLEnvWrapper[A, AS]):
    """Env wrapper that adds the last action taken by the agents to the extra features."""

    def __init__(self, env: MARLEnv[A, AS]):
        assert len(env.extra_shape) == 1, "Adding last action is only possible with 1D extras"
        super().__init__(
            env,
            extra_shape=(env.extra_shape[0] + env.n_actions,),
            state_extra_shape=(env.state_extra_shape[0] + env.n_actions * env.n_agents,),
            extra_meanings=env.extras_meanings + ["Last action"] * env.n_actions,
        )
        self.state_extra_index = env.state_extra_shape[0]
        self.last_one_hot_actions = np.zeros((env.n_agents, env.n_actions), dtype=np.float32)

    def reset(self):
        obs, state = super().reset()
        self.last_one_hot_actions = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
        obs.add_extra(self.last_one_hot_actions)
        state.add_extra(self.last_one_hot_actions.flatten())
        return obs, state

    def step(self, actions: A):
        step = super().step(actions)
        match self.wrapped.action_space:
            case ContinuousActionSpace():
                self.last_actions = actions
            case DiscreteActionSpace():
                self.last_one_hot_actions = self.compute_one_hot_actions(actions)  # type: ignore
            case other:
                raise NotImplementedError(f"Action space {other} not supported")
        step.obs.add_extra(self.last_one_hot_actions)
        step.state.add_extra(self.last_one_hot_actions.flatten())
        return step

    def get_state(self):
        state = super().get_state()
        state.add_extra(self.last_one_hot_actions.flatten())
        return state

    def set_state(self, state: State):
        flattened_one_hots = state.extras[self.state_extra_index : self.state_extra_index + self.n_agents * self.n_actions]
        self.last_one_hot_actions = flattened_one_hots.reshape(self.n_agents, self.n_actions)
        return super().set_state(state)

    def compute_one_hot_actions(self, actions: DiscreteActionType) -> npt.NDArray:
        one_hot_actions = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
        index = np.arange(self.n_agents)
        one_hot_actions[index, actions] = 1.0
        return one_hot_actions
