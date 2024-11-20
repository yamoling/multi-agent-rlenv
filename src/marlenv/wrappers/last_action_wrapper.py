import numpy as np
import numpy.typing as npt
from marlenv.models import ActionSpace, State
from .rlenv_wrapper import RLEnvWrapper, MARLEnv
from typing import TypeVar

A = TypeVar("A", bound=ActionSpace)
D = TypeVar("D")
S = TypeVar("S")
R = TypeVar("R", bound=float | np.ndarray)


class LastAction(RLEnvWrapper[A, D, S, R]):
    """Env wrapper that adds the last action taken by the agents to the extra features."""

    def __init__(self, env: MARLEnv[A, D, S, R]):
        assert len(env.extra_shape) == 1, "Adding last action is only possible with 1D extras"
        super().__init__(
            env,
            extra_shape=(env.extra_shape[0] + env.n_actions,),
            state_extra_shape=(env.state_extra_shape[0] + env.n_actions * env.n_agents,),
        )
        self.state_extra_index = env.state_extra_shape[0]
        self.last_one_hot_actions = np.zeros((env.n_agents, env.n_actions), dtype=np.float32)

    def reset(self):
        obs, state = super().reset()
        self.last_one_hot_actions = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
        obs.add_extra(self.last_one_hot_actions)
        state.add_extra(self.last_one_hot_actions.flatten())
        return obs, state

    def step(self, actions: npt.NDArray[np.int32]):
        step = super().step(actions)
        self.last_one_hot_actions = self.compute_one_hot_actions(actions)
        step.obs.add_extra(self.last_one_hot_actions)
        step.state.add_extra(self.last_one_hot_actions.flatten())
        return step

    def get_state(self):
        state = super().get_state()
        state.add_extra(self.last_one_hot_actions.flatten())
        return state

    def set_state(self, state: State[S]):
        flattened_one_hots = state.extras[self.state_extra_index : self.state_extra_index + self.n_agents * self.n_actions]
        self.last_one_hot_actions = flattened_one_hots.reshape(self.n_agents, self.n_actions)
        return super().set_state(state)

    def compute_one_hot_actions(self, actions: npt.NDArray[np.int32]):
        one_hot_actions = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
        index = np.arange(self.n_agents)
        one_hot_actions[index, actions] = 1.0
        return one_hot_actions
