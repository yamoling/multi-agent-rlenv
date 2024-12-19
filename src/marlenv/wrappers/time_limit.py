from dataclasses import dataclass
from typing import Optional

import numpy as np

from marlenv.models import Observation, State

from .rlenv_wrapper import MARLEnv, RLEnvWrapper, A, AS


@dataclass
class TimeLimit(RLEnvWrapper[A, AS]):
    """
    Limits the number of time steps for an episode. When the number of steps is reached, then the episode is truncated.

    - If the `add_extra` flag is set to True, then an extra signal is added to the observation, which is the ratio of the
    current step over the maximum number of steps. In this case, the done flag is also set to True when the maximum
    number of steps is reached.
    - The `truncated` flag is only set to `True` when the maximum number of steps is reached and the episode is not done.
    - The `truncation_penalty` is subtracted from the reward when the episode is truncated. This is only possible when
    the `add_extra` flag is set to True, otherwise the agent is not able to anticipate this penalty.
    """

    step_limit: int
    add_extra: bool
    truncation_penalty: float

    def __init__(
        self,
        env: MARLEnv[A, AS],
        step_limit: int,
        add_extra: bool = True,
        truncation_penalty: Optional[float] = None,
    ) -> None:
        assert len(env.extra_shape) == 1
        assert len(env.state_extra_shape) == 1
        extras_shape = env.extra_shape
        state_extras_shape = env.state_extra_shape
        self.extra_index = 0
        if add_extra:
            dims = env.extra_shape[0]
            self.extra_index = dims
            extras_shape = (dims + 1,)
            state_extras_shape = (env.state_extra_shape[0] + 1,)
        super().__init__(env, extra_shape=extras_shape, state_extra_shape=state_extras_shape)
        self.step_limit = step_limit
        self._current_step = 0
        self.add_extra = add_extra
        if truncation_penalty is None:
            truncation_penalty = 0.0
        else:
            assert add_extra, "The truncation penalty can only be set when the add_extra flag is set to True, otherwise agents are not able to anticipate this punishment."
        assert truncation_penalty >= 0, "The truncation penalty must be a positive value."
        self.truncation_penalty = truncation_penalty

    def reset(self):
        self._current_step = 0
        obs, state = super().reset()
        if self.add_extra:
            self.add_time_extra(obs, state)
        return obs, state

    def step(self, actions):
        self._current_step += 1
        step = super().step(actions)
        if self.add_extra:
            self.add_time_extra(step.obs, step.state)
        # If we reach the time limit
        if self._current_step >= self.step_limit:
            # And the episode is not done
            if not step.done:
                step.truncated = True
                step.reward = step.reward - self.truncation_penalty  # type: ignore
                step.done = self.add_extra
        return step

    def add_time_extra(self, obs: Observation, state: State):
        counter = self._current_step / self.step_limit
        time_ratio = np.full(
            (self.n_agents, 1),
            counter,
            dtype=np.float32,
        )
        obs.add_extra(time_ratio)
        state.add_extra(counter)

    def get_state(self):
        state = super().get_state()
        if self.add_extra:
            state.add_extra(self._current_step / self.step_limit)
        return state

    def set_state(self, state: State):
        if self.add_extra:
            self._current_step = int(state.extras[self.extra_index] * self.step_limit)
        super().set_state(state)
