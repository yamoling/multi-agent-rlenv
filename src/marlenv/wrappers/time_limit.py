from dataclasses import dataclass
from typing import Optional

import numpy as np

from marlenv.models import Observation, State

from .rlenv_wrapper import A, D, MARLEnv, RLEnvWrapper, S, R


@dataclass
class TimeLimit(RLEnvWrapper[A, D, S, R]):
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
        env: MARLEnv[A, D, S, R],
        step_limit: int,
        add_extra: bool = True,
        truncation_penalty: Optional[float] = None,
    ) -> None:
        assert len(env.extra_shape) == 1
        extras_shape = env.extra_shape
        if add_extra:
            dims = env.extra_shape[0]
            extras_shape = (dims + 1,)
        super().__init__(env, extra_shape=extras_shape)
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
        step.obs
        if self.add_extra:
            self.add_time_extra(step.obs, step.state)
        # If we reach the time limit
        if self._current_step >= self.step_limit:
            # And the episode is not done
            if not step.done:
                # then we set the truncation flag
                step.truncated = True
                step.reward -= self.truncation_penalty  # type: ignore
                if self.add_extra:
                    step.done = True
        return step

    def add_time_extra(self, obs: Observation[D], state: State[S]):
        counter = self._current_step / self.step_limit
        state.add_extra(counter)
        time_ratio = np.full(
            (self.n_agents, 1),
            counter,
            dtype=np.float32,
        )
        obs.add_extra(time_ratio)
