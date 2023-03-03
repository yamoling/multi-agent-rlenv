from .rlenv_wrapper import RLEnvWrapper, RLEnv

class TimeLimitWrapper(RLEnvWrapper):
    def __init__(self, env: RLEnv, step_limit: int) -> None:
        super().__init__(env)
        self._step_limit = step_limit
        self._current_step = 0


    def reset(self):
        self._current_step = 0
        return super().reset()

    def step(self, actions):
        self._current_step += 1
        obs_, reward, done, info =  super().step(actions)
        done = done or (self._current_step >= self._step_limit)
        return obs_, reward, done, info

    def summary(self) -> dict[str, str]:
        return {
            **super().summary(),
            "time_limit": self._step_limit
        }
            