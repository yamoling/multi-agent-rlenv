from .rlenv_wrapper import RLEnvWrapper, RLEnv


class TimePenaltyWrapper(RLEnvWrapper):
    def __init__(self, env: RLEnv, penalty: float) -> None:
        super().__init__(env)
        self.penalty = penalty

    def step(self, action: int) -> tuple:
        obs, reward, *rest = self.wrapped.step(action)
        reward -= self.penalty
        return obs, reward, *rest
