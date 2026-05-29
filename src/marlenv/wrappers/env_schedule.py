from dataclasses import dataclass
from typing import TypeVar

from marlenv.wrappers.rlenv_wrapper import MARLEnv, RLEnvWrapper

A = TypeVar("A")


@dataclass
class EnvSchedule(RLEnvWrapper[A]):
    """
    Schedules the environments to change after a certain number of resets (i.e. episodes).
    """

    envs: list[tuple[int, MARLEnv[A]]]

    def __init__(self, envs: dict[int, MARLEnv[A]]):
        """
        Args:
            envs: A dictionary mapping the number of resets to the environment to use after that many resets. For instance, `{0: env1, 100: env2}` means that env1 will be used for the first 100 resets, and then env2 will be used for all subsequent resets.
        """
        assert len(envs) >= 1, "At least one environment must be provided"
        assert all(k >= 0 for k in envs.keys()), "Environment time steps must be non-negative"
        e = list(envs.values())[0]
        for env in envs.values():
            if not e.has_same_inouts(env):
                raise ValueError("All environments must have the same observation, extras and action spaces")
        sorted_envs = sorted(envs.items(), key=lambda x: x[0])
        assert sorted_envs[0][0] == 0, "The first environment must start at reset 0"
        self.envs = sorted_envs
        self.current_index = 0
        super().__init__(sorted_envs[0][1])

    @property
    def current_env_end(self):
        """The number of resets at which the current environment will end and the next one will start. The last environment lasts indefinitely."""
        if self.current_index + 1 < len(self.envs):
            return self.envs[self.current_index + 1][0]
        return float("inf")

    def reset(self, *, seed: int | None = None):
        self.n_resets += 1
        if self.n_resets >= self.current_env_end:
            self.current_index += 1
            self.wrapped = self.envs[self.current_index][1]
        return super().reset(seed=seed)
