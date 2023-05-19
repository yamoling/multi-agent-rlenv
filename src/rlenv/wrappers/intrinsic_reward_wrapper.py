import math
from abc import abstractmethod
from rlenv.models import Observation
from .rlenv_wrapper import RLEnvWrapper, RLEnv


class IntrinsicReward(RLEnvWrapper):
    @abstractmethod
    def _compute_extrinsic_reward(self, obs: Observation) -> float:
        """Extrinsic reward computation"""

    def step(self, actions):
        obs, reward, *rest = super().step(actions)
        reward += self._compute_extrinsic_reward(obs)
        return obs, reward, *rest


class LinearStateCount(IntrinsicReward):
    def __init__(self, env: RLEnv, additional_reward: float, anneal_on: int) -> None:
        super().__init__(env)
        self.additional_reward = additional_reward
        self.anneal_on = anneal_on
        self.visit_count = {}

    def _compute_extrinsic_reward(self, obs: Observation) -> float:
        h = hash(obs.data.tobytes())
        visit_count = self.visit_count.get(h, 0)
        if visit_count > self.anneal_on:
            return 0.0
        self.visit_count[h] = visit_count + 1
        return self.additional_reward * (1 - ((self.anneal_on - visit_count) / self.anneal_on))

    def kwargs(self) -> dict[str,]:
        return {"additional_reward": self.additional_reward, "anneal_on": self.anneal_on}


class DecreasingExpStateCount(IntrinsicReward):
    """
    Decreasing exponential extrinsic reward based on the state visit count.
    extrinsic_reward(visit_count) = scale_by * e^(visit_count/anneal)
    """

    def __init__(self, env: RLEnv, scale_by: float, anneal: float = 1.0) -> None:
        super().__init__(env)
        self.scale = scale_by
        self.anneal = anneal
        # After visiting the state 3*anneal, the value is almost zero for any anneal value
        self.max_visit_count = int(3 * anneal)
        self.visit_count: dict[int, int] = {}

    def _compute_extrinsic_reward(self, obs: Observation) -> float:
        h = hash(obs.data.tobytes())
        visit_count = self.visit_count.get(h, 0)
        if visit_count > self.max_visit_count:
            return 0
        self.visit_count[h] = visit_count + 1
        return self.scale * math.exp(-visit_count / self.anneal)

    def kwargs(self) -> dict[str,]:
        return {"scale_by": self.scale, "anneal": self.anneal}
