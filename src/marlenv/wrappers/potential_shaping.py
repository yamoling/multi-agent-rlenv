from abc import abstractmethod, ABC
from .rlenv_wrapper import RLEnvWrapper
from marlenv import Space, MARLEnv, Observation
from typing import TypeVar, Optional

A = TypeVar("A", bound=Space)


class PotentialShaping(RLEnvWrapper[A], ABC):
    """
    Potential shaping for the Laser Learning Environment (LLE).

    https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
    """

    gamma: float

    def __init__(
        self,
        env: MARLEnv,
        gamma: float = 1.0,
        extra_shape: Optional[tuple[int]] = None,
    ):
        super().__init__(env, extra_shape=extra_shape)
        self.gamma = gamma
        self.current_potential = self.compute_potential()

    def add_extras(self, obs: Observation) -> Observation:
        """Add the extras related to potential shaping. Does nothing by default."""
        return obs

    def reset(self):
        obs, state = super().reset()
        self.current_potential = self.compute_potential()
        return self.add_extras(obs), state

    def step(self, actions):
        phi_t = self.current_potential
        step = super().step(actions)

        self.current_potential = self.compute_potential()
        shaped_reward = self.gamma * self.current_potential - phi_t
        step.obs = self.add_extras(step.obs)
        step.reward += shaped_reward
        return step

    @abstractmethod
    def compute_potential(self) -> float:
        """Compute the potential of the current state of the environment."""
