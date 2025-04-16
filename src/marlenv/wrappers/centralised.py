from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np
import numpy.typing as npt

from marlenv.models import DiscreteSpace, MARLEnv, MultiDiscreteSpace, Observation

from .rlenv_wrapper import RLEnvWrapper


@dataclass
class Centralized(RLEnvWrapper[MultiDiscreteSpace]):
    joint_action_space: DiscreteSpace

    def __init__(self, env: MARLEnv[MultiDiscreteSpace]):
        if not isinstance(env.action_space, MultiDiscreteSpace):
            raise NotImplementedError(f"Action space {env.action_space} not supported")
        joint_observation_shape = (env.observation_shape[0] * env.n_agents, *env.observation_shape[1:])
        super().__init__(
            env,
            n_agents=1,
            observation_shape=joint_observation_shape,
            state_shape=env.state_shape,
            state_extra_shape=env.extras_shape,
            action_space=self._make_joint_action_space(env),
        )

    def reset(self):
        obs, state = super().reset()
        return self._joint_observation(obs), state

    def get_observation(self):
        obs = super().get_observation()
        return self._joint_observation(obs)

    def _make_joint_action_space(self, env: MARLEnv[MultiDiscreteSpace]):
        agent_actions = list[list[str]]()
        for agent in range(env.n_agents):
            agent_actions.append([f"{agent}-{action}" for action in env.action_space.labels])
        action_names = [str(a) for a in product(*agent_actions)]
        return DiscreteSpace(env.n_actions**env.n_agents, action_names).repeat(1)

    def step(self, actions: npt.NDArray | Sequence):
        action = actions[0]
        individual_actions = self._individual_actions(action)
        individual_actions = np.array(individual_actions)
        step = self.wrapped.step(individual_actions)  # type: ignore
        step.obs = self._joint_observation(step.obs)
        return step

    def _individual_actions(self, joint_action: int):
        individual_actions = list[int]()
        for _ in range(self.wrapped.n_agents):
            action = joint_action % self.wrapped.n_actions
            joint_action = joint_action // self.wrapped.n_actions
            individual_actions.append(action)
        individual_actions.reverse()
        return individual_actions

    def available_actions(self):
        individual_available = self.wrapped.available_actions()
        joint_available = list[float]()
        for actions in product(*individual_available):
            joint_available.append(True if all(actions) else False)
        available_actions = np.array(joint_available, dtype=bool)
        return available_actions.reshape((self.n_agents, self.n_actions))

    def _joint_observation(self, obs: Observation):
        obs.data = np.concatenate(obs.data, axis=0)
        obs.extras = np.concatenate(obs.extras, axis=0)
        # Unsqueze the first dimension since there is one agent
        obs.data = np.expand_dims(obs.data, axis=0)
        obs.extras = np.expand_dims(obs.extras, axis=0)
        obs.available_actions = self.available_actions()
        return obs
