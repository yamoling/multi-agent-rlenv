from itertools import product
import numpy as np
from rlenv.models import RLEnv, DiscreteActionSpace, Observation
from .rlenv_wrapper import RLEnvWrapper, A


class Centralised(RLEnvWrapper[A]):
    def __init__(self, env: RLEnv[A]):
        if not isinstance(env.action_space, DiscreteActionSpace):
            raise NotImplementedError(f"Action space {env.action_space} not supported")
        joint_observation_shape = (env.observation_shape[0] * env.n_agents, *env.observation_shape[1:])
        super().__init__(
            env,
            joint_observation_shape,
            env.state_shape,
            env.extra_feature_shape,
            action_space=self._make_joint_action_space(env),  # type: ignore
        )

    def reset(self):
        obs = super().reset()
        return self._joint_observation(obs)

    def _make_joint_action_space(self, env: RLEnv[DiscreteActionSpace]):
        agent_actions = list[list[str]]()
        for agent in range(env.n_agents):
            agent_actions.append([f"{agent}-{action}" for action in env.action_space.action_names])
        action_names = [str(a) for a in product(*agent_actions)]
        return DiscreteActionSpace(1, env.n_actions**env.n_agents, action_names)

    def step(self, actions):
        individual_actions = self._individual_actions(actions[0])
        obs, *rest = self.wrapped.step(individual_actions)
        return self._joint_observation(obs), *rest

    def _individual_actions(self, joint_action: int):
        individual_actions = []
        for _ in range(self.wrapped.n_agents):
            action = joint_action % self.wrapped.n_actions
            joint_action = joint_action // self.wrapped.n_actions
            individual_actions.append(action)
        individual_actions.reverse()
        return np.array(individual_actions)

    def _joint_observation(self, obs: Observation):
        obs.data = np.concatenate(obs.data, axis=0)
        return obs
