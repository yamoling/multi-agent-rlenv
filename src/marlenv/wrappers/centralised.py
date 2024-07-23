from itertools import product
import numpy as np
from marlenv.models import RLEnv, DiscreteSpace, Observation, ActionSpace
from .rlenv_wrapper import RLEnvWrapper, A


class Centralised(RLEnvWrapper[A]):
    def __init__(self, env: RLEnv[A]):
        if not isinstance(env.action_space.individual_action_space, DiscreteSpace):
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

    def _make_joint_action_space(self, env: RLEnv[A]):
        agent_actions = list[list[str]]()
        for agent in range(env.n_agents):
            agent_actions.append([f"{agent}-{action}" for action in env.action_space.action_names])
        action_names = [str(a) for a in product(*agent_actions)]
        return ActionSpace(1, DiscreteSpace(env.n_actions**env.n_agents, action_names))

    def step(self, actions):
        action = list(actions)[0]
        individual_actions = self._individual_actions(action)
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
