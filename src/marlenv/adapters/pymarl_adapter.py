from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from marlenv.models import DiscreteActionSpace, MARLEnv
from marlenv.wrappers import TimeLimit


@dataclass
class PymarlAdapter:
    """
    There is no official interface for PyMARL but aims at complying
    with the pymarl-qplex code base.
    """

    def __init__(self, env: MARLEnv[Sequence | npt.NDArray, DiscreteActionSpace], episode_limit: int):
        assert env.reward_space.size == 1, "Only single objective environments are supported."
        self.env = TimeLimit(env, episode_limit, add_extra=False)
        # Required by PyMarl
        self.episode_limit = episode_limit
        self.current_observation = None
        self.current_state = None
        self.n_agents = self.env.n_agents
        self.n_actions = self.env.n_actions

    def step(self, actions) -> tuple[float, bool, dict[str, Any]]:
        """Returns reward, terminated, info"""
        step = self.env.step(actions)
        self.current_observation = step.obs
        return float(step.reward[0]), step.is_terminal, step.info

    def get_obs(self):
        """Returns all agent observations in a list"""
        if self.current_observation is None:
            raise ValueError("No observation available. Call reset() first.")
        # If there are no extras, return the data
        if self.current_observation.extras.size == 0:
            return self.current_observation.data
        return np.concatenate([self.current_observation.data, self.current_observation.extras], axis=-1)

    def get_obs_agent(self, agent_id: int):
        """Returns observation for agent_id"""
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        """Returns the shape of the observation"""
        if len(self.env.observation_shape) == 1:
            return self.env.observation_shape[0]
        return self.env.observation_shape

    def get_state(self):
        return self.env.get_state().data

    def get_state_size(self):
        """Returns the shape of the state"""
        if len(self.env.state_shape) == 1:
            return self.env.state_shape[0]
        return self.env.state_shape

    def get_avail_actions(self):
        return self.env.available_actions()

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return self.env.available_actions()

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        # Note: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.env.n_actions

    def reset(self):
        self.current_observation, self.current_state = self.env.reset()

    def render(self):
        return self.env.render()

    def close(self):
        return

    def seed(self):
        self.env.seed(0)

    def save_replay(self):
        return

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.env.n_agents,
            "episode_limit": self.env.step_limit,
        }
        try:
            env_info["unit_dim"] = self.env.agent_state_size
        except NotImplementedError:
            env_info["unit_dim"] = 0
        return env_info
