from dataclasses import dataclass
from typing import Sequence, overload

import numpy as np
import numpy.typing as npt
from smac.env import StarCraft2Env

from marlenv.models import DiscreteActionSpace, MARLEnv, Observation, State, Step


@dataclass
class SMAC(MARLEnv[Sequence[int] | npt.NDArray, DiscreteActionSpace]):
    """Wrapper for the SMAC environment to work with this framework"""

    @overload
    def __init__(
        self,
        map_name="8m",
        step_mul=8,
        move_amount=2,
        difficulty="7",
        game_version=None,
        seed=None,
        continuing_episode=False,
        obs_all_health=True,
        obs_own_health=True,
        obs_last_action=False,
        obs_pathing_grid=False,
        obs_terrain_height=False,
        obs_instead_of_state=False,
        obs_timestep_number=False,
        state_last_action=True,
        state_timestep_number=False,
        reward_sparse=False,
        reward_only_positive=True,
        reward_death_value=10,
        reward_win=200,
        reward_defeat=0,
        reward_negative_scale=0.5,
        reward_scale=True,
        reward_scale_rate=20,
        replay_dir="",
        replay_prefix="",
        window_size_x=1920,
        window_size_y=1200,
        heuristic_ai=False,
        heuristic_rest=False,
        debug=False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        map_name : str, optional
            The name of the SC2 map to play (default is "8m"). The full list
            can be found by running bin/map_list.
        step_mul : int, optional
            How many game steps per agent step (default is 8). None
            indicates to use the default map step_mul.
        move_amount : float, optional
            How far away units are ordered to move per step (default is 2).
        difficulty : str, optional
            The difficulty of built-in computer AI bot (default is "7").
        game_version : str, optional
            StarCraft II game version (default is None). None indicates the
            latest version.
        seed : int, optional
            Random seed used during game initialisation. This allows to
        continuing_episode : bool, optional
            Whether to consider episodes continuing or finished after time
            limit is reached (default is False).
        obs_all_health : bool, optional
            Agents receive the health of all units (in the sight range) as part
            of observations (default is True).
        obs_own_health : bool, optional
            Agents receive their own health as a part of observations (default
            is False). This flag is ignored when obs_all_health == True.
        obs_last_action : bool, optional
            Agents receive the last actions of all units (in the sight range)
            as part of observations (default is False).
        obs_pathing_grid : bool, optional
            Whether observations include pathing values surrounding the agent
            (default is False).
        obs_terrain_height : bool, optional
            Whether observations include terrain height values surrounding the
            agent (default is False).
        obs_instead_of_state : bool, optional
            Use combination of all agents' observations as the global state
            (default is False).
        obs_timestep_number : bool, optional
            Whether observations include the current timestep of the episode
            (default is False).
        state_last_action : bool, optional
            Include the last actions of all agents as part of the global state
            (default is True).
        state_timestep_number : bool, optional
            Whether the state include the current timestep of the episode
            (default is False).
        reward_sparse : bool, optional
            Receive 1/-1 reward for winning/loosing an episode (default is
            False). Whe rest of reward parameters are ignored if True.
        reward_only_positive : bool, optional
            Reward is always positive (default is True).
        reward_death_value : float, optional
            The amount of reward received for killing an enemy unit (default
            is 10). This is also the negative penalty for having an allied unit
            killed if reward_only_positive == False.
        reward_win : float, optional
            The reward for winning in an episode (default is 200).
        reward_defeat : float, optional
            The reward for loosing in an episode (default is 0). This value
            should be nonpositive.
        reward_negative_scale : float, optional
            Scaling factor for negative rewards (default is 0.5). This
            parameter is ignored when reward_only_positive == True.
        reward_scale : bool, optional
            Whether or not to scale the reward (default is True).
        reward_scale_rate : float, optional
            Reward scale rate (default is 20). When reward_scale == True, the
            reward received by the agents is divided by (max_reward /
            reward_scale_rate), where max_reward is the maximum possible
            reward per episode without considering the shield regeneration
            of Protoss units.
        replay_dir : str, optional
            The directory to save replays (default is None). If None, the
            replay will be saved in Replays directory where StarCraft II is
            installed.
        replay_prefix : str, optional
            The prefix of the replay to be saved (default is None). If None,
            the name of the map will be used.
        window_size_x : int, optional
            The length of StarCraft II window size (default is 1920).
        window_size_y: int, optional
            The height of StarCraft II window size (default is 1200).
        heuristic_ai: bool, optional
            Whether or not to use a non-learning heuristic AI (default False).
        heuristic_rest: bool, optional
            At any moment, restrict the actions of the heuristic AI to be
            chosen from actions available to RL agents (default is False).
            Ignored if heuristic_ai == False.
        debug: bool, optional
            Log messages about observations, state, actions and rewards for
            debugging purposes (default is False).
        """

    @overload
    def __init__(self, env: StarCraft2Env): ...

    def __init__(self, env_or_map_name, **kwargs):  # type: ignore
        match env_or_map_name:
            case StarCraft2Env():
                self._env = env_or_map_name
                map_name = env_or_map_name.map_name
            case str():
                map_name = env_or_map_name
                self._env = StarCraft2Env(map_name=map_name, **kwargs)
            case other:
                raise ValueError(f"Invalid argument type: {type(other)}")
        self._env = StarCraft2Env(map_name=map_name)
        action_space = DiscreteActionSpace(self._env.n_agents, self._env.n_actions)
        self._env_info = self._env.get_env_info()
        super().__init__(
            action_space=action_space,
            observation_shape=(self._env_info["obs_shape"],),
            state_shape=(self._env_info["state_shape"],),
        )
        self._seed = self._env.seed()
        self.name = f"smac-{self._env.map_name}"

    def reset(self):
        obs, state = self._env.reset()
        obs = Observation(np.array(obs), self.available_actions(), state)
        return obs

    def get_observation(self):
        return self._env.get_obs()

    def get_state(self):
        return State(self._env.get_state())

    def step(self, actions):
        reward, done, info = self._env.step(actions)
        obs = Observation(
            self._env.get_obs(),  # type: ignore
            self.available_actions(),
        )
        state = self.get_state()
        step = Step(
            obs,
            state,
            reward,
            done,
            False,
            info,
        )
        return step

    def available_actions(self) -> npt.NDArray[np.bool_]:
        return np.array(self._env.get_avail_actions()) == 1

    def get_image(self):
        return self._env.render(mode="rgb_array")

    def seed(self, seed_value: int):
        self._env = StarCraft2Env(map_name=self._env.map_name, seed=seed_value)
