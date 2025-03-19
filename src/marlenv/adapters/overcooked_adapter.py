import sys
from dataclasses import dataclass
from typing import Literal, Sequence
from copy import deepcopy

import cv2
import numpy as np
import numpy.typing as npt
import pygame
from marlenv.models import ContinuousSpace, DiscreteActionSpace, MARLEnv, Observation, State, Step

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import Action, OvercookedGridworld, OvercookedState
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer


@dataclass
class Overcooked(MARLEnv[Sequence[int] | npt.NDArray, DiscreteActionSpace]):
    horizon: int

    def __init__(self, oenv: OvercookedEnv):
        self._oenv = oenv
        assert isinstance(oenv.mdp, OvercookedGridworld)
        self._mdp = oenv.mdp
        self._visualizer = StateVisualizer()
        width, height, layers = tuple(self._mdp.lossless_state_encoding_shape)
        # -1 because we extract the "urgent" layer to the extras
        shape = (int(layers - 1), int(width), int(height))
        super().__init__(
            action_space=DiscreteActionSpace(
                n_agents=self._mdp.num_players,
                n_actions=Action.NUM_ACTIONS,
                action_names=[Action.ACTION_TO_CHAR[a] for a in Action.ALL_ACTIONS],
            ),
            observation_shape=shape,
            extras_shape=(2,),
            extras_meanings=["timestep", "urgent"],
            state_shape=shape,
            state_extra_shape=(2,),
            reward_space=ContinuousSpace.from_shape(1),
        )
        self.horizon = int(self._oenv.horizon)

    @property
    def state(self) -> OvercookedState:
        """Current state of the environment"""
        return self._oenv.state

    def set_state(self, state: State):
        raise NotImplementedError("Not yet implemented")

    @property
    def time_step(self):
        return self.state.timestep

    def _state_data(self):
        players_layers = self._mdp.lossless_state_encoding(self.state)
        state = np.array(players_layers, dtype=np.float32)
        # Use axes (agents, channels, height, width) instead of (agents, height, width, channels)
        state = np.transpose(state, (0, 3, 1, 2))
        # The last last layer is for "urgency", put it in the extras
        urgency = float(np.all(state[:, -1]))
        state = state[:, :-1]
        return state, urgency

    def get_state(self):
        data, is_urgent = self._state_data()
        return State(data[0], np.array([self.time_step / self.horizon, is_urgent], dtype=np.float32))

    def get_observation(self) -> Observation:
        data, is_urgent = self._state_data()
        return Observation(
            data=data,
            available_actions=self.available_actions(),
            extras=np.array([[self.time_step / self.horizon, is_urgent]] * self.n_agents, dtype=np.float32),
        )

    def available_actions(self):
        available_actions = np.full((self.n_agents, self.n_actions), False)
        actions = self._mdp.get_actions(self._oenv.state)
        for agent_num, agent_actions in enumerate(actions):
            for action in agent_actions:
                available_actions[agent_num, Action.ACTION_TO_INDEX[action]] = True
        return np.array(available_actions, dtype=np.bool)

    def step(self, actions: Sequence[int] | npt.NDArray[np.int32 | np.int64]) -> Step:
        actions = [Action.ALL_ACTIONS[a] for a in actions]
        _, reward, done, info = self._oenv.step(actions, display_phi=True)
        return Step(
            obs=self.get_observation(),
            state=self.get_state(),
            reward=np.array([reward]),
            done=done,
            truncated=False,
            info=info,
        )

    def reset(self):
        self._oenv.reset()
        return self.get_observation(), self.get_state()

    def __deepcopy__(self, memo: dict):
        mdp = deepcopy(self._mdp)
        return Overcooked(OvercookedEnv.from_mdp(mdp, horizon=self.horizon))

    def __getstate__(self):
        return {"horizon": self.horizon, "mdp": self._mdp}

    def __setstate__(self, state: dict):
        from overcooked_ai_py.mdp.overcooked_mdp import Recipe

        mdp = state["mdp"]
        Recipe.configure(mdp.recipe_config)
        self.__init__(OvercookedEnv.from_mdp(state["mdp"], horizon=state["horizon"]))

    def get_image(self):
        rewards_dict = {}  # dictionary of details you want rendered in the UI
        for key, value in self._oenv.game_stats.items():
            if key in [
                "cumulative_shaped_rewards_by_agent",
                "cumulative_sparse_rewards_by_agent",
            ]:
                rewards_dict[key] = value

        image = self._visualizer.render_state(
            state=self._oenv.state,
            grid=self._mdp.terrain_mtx,
            hud_data=StateVisualizer.default_hud_data(self._oenv.state, **rewards_dict),
        )

        image = pygame.surfarray.array3d(image)
        image = np.flip(np.rot90(image, 3), 1)
        # Depending on the platform, the image may need to be converted to RGB
        if sys.platform in ("linux", "linux2"):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    @staticmethod
    def from_layout(
        layout: Literal[
            "asymmetric_advantages",
            "asymmetric_advantages_tomato",
            "bonus_order_test",
            "bottleneck",
            "centre_objects",
            "centre_pots",
            "coordination_ring",
            "corridor",
            "counter_circuit",
            "counter_circuit_o_1order",
            "cramped_corridor",
            "cramped_room",
            "cramped_room_o_3orders",
            "cramped_room_single",
            "cramped_room_tomato",
            "five_by_five",
            "forced_coordination",
            "forced_coordination_tomato",
            "inverse_marshmallow_experiment",
            "large_room",
            "long_cook_time",
            "marshmallow_experiment_coordination",
            "marshmallow_experiment",
            "mdp_test",
            "m_shaped_s",
            "multiplayer_schelling",
            "pipeline",
            "scenario1_s",
            "scenario2",
            "scenario2_s",
            "scenario3",
            "scenario4",
            "schelling",
            "schelling_s",
            "simple_o",
            "simple_o_t",
            "simple_tomato",
            "small_corridor",
            "soup_coordination",
            "tutorial_0",
            "tutorial_1",
            "tutorial_2",
            "tutorial_3",
            "unident",
            "you_shall_not_pass",
        ],
        horizon: int = 400,
    ):
        mdp = OvercookedGridworld.from_layout_name(layout)
        return Overcooked(OvercookedEnv.from_mdp(mdp, horizon=horizon))
