import numpy as np
import pygame
import cv2
import sys
from marlenv.models import MARLEnv, State, Observation, Step, DiscreteActionSpace
from typing import Literal, Sequence
import numpy.typing as npt
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from dataclasses import dataclass


@dataclass
class Overcooked(MARLEnv[Sequence[int] | npt.NDArray, DiscreteActionSpace]):
    def __init__(self, oenv: OvercookedEnv):
        self._oenv = oenv
        assert isinstance(oenv.mdp, OvercookedGridworld)
        self._mdp = oenv.mdp
        self.visualizer = StateVisualizer()
        super().__init__(
            action_space=DiscreteActionSpace(n_agents=self._mdp.num_players, n_actions=Action.NUM_ACTIONS),
            observation_shape=(1,),
            state_shape=(1,),
        )

    def _state_data(self):
        state = self._oenv.state
        state = np.array(self._mdp.lossless_state_encoding(state))
        # Use axes (agents, channels, height, width) instead of (agents, height, width, channels)
        state = np.transpose(state, (0, 3, 1, 2))
        return state

    def get_state(self):
        return State(self._state_data())

    def get_observation(self) -> Observation:
        return Observation(self._state_data(), self.available_actions())

    def available_actions(self):
        available_actions = np.full((self.n_agents, self.n_actions), False)
        actions = self._mdp.get_actions(self._oenv.state)
        for agent_num, agent_actions in enumerate(actions):
            for action in agent_actions:
                available_actions[agent_num, Action.ACTION_TO_INDEX[action]] = True
        return np.array(available_actions)

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

    def get_image(self):
        rewards_dict = {}  # dictionary of details you want rendered in the UI
        for key, value in self._oenv.game_stats.items():
            if key in [
                "cumulative_shaped_rewards_by_agent",
                "cumulative_sparse_rewards_by_agent",
            ]:
                rewards_dict[key] = value

        image = self.visualizer.render_state(
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
