import numpy as np
import pytest

import marlenv
from marlenv import ContinuousSpace, DiscreteMockEnv, MARLEnv, Observation, State, MultiDiscreteSpace
from marlenv.adapters import PymarlAdapter

skip_gym = not marlenv.adapters.HAS_GYM
skip_pettingzoo = not marlenv.adapters.HAS_PETTINGZOO
skip_smac = not marlenv.adapters.HAS_SMAC
# Check for "overcooked_ai_py.mdp" specifically because after uninstalling, the package
# can still be found because of some remaining .pkl file.
skip_overcooked = not marlenv.adapters.HAS_OVERCOOKED


@pytest.mark.skipif(skip_gym, reason="Gymnasium is not installed")
def test_gym_adapter_discrete():
    # Discrete action space
    env = marlenv.make("CartPole-v1")
    assert isinstance(env.action_space, MultiDiscreteSpace)
    obs, state = env.reset()
    assert isinstance(obs, Observation)
    assert isinstance(state, State)
    assert isinstance(env, MARLEnv)
    assert env.n_actions == 2
    assert env.n_agents == 1

    step = env.step(env.sample_action())
    assert isinstance(step.obs, Observation)
    assert isinstance(step.reward, np.ndarray)
    assert step.reward.shape == (1,)
    assert isinstance(step.done, bool)
    assert isinstance(step.truncated, bool)
    assert isinstance(step.info, dict)


@pytest.mark.skipif(skip_gym, reason="Gymnasium is not installed")
def test_gym_adapter_continuous():
    env = marlenv.make("Pendulum-v1")
    assert isinstance(env.action_space, ContinuousSpace)
    obs, state = env.reset()
    assert isinstance(obs, Observation)
    assert isinstance(state, State)
    assert isinstance(env, MARLEnv)
    assert env.n_actions == 1
    assert env.n_agents == 1

    step = env.step(env.sample_action())
    assert isinstance(step.obs, Observation)
    assert isinstance(step.reward, np.ndarray)
    assert step.reward.shape == (1,)
    assert isinstance(step.done, bool)
    assert isinstance(step.truncated, bool)
    assert isinstance(step.info, dict)

    # Continuous action space
    env = marlenv.make("Pendulum-v1")
    env.reset()


@pytest.mark.skipif(skip_pettingzoo, reason="PettingZoo is not installed")
def test_pettingzoo_adapter_discrete_action():
    # https://pettingzoo.farama.org/environments/sisl/pursuit/#pursuit
    from pettingzoo.sisl import pursuit_v4

    env = marlenv.adapters.PettingZoo(pursuit_v4.parallel_env())
    env.reset()
    action = env.action_space.sample()
    step = env.step(action)
    assert isinstance(step.obs, Observation)
    assert isinstance(step.reward, np.ndarray)
    assert step.reward.shape == (1,)
    assert isinstance(step.done, bool)
    assert isinstance(step.truncated, bool)
    assert isinstance(step.info, dict)
    assert env.n_agents == 8
    assert env.n_actions == 5
    assert isinstance(env.action_space, MultiDiscreteSpace)


@pytest.mark.skipif(skip_pettingzoo, reason="PettingZoo is not installed")
def test_pettingzoo_adapter_continuous_action():
    from pettingzoo.sisl import waterworld_v4

    # https://pettingzoo.farama.org/environments/sisl/waterworld/
    env = marlenv.adapters.PettingZoo(waterworld_v4.parallel_env())
    env.reset()
    action = env.action_space.sample()
    step = env.step(action)
    assert isinstance(step.obs, Observation)
    assert isinstance(step.reward, np.ndarray)
    assert step.reward.shape == (1,)
    assert isinstance(step.done, bool)
    assert isinstance(step.truncated, bool)
    assert isinstance(step.info, dict)
    assert env.n_actions == 2
    assert env.n_agents == 2
    assert isinstance(env.action_space, ContinuousSpace)


def _check_env_3m(env):
    from marlenv.adapters import SMAC

    assert isinstance(env, SMAC)
    obs = env.reset()
    assert isinstance(obs, Observation)
    assert env.n_agents == 3
    assert isinstance(env.action_space, MultiDiscreteSpace)

    step = env.random_step()
    assert isinstance(step.obs, Observation)
    assert isinstance(step.reward, np.ndarray)
    assert step.reward.shape == (1,)
    assert isinstance(step.done, bool)
    assert isinstance(step.truncated, bool)
    assert isinstance(step.info, dict)


@pytest.mark.skipif(skip_smac, reason="SMAC is not installed")
def test_smac_from_class():
    from smac.env import StarCraft2Env

    from marlenv.adapters import SMAC

    env = SMAC(StarCraft2Env("3m"))
    _check_env_3m(env)


@pytest.mark.skipif(skip_smac, reason="SMAC is not installed")
def test_smac_render():
    from marlenv.adapters import SMAC

    env = SMAC("3m")
    env.reset()
    env.render()


@pytest.mark.skipif(skip_overcooked, reason="Overcooked is not installed")
def test_overcooked_attributes():
    from overcooked_ai_py.mdp.overcooked_mdp import Action

    from marlenv.adapters import Overcooked

    env = Overcooked.from_layout("simple_o")
    height, width = env._mdp.shape
    assert env.n_agents == 2
    assert env.n_actions == Action.NUM_ACTIONS
    assert env.observation_shape == (25, height, width)
    assert env.reward_space.shape == (1,)
    assert env.extras_shape == (2,)
    assert not env.is_multi_objective


@pytest.mark.skipif(skip_overcooked, reason="Overcooked is not installed")
def test_overcooked_obs_state():
    from marlenv.adapters import Overcooked

    HORIZON = 100
    env = Overcooked.from_layout("coordination_ring", horizon=HORIZON)
    height, width = env._mdp.shape
    obs, state = env.reset()
    for i in range(HORIZON):
        assert obs.data.dtype == np.float32
        assert state.data.dtype == np.float32
        assert obs.extras.dtype == np.float32
        assert state.extras.dtype == np.float32
        assert obs.shape == (25, height, width)
        assert obs.extras_shape == (2,)
        assert state.shape == (25, height, width)
        assert state.extras_shape == (2,)

        assert np.all(obs.extras[:, 0] == i / HORIZON)
        assert np.all(state.extras[0] == i / HORIZON)

        step = env.random_step()
        obs = step.obs
        state = step.state
        if i < HORIZON - 1:
            assert not step.done
        else:
            assert step.done


@pytest.mark.skipif(skip_overcooked, reason="Overcooked is not installed")
def test_overcooked_shaping():
    from marlenv.adapters import Overcooked

    UP = 0
    RIGHT = 2
    STAY = 4
    INTERACT = 5
    grid = [
        ["X", "X", "X", "D", "X"],
        ["X", "O", "S", "2", "X"],
        ["X", "1", "P", " ", "X"],
        ["X", "T", "S", " ", "X"],
        ["X", "X", "X", "X", "X"],
    ]

    env = Overcooked.from_grid(grid)
    env.reset()
    actions_rewards = [
        ([UP, STAY], False),
        ([INTERACT, STAY], False),
        ([RIGHT, STAY], False),
        ([INTERACT, STAY], True),
    ]

    for action, expected_reward in actions_rewards:
        step = env.step(action)
        if expected_reward:
            assert step.reward.item() > 0


@pytest.mark.skipif(skip_overcooked, reason="Overcooked is not installed")
def test_overcooked_name():
    from marlenv.adapters import Overcooked

    grid = [
        ["X", "X", "X", "D", "X"],
        ["X", "O", "S", "2", "X"],
        ["X", "1", "P", " ", "X"],
        ["X", "T", "S", " ", "X"],
        ["X", "X", "X", "X", "X"],
    ]

    env = Overcooked.from_grid(grid)
    assert env.name == "Overcooked-custom-layout"

    env = Overcooked.from_grid(grid, layout_name="my incredible grid")
    assert env.name == "Overcooked-my incredible grid"

    env = Overcooked.from_layout("asymmetric_advantages")
    assert env.name == "Overcooked-asymmetric_advantages"


def test_pymarl():
    LIMIT = 20
    N_AGENTS = 2
    N_ACTIONS = 5
    UNIT_STATE_SIZE = 1
    OBS_SIZE = 42
    REWARD_STEP = 1
    env = PymarlAdapter(
        DiscreteMockEnv(
            N_AGENTS,
            n_actions=N_ACTIONS,
            agent_state_size=UNIT_STATE_SIZE,
            obs_size=OBS_SIZE,
            reward_step=REWARD_STEP,
        ),
        LIMIT,
    )

    info = env.get_env_info()
    assert info["n_agents"] == N_AGENTS
    assert info["n_actions"] == N_ACTIONS
    assert env.get_total_actions() == N_ACTIONS
    assert info["state_shape"] == UNIT_STATE_SIZE * N_AGENTS
    assert env.get_state_size() == UNIT_STATE_SIZE * N_AGENTS
    assert info["obs_shape"] == OBS_SIZE
    assert env.get_obs_size() == OBS_SIZE
    assert info["episode_limit"] == LIMIT

    try:
        env.get_obs()
        assert False, "Should raise ValueError because the environment has not yet been reset"
    except ValueError:
        pass

    env.reset()
    obs = env.get_obs()
    assert obs.shape == (N_AGENTS, OBS_SIZE)
    state = env.get_state()
    assert len(state.shape) == 1 and state.shape[0] == env.get_state_size()

    for _ in range(LIMIT - 1):
        reward, done, _ = env.step([0] * N_AGENTS)
        assert reward == REWARD_STEP
        assert not done
    reward, done, _ = env.step([0] * N_AGENTS)
    assert reward == REWARD_STEP
    assert done
