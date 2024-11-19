try:
    import gymnasium

    skip_gym = False
except ImportError:
    skip_gym = True

try:
    import pettingzoo

    skip_pettingzoo = False
except ImportError:
    skip_pettingzoo = True

try:
    import smac

    skip_smac = False
except ImportError:
    skip_smac = True


import marlenv
import numpy as np

from marlenv import Observation, DiscreteMockEnv, MARLEnv, DiscreteActionSpace
from marlenv.adapters import PymarlAdapter
import pytest


@pytest.mark.skipif(skip_gym, reason="Gymnasium is not installed")
def test_gym_adapter():
    # Discrete action space
    env = marlenv.make("CartPole-v1")
    env.reset()
    assert isinstance(env, MARLEnv)
    # assert not isinstance(env, marlenv.MOMARLEnv)
    assert env.n_actions == 2
    assert env.n_agents == 1

    obs, r, done, truncated, info = env.step(env.action_space.sample())
    assert isinstance(obs, Observation)
    assert isinstance(r, float)
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

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
    obs, r, done, truncated, info = env.step(action)
    assert isinstance(obs, Observation)
    assert isinstance(r, np.ndarray)
    assert r.shape == (1,)
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert env.n_agents == 8
    assert env.n_actions == 5
    assert isinstance(env.action_space, marlenv.DiscreteActionSpace)


@pytest.mark.skipif(skip_pettingzoo, reason="PettingZoo is not installed")
def test_pettingzoo_adapter_continuous_action():
    from pettingzoo.sisl import waterworld_v4

    # https://pettingzoo.farama.org/environments/sisl/waterworld/
    env = marlenv.adapters.PettingZoo(waterworld_v4.parallel_env())
    env.reset()
    action = env.action_space.sample()
    obs, r, done, truncated, info = env.step(action)
    assert isinstance(obs, Observation)
    assert isinstance(r, np.ndarray)
    assert r.shape == (1,)
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert env.n_actions == 2
    assert env.n_agents == 2
    assert isinstance(env.action_space, marlenv.ContinuousActionSpace)


def _check_env_3m(env: MARLEnv):
    obs = env.reset()
    assert isinstance(obs, Observation)
    assert env.n_agents == 3
    assert isinstance(env.action_space, DiscreteActionSpace)

    obs, r, done, truncated, info = env.step(env.action_space.sample(env.available_actions()))
    assert isinstance(obs, Observation)
    assert isinstance(r, np.ndarray)
    assert r.shape == (1,)
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


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
    env.render("human")


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
