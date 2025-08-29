import numpy as np
import pytest

import marlenv
from marlenv import ContinuousSpace, DiscreteMockEnv, MARLEnv, Observation, State, MultiDiscreteSpace
from marlenv.adapters import PymarlAdapter

skip_gym = not marlenv.adapters.HAS_GYM
skip_pettingzoo = not marlenv.adapters.HAS_PETTINGZOO
skip_smac = not marlenv.adapters.HAS_SMAC


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
    from pettingzoo.mpe import simple_v3

    env = marlenv.adapters.PettingZoo(simple_v3.parallel_env(continuous_actions=True))
    env.reset()
    action = env.action_space.sample()
    step = env.step(action)
    assert isinstance(step.obs, Observation)
    assert isinstance(step.reward, np.ndarray)
    assert step.reward.shape == (1,)
    assert isinstance(step.done, bool)
    assert isinstance(step.truncated, bool)
    assert isinstance(step.info, dict)
    assert env.n_actions == 5
    assert env.n_agents == 1
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
