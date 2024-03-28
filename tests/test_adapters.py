import numpy as np
import rlenv

from rlenv.adapters import PymarlAdapter
from rlenv import MockEnv, Observation, DiscreteActionSpace

try:

    def test_gym_adapter():
        # Discrete action space
        env = rlenv.make("CartPole-v1")
        env.reset()
        assert env.n_actions == 2
        assert env.n_agents == 1
        assert env.reward_size == 1

        obs, r, done, truncated, info = env.step(env.action_space.sample())
        assert isinstance(obs, Observation)
        assert isinstance(r, np.ndarray)
        assert r.shape == (1,)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        # Continuous action space
        env = rlenv.make("Pendulum-v1")
        env.reset()
except ImportError:
    # Skip the test if gym is not installed
    pass

try:
    from pettingzoo.sisl import pursuit_v4, waterworld_v4

    def test_pettingzoo_adapter_discrete_action():
        # https://pettingzoo.farama.org/environments/sisl/pursuit/#pursuit
        env = rlenv.make(pursuit_v4.parallel_env())
        env.reset()
        obs, r, done, truncated, info = env.step(env.action_space.sample())
        assert isinstance(obs, Observation)
        assert isinstance(r, np.ndarray)
        assert r.shape == (1,)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert env.n_agents == 8
        assert env.n_actions == 5
        assert isinstance(env.action_space, rlenv.DiscreteActionSpace)

    def test_pettingzoo_adapter_continuous_action():
        # https://pettingzoo.farama.org/environments/sisl/waterworld/
        env = rlenv.make(waterworld_v4.parallel_env())
        env.reset()
        obs, r, done, truncated, info = env.step(env.action_space.sample())
        assert isinstance(obs, Observation)
        assert isinstance(r, np.ndarray)
        assert r.shape == (1,)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert env.n_actions == 2
        assert env.n_agents == 2
        assert isinstance(env.action_space, rlenv.ContinuousActionSpace)
except ImportError:
    # Skip the test if pettingzoo is not installed
    pass

# Only perform the tests if SMAC is installed.
try:
    from rlenv.adapters import SMAC
    from smac.env import StarCraft2Env

    def _check_env_3m(env: SMAC):
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

    def test_smac_from_class():
        env = SMAC(StarCraft2Env("3m"))
        _check_env_3m(env)

    def test_smac_from_str():
        env = SMAC("3m")
        _check_env_3m(env)

    def test_smac_render():
        env = SMAC("3m")
        env.reset()
        env.render("human")
except ImportError:
    # Skip the test if SMAC is not installed
    pass


def test_pymarl():
    LIMIT = 20
    N_AGENTS = 2
    env = PymarlAdapter(MockEnv(N_AGENTS), LIMIT)

    info = env.get_env_info()
    assert info["n_agents"] == N_AGENTS
    assert info["n_actions"] == MockEnv.N_ACTIONS
    assert env.get_total_actions() == MockEnv.N_ACTIONS
    assert info["state_shape"] == MockEnv.UNIT_STATE_SIZE * N_AGENTS
    assert env.get_state_size() == MockEnv.UNIT_STATE_SIZE * N_AGENTS
    assert info["obs_shape"] == MockEnv.OBS_SIZE
    assert env.get_obs_size() == MockEnv.OBS_SIZE
    assert env.episode_limit == LIMIT

    try:
        env.get_obs()
        assert False, "Should raise ValueError because the environment has not yet been reset"
    except ValueError:
        pass

    env.reset()
    obs = env.get_obs()
    assert obs.shape == (N_AGENTS, MockEnv.OBS_SIZE)
    state = env.get_state()
    assert len(state.shape) == 1 and state.shape[0] == env.get_state_size()

    for _ in range(LIMIT - 1):
        reward, done, _ = env.step([0] * N_AGENTS)
        assert reward == MockEnv.REWARD_STEP
        assert not done
    reward, done, _ = env.step([0] * N_AGENTS)
    assert reward == MockEnv.REWARD_STEP
    assert done
