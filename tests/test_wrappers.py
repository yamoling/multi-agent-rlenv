import numpy as np
from rlenv import Builder
import rlenv
from .mock_env import MockEnv


def test_padding():
    PAD_SIZE = 2
    env = Builder(MockEnv(5)).pad("extra", PAD_SIZE).build()
    assert env.extra_feature_shape == (PAD_SIZE,)

    env = Builder(MockEnv(5)).pad("obs", PAD_SIZE).build()
    assert env.observation_shape == (MockEnv.OBS_SIZE + PAD_SIZE,)


def test_available_actions():
    N_AGENTS = 5
    env = Builder(MockEnv(N_AGENTS)).available_actions().build()

    assert env.extra_feature_shape == (5,)
    obs = env.reset()
    assert np.array_equal(obs.extras, np.ones((N_AGENTS, MockEnv.N_ACTIONS), dtype=np.float32))


def test_agent_id():
    env = Builder(MockEnv(5)).agent_id().build()

    assert env.extra_feature_shape == (5,)
    obs = env.reset()
    assert np.array_equal(obs.extras, np.identity(5, dtype=np.float32))


def test_penalty_wrapper():
    env = Builder(MockEnv(1)).time_penalty(0.1).build()
    done = False
    while not done:
        _, reward, done, *_ = env.step(np.array([0]))
        assert reward == MockEnv.REWARD_STEP - 0.1


def test_time_limit_wrapper():
    MAX_T = 5
    env = Builder(MockEnv(1)).time_limit(MAX_T).build()
    assert env.extra_feature_shape == (0,)
    stop = False
    t = 0
    while not stop:
        _, _, done, truncated, _ = env.step(np.array([0]))
        stop = done or truncated
        t += 1
    assert t == MAX_T


def test_time_limit_wrapper_with_extra():
    MAX_T = 5
    env = Builder(MockEnv(5)).time_limit(MAX_T, add_extra=True).build()
    assert env.extra_feature_shape == (1,)
    obs = env.reset()
    assert obs.extras.shape == (5, 1)
    stop = False
    t = 0
    while not stop:
        obs, _, done, truncated, _ = env.step(np.array([0]))
        stop = done or truncated
        t += 1
    assert t == MAX_T
    assert np.all(obs.extras[:] == 1)


def test_blind_wrapper():
    def test(env: rlenv.RLEnv):
        obs = env.reset()
        assert np.any(obs.data != 0)
        obs, r, done, truncated, info = env.step(env.action_space.sample())
        assert np.all(obs.data == 0)

    env = rlenv.Builder(MockEnv(5)).blind(p=1).build()
    test(env)
    env = rlenv.wrappers.Blind(MockEnv(5), p=1)
    test(env)


def test_last_action():
    env = Builder(MockEnv(2)).last_action().build()
    assert env.extra_feature_shape == (env.n_actions,)
    obs = env.reset()
    assert np.all(obs.extras == 0)
    obs, _, _, _, _ = env.step(np.array([0, 1]))
    one_hot_actions = np.zeros((2, env.n_actions), dtype=np.float32)
    one_hot_actions[0, 0] = 1.0
    one_hot_actions[1, 1] = 1.0
    assert np.all(obs.extras == one_hot_actions)
