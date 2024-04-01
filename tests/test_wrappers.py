import numpy as np
from rlenv import Builder, MockEnv
from rlenv.wrappers import Centralised, AvailableActionsMask
import rlenv


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
    N_OBJECTIVES = 5
    env = Builder(MockEnv(1, N_OBJECTIVES)).time_penalty(0.1).build()
    done = False
    while not done:
        _, reward, done, *_ = env.step(np.array([0]))
        assert reward == [MockEnv.REWARD_STEP - 0.1] * N_OBJECTIVES


def test_time_limit_wrapper():
    MAX_T = 5
    env = Builder(MockEnv(1)).time_limit(MAX_T).build()
    assert env.extra_feature_shape == (0,)
    stop = False
    t = 0
    while not stop:
        _, _, done, truncated, _ = env.step(np.array([0]))
        stop = done or truncated
        assert not done
        t += 1
    assert t == MAX_T
    assert truncated
    assert not done


def test_time_limit_wrapper_with_extra():
    """
    When an extra is given as input, the environment should be 'done' and 'truncated' when the time limit is reached.
    """
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
    assert done
    assert not truncated


def test_time_limit_wrapper_with_truncation_penalty():
    MAX_T = 5
    env = Builder(MockEnv(5)).time_limit(MAX_T, add_extra=True, truncation_penalty=0.1).build()
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
    assert done
    assert truncated


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


def test_centralised_shape():
    env = Builder(MockEnv(2)).centralised().time_limit(50, True).build()
    assert env.observation_shape == (2 * MockEnv.OBS_SIZE,)
    assert env.n_agents == 1
    assert env.n_actions == MockEnv.N_ACTIONS**2
    assert env.extra_feature_shape == (1,)


def test_centralised_action():
    env = Centralised(MockEnv(2))
    for action1 in range(MockEnv.N_ACTIONS):
        for action2 in range(MockEnv.N_ACTIONS):
            joint_action = action1 * MockEnv.N_ACTIONS + action2
            expected_individual_actions = np.array([action1, action2])
            individual_actions = env._individual_actions(joint_action)
            assert np.array_equal(individual_actions, expected_individual_actions)


def test_centralised_obs_and_state():
    wrapped = MockEnv(2)
    env = Centralised(wrapped)
    assert env.observation_shape == (2 * MockEnv.OBS_SIZE,)
    assert env.state_shape == (MockEnv.UNIT_STATE_SIZE * wrapped.n_agents,)
    obs = env.reset()
    assert obs.data.shape == (1, *env.observation_shape)
    assert obs.state.shape == env.state_shape
    obs, *_ = env.step(np.array([0]))
    assert obs.data.shape == (1, *env.observation_shape)
    assert obs.state.shape == env.state_shape


def test_centralised_available_actions():
    N_AGENTS = 2
    mock = MockEnv(N_AGENTS)
    env = Builder(mock).centralised().build()
    available = env.available_actions()
    assert available.shape == (1, MockEnv.N_ACTIONS**N_AGENTS)
    assert np.all(available == 1)

    mask = np.zeros((N_AGENTS, MockEnv.N_ACTIONS))
    mask[0, 0] = 1
    mask[1, 0] = 1
    env = Centralised(AvailableActionsMask(mock, mask))
    expected_joint_mask = np.zeros((1, MockEnv.N_ACTIONS**N_AGENTS))
    expected_joint_mask[0, 0] = 1
    obs = env.reset()
    assert np.array_equal(obs.available_actions, expected_joint_mask)
    obs, *_ = env.step([0])
    assert np.array_equal(obs.available_actions, expected_joint_mask)


def test_available_action_mask():
    N_AGENTS = 2
    wrapped = MockEnv(N_AGENTS)

    try:
        AvailableActionsMask(wrapped, np.zeros((N_AGENTS, MockEnv.N_ACTIONS)))
        assert False, "It should not be possible to mask all actions"
    except AssertionError:
        pass

    try:
        AvailableActionsMask(wrapped, np.zeros((N_AGENTS, MockEnv.N_ACTIONS + 1)))
        assert False, "It should not be possible to mask all actions"
    except AssertionError:
        pass

    mask = np.array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]])
    env = AvailableActionsMask(wrapped, mask)
    obs = env.reset()
    assert np.array_equal(obs.available_actions, mask)
    obs, *_ = env.step([0, 1])
    assert np.array_equal(obs.available_actions, mask)
