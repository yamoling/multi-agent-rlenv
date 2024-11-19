import numpy as np
from typing import Any
from marlenv import Builder, DiscreteMOMockEnv, DiscreteMockEnv
from marlenv.wrappers import Centralised, AvailableActionsMask
import marlenv


def test_padding():
    PAD_SIZE = 2
    mock = DiscreteMockEnv(5)
    env = Builder(mock).pad("extra", PAD_SIZE).build()
    assert env.extra_shape == (PAD_SIZE + mock.extra_shape[0],)

    mock = DiscreteMockEnv(5)
    env = Builder(mock).pad("obs", PAD_SIZE).build()
    assert env.observation_shape == (mock.observation_shape[0] + PAD_SIZE,)


def test_available_actions():
    N_AGENTS = 5
    mock = DiscreteMockEnv(N_AGENTS)
    env = Builder(mock).available_actions().build()

    assert env.extra_shape == (5 + mock.extra_shape[0],)
    obs = env.reset()
    assert np.array_equal(obs.extras, np.ones((N_AGENTS, env.n_actions), dtype=np.float32))


def test_agent_id():
    env = Builder(DiscreteMockEnv(5)).agent_id().build()

    assert env.extra_shape == (5,)
    obs = env.reset()
    assert np.array_equal(obs.extras, np.identity(5, dtype=np.float32))


def test_penalty_wrapper():
    N_OBJECTIVES = 5
    mock = DiscreteMOMockEnv(1, N_OBJECTIVES, reward_step=1)
    env = Builder(mock).time_penalty(0.1).build()
    expected = np.array([0.9] * N_OBJECTIVES, dtype=np.float32)
    done = False
    while not done:
        _, reward, done, *_ = env.step(np.array([0], dtype=np.int64))
        assert np.array_equal(reward, expected)


def test_time_limit_wrapper():
    MAX_T = 5
    env = Builder(DiscreteMockEnv(1)).time_limit(MAX_T).build()
    assert env.extra_shape == (1,)
    done = False
    t = 0
    while not done:
        obs, _, done, truncated, _ = env.step(np.array([0]))
        assert obs.extras.shape == (env.n_agents, 1)
        assert obs.state_extras.shape == (1,)
        t += 1
    assert t == MAX_T
    assert truncated
    assert done


def test_truncated_and_done():
    END_GAME = 10
    env = marlenv.wrappers.TimeLimit(DiscreteMockEnv(2, end_game=END_GAME), END_GAME)
    obs = env.reset()
    episode = marlenv.EpisodeBuilder()
    done = truncated = False
    while not episode.is_finished:
        action = env.action_space.sample()
        next_obs, r, done, truncated, info = env.step(action)
        episode.add(marlenv.Transition(obs, action, r, done, info, next_obs, truncated))
        obs = next_obs
    assert done
    assert not truncated, "The episode is done, so it does not have to be truncated even though the time limit is reached at the same time."
    episode = episode.build()

    assert np.all(episode.dones[:-1] == 0)
    assert episode.dones[-1] == 1


def test_time_limit_wrapper_with_extra():
    """
    When an extra is given as input, the environment should be 'done' and 'truncated' when the time limit is reached.
    """
    MAX_T = 5
    env = Builder(DiscreteMockEnv(5)).time_limit(MAX_T, add_extra=True).build()
    assert env.extra_shape == (1,)
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


def test_wrong_truncation_penalty():
    try:
        Builder(DiscreteMockEnv(1)).time_limit(5, add_extra=True, truncation_penalty=-0.1).build()
        assert False, "It should not be possible to set a negative truncation penalty"
    except AssertionError:
        pass

    try:
        Builder(DiscreteMockEnv(1)).time_limit(5, add_extra=False, truncation_penalty=0.1).build()
        assert False, "It should not be possible to set a truncation penalty without adding the extra feature"
    except AssertionError:
        pass


def test_time_limit_wrapper_with_truncation_penalty():
    MAX_T = 5
    env = Builder(DiscreteMockEnv(5)).time_limit(MAX_T, add_extra=True, truncation_penalty=0.1).build()
    assert env.extra_shape == (1,)
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
    def test(env: marlenv.MARLEnv[Any]):
        obs = env.reset()
        assert np.any(obs.data != 0)
        obs, r, done, truncated, info = env.step(env.action_space.sample())
        assert np.all(obs.data == 0)

    env = marlenv.Builder(DiscreteMockEnv(5)).blind(p=1).build()
    test(env)
    env = marlenv.wrappers.Blind(DiscreteMockEnv(5), p=1)
    test(env)


def test_last_action():
    env = Builder(DiscreteMockEnv(2)).last_action().build()
    assert env.extra_shape == (env.n_actions,)
    obs = env.reset()
    assert np.all(obs.extras == 0)
    obs, _, _, _, _ = env.step(np.array([0, 1]))
    one_hot_actions = np.zeros((2, env.n_actions), dtype=np.float32)
    one_hot_actions[0, 0] = 1.0
    one_hot_actions[1, 1] = 1.0
    assert np.all(obs.extras == one_hot_actions)


def test_centralised_shape():
    mock = DiscreteMockEnv(2)
    env = Builder(mock).centralised().time_limit(50, True).build()
    assert env.observation_shape == (2 * mock.obs_size,)
    assert env.n_agents == 1
    assert env.n_actions == mock.n_actions**2
    assert env.extra_shape == (1,)
    obs = env.reset()
    assert obs.data.shape == (1, *env.observation_shape)
    assert obs.extras.shape == (1, *env.extra_shape)


def test_centralised_action():
    mock = DiscreteMockEnv(2)
    env = Centralised(mock)
    for action1 in range(mock.n_actions):
        for action2 in range(mock.n_actions):
            joint_action = action1 * mock.n_actions + action2
            expected_individual_actions = np.array([action1, action2])
            individual_actions = env._individual_actions(joint_action)
            assert np.array_equal(individual_actions, expected_individual_actions)


def test_centralised_obs_and_state():
    wrapped = DiscreteMockEnv(2)
    env = Centralised(wrapped)
    assert env.observation_shape == (2 * wrapped.obs_size,)
    assert env.state_shape == (wrapped.agent_state_size * wrapped.n_agents,)
    obs = env.reset()
    assert obs.data.shape == (1, *env.observation_shape)
    assert obs.state.shape == env.state_shape
    obs, *_ = env.step(np.array([0]))
    assert obs.data.shape == (1, *env.observation_shape)
    assert obs.state.shape == env.state_shape


def test_centralised_available_actions():
    N_AGENTS = 2
    mock = DiscreteMockEnv(N_AGENTS)
    env = Builder(mock).centralised().build()
    available = env.available_actions()
    assert available.shape == (1, mock.n_actions**N_AGENTS)
    assert np.all(available == 1)

    mask = np.zeros((N_AGENTS, mock.n_actions), dtype=np.bool_)
    mask[0, 0] = True
    mask[1, 0] = True
    env = Centralised(AvailableActionsMask(mock, mask))
    expected_joint_mask = np.zeros((1, mock.n_actions**N_AGENTS))
    expected_joint_mask[0, 0] = 1
    obs = env.reset()
    assert np.array_equal(obs.available_actions, expected_joint_mask)
    obs, *_ = env.step([0])
    assert np.array_equal(obs.available_actions, expected_joint_mask)


def test_available_action_mask():
    N_AGENTS = 2
    N_ACTIONS = 5
    wrapped = DiscreteMockEnv(N_AGENTS, n_actions=N_ACTIONS)

    try:
        AvailableActionsMask(wrapped, np.zeros((N_AGENTS, N_ACTIONS), dtype=bool))
        assert False, "It should not be possible to mask all actions"
    except AssertionError:
        pass

    try:
        AvailableActionsMask(wrapped, np.zeros((N_AGENTS, N_ACTIONS + 1), dtype=bool))
        assert False, "It should not be possible to mask all actions"
    except AssertionError:
        pass

    mask = np.array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]], dtype=bool)
    env = AvailableActionsMask(wrapped, mask)
    obs = env.reset()
    assert np.array_equal(obs.available_actions, mask)
    obs, *_ = env.step([0, 1])
    assert np.array_equal(obs.available_actions, mask)


def test_wrapper_reward_shape():
    mock = DiscreteMOMockEnv(1)
    env = Builder(mock).time_penalty(0.1).last_action().available_actions().build()

    assert mock.is_multi_objective == env.is_multi_objective
    assert mock.reward_space == env.reward_space


def test_builder_action_mask():
    env = DiscreteMockEnv()
    mask = np.full((env.n_agents, env.n_actions), True)
    mask[0, 0] = False
    mask[1, 1] = False
    new_env = marlenv.Builder(env).mask_actions(mask).build()
    assert env.extra_shape == new_env.extra_shape
