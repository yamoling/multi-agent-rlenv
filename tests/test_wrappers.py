import numpy as np
import pytest

import marlenv
from marlenv import Builder, MARLEnv, catalog
from marlenv.catalog import DiscreteMOMockEnv
from marlenv.wrappers import ActionRandomizer, AvailableActionsMask, Centralized, DelayedReward, EnvPool, LastAction, TimeLimit


def test_padding():
    PAD_SIZE = 2
    mock = catalog.DiscreteMockEnv(5)
    env = Builder(mock).pad("extra", PAD_SIZE).build()
    assert env.extras_shape == (PAD_SIZE + mock.extras_shape[0],)
    env.reset()
    for _ in range(10):
        env.step(env.action_space.sample())

    mock = catalog.DiscreteMockEnv(5)
    env = Builder(mock).pad("obs", PAD_SIZE).build()
    assert env.observation_shape == (mock.observation_shape[0] + PAD_SIZE,)
    env.reset()
    for _ in range(10):
        env.step(env.action_space.sample())


def test_extras_padding_labels():
    PAD_SIZE = 2
    mock = catalog.DiscreteMockEnv(5)
    env = Builder(mock).pad("extra", PAD_SIZE, label="the-extras").build()
    assert all(m.startswith("the-extras") for m in env.extras_meanings)

    env = Builder(mock).pad("extra", PAD_SIZE).build()
    assert all(m.startswith("Padding") for m in env.extras_meanings)


def test_available_actions():
    N_AGENTS = 5
    mock = catalog.DiscreteMockEnv(N_AGENTS)
    env = Builder(mock).available_actions().build()

    assert env.extras_shape == (5 + mock.extras_shape[0],)
    obs, _ = env.reset()
    assert np.array_equal(obs.extras, np.ones((N_AGENTS, env.n_actions), dtype=np.float32))


def test_agent_id():
    env = Builder(catalog.DiscreteMockEnv(5)).agent_id().build()
    assert env.extras_shape == (5,)
    obs, _ = env.reset()
    assert np.array_equal(obs.extras, np.identity(5, dtype=np.float32))


def test_penalty_wrapper():
    N_OBJECTIVES = 5
    mock = DiscreteMOMockEnv(1, N_OBJECTIVES, reward_step=1)
    env = Builder(mock).time_penalty(0.1).build()
    expected = np.array([0.9] * N_OBJECTIVES, dtype=np.float32)
    done = False
    while not done:
        step = env.step([0])
        done = step.done
        assert np.array_equal(step.reward, expected)


def test_time_limit_wrapper():
    MAX_T = 5
    env = Builder(catalog.DiscreteMockEnv(1)).time_limit(MAX_T).build()
    assert env.extras_shape == (1,)
    assert env.state_extra_shape == (1,)
    t = 1
    step = env.step(np.array([0]))
    while not step.done:
        assert step.obs.extras.shape == (env.n_agents, 1)
        assert step.state.extras_shape == (1,)
        step = env.step(np.array([0]))
        t += 1
    assert t == MAX_T
    assert step.truncated
    assert step.done


def test_truncated_and_done():
    END_GAME = 10
    env = marlenv.wrappers.TimeLimit(catalog.DiscreteMockEnv(2, end_game=END_GAME), END_GAME)
    obs, state = env.reset()
    episode = marlenv.Episode.new(obs, state)
    action = env.action_space.sample()
    step = env.step(action)
    while not episode.is_finished:
        episode.add(marlenv.Transition.from_step(obs, state, action, step))
        obs = step.obs
        state = step.state
        action = env.action_space.sample()
        step = env.step(action)

    assert step.done
    assert not step.truncated, (
        "The episode is done, so it does not have to be truncated even though the time limit is reached at the same time."
    )

    assert np.all(episode.dones[:-1] == 0)
    assert episode.dones[-1] == 1


def test_time_limit_wrapper_with_extra():
    """
    When an extra is given as input, the environment should be 'done' and 'truncated' when the time limit is reached.
    """
    MAX_T = 5
    env = Builder(catalog.DiscreteMockEnv(5)).time_limit(MAX_T, add_extra=True).build()
    assert env.extras_shape == (1,)
    obs, _ = env.reset()
    assert obs.extras.shape == (5, 1)
    t = 1
    step = env.step(np.array([0]))
    while not step.is_terminal:
        step = env.step(np.array([0]))
        t += 1
    assert t == MAX_T
    assert np.all(step.obs.extras == 1.0)
    assert step.done
    assert step.truncated


def test_wrong_truncation_penalty():
    try:
        Builder(catalog.DiscreteMockEnv(1)).time_limit(5, add_extra=True, truncation_penalty=-0.1).build()
        assert False, "It should not be possible to set a negative truncation penalty"
    except AssertionError:
        pass

    try:
        Builder(catalog.DiscreteMockEnv(1)).time_limit(5, add_extra=False, truncation_penalty=0.1).build()
        assert False, "It should not be possible to set a truncation penalty without adding the extra feature"
    except AssertionError:
        pass


def test_time_limit_wrapper_with_truncation_penalty():
    MAX_T = 5
    env = Builder(catalog.DiscreteMockEnv(5)).time_limit(MAX_T, add_extra=True, truncation_penalty=0.1).build()
    assert env.extras_shape == (1,)
    obs, _ = env.reset()
    assert obs.extras.shape == (5, 1)
    t = 1
    step = env.step(np.array([0]))
    while not step.is_terminal:
        step = env.step(np.array([0]))
        t += 1
    assert t == MAX_T
    assert np.all(step.obs.extras[:] == 1)
    assert step.done
    assert step.truncated


def test_blind_wrapper():
    def test(env: marlenv.MARLEnv):
        obs, _ = env.reset()
        assert np.any(obs.data != 0)
        step = env.step(env.action_space.sample())
        assert np.all(step.obs.data == 0)

    env = marlenv.Builder(catalog.DiscreteMockEnv(5)).blind(p=1).build()
    test(env)
    env = marlenv.wrappers.Blind(catalog.DiscreteMockEnv(5), p=1)
    test(env)


def test_last_action():
    env = Builder(catalog.DiscreteMockEnv(2)).last_action().build()
    assert env.extras_shape == (env.n_actions,)
    obs, _ = env.reset()
    assert np.all(obs.extras == 0)
    step = env.step(np.array([0, 1]))
    one_hot_actions = np.zeros((2, env.n_actions), dtype=np.float32)
    one_hot_actions[0, 0] = 1.0
    one_hot_actions[1, 1] = 1.0
    assert np.all(step.obs.extras == one_hot_actions)


def test_centralised_shape():
    mock = catalog.DiscreteMockEnv(2)
    env = Builder(mock).centralised().time_limit(50, True).build()
    assert env.observation_shape == (2 * mock.obs_size,)
    assert env.n_agents == 1
    assert env.n_actions == mock.n_actions**2
    assert env.extras_shape == (1,)
    obs, _ = env.reset()
    assert obs.data.shape == (1, *env.observation_shape)
    assert obs.extras.shape == (1, *env.extras_shape)


def test_centralised_action():
    mock = catalog.DiscreteMockEnv(2)
    env = Centralized(mock)
    for action1 in range(mock.n_actions):
        for action2 in range(mock.n_actions):
            joint_action = action1 * mock.n_actions + action2
            expected_individual_actions = np.array([action1, action2])
            individual_actions = env._individual_actions(joint_action)
            assert np.array_equal(individual_actions, expected_individual_actions)


def test_centralised_obs_and_state():
    wrapped = catalog.DiscreteMockEnv(2)
    env = Centralized(wrapped)
    assert env.observation_shape == (2 * wrapped.obs_size,)
    assert env.state_shape == (wrapped.agent_state_size * wrapped.n_agents,)
    obs, state = env.reset()
    assert obs.data.shape == (1, *env.observation_shape)
    assert state.data.shape == env.state_shape
    step = env.step(np.array([0]))
    assert step.obs.data.shape == (1, *env.observation_shape)
    assert step.state.data.shape == env.state_shape


def test_centralised_available_actions():
    N_AGENTS = 2
    mock = catalog.DiscreteMockEnv(N_AGENTS)
    env = Builder(mock).centralised().build()
    available = env.available_actions()
    assert available.shape == (1, mock.n_actions**N_AGENTS)
    assert np.all(available == 1)

    mask = np.zeros((N_AGENTS, mock.n_actions), dtype=np.bool)
    mask[0, 0] = True
    mask[1, 0] = True
    env = Centralized(AvailableActionsMask(mock, mask))
    expected_joint_mask = np.zeros((1, mock.n_actions**N_AGENTS))
    expected_joint_mask[0, 0] = 1
    obs, _ = env.reset()
    assert np.array_equal(obs.available_actions, expected_joint_mask)
    step = env.step([0])
    assert np.array_equal(step.obs.available_actions, expected_joint_mask)


def test_available_action_mask():
    N_AGENTS = 2
    N_ACTIONS = 5
    wrapped = catalog.DiscreteMockEnv(N_AGENTS, n_actions=N_ACTIONS)

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
    obs, _ = env.reset()
    assert np.array_equal(obs.available_actions, mask)
    step = env.step([0, 1])
    assert np.array_equal(step.obs.available_actions, mask)


def test_mask_actions_builder_int():
    N_AGENTS = 5
    mock = catalog.DiscreteMockEnv(N_AGENTS)
    for prevented_action in range(mock.n_actions):
        env = Builder(mock).mask_actions(prevented_action).build()
        obs, _ = env.reset()
        for agent_available_actions in obs.available_actions:
            assert not agent_available_actions[prevented_action].item()


def test_mask_actions_builder_int_list():
    mock = catalog.DiscreteMockEnv()
    for prevented_actions in range(1, mock.n_actions - 1):
        prevented_actions = list(range(prevented_actions))
        env = Builder(mock).mask_actions(prevented_actions).build()
        obs, _ = env.reset()
        for agent_available_actions in obs.available_actions:
            for prevented in prevented_actions:
                assert not agent_available_actions[prevented].item()


def test_mask_actions_builder_errors_too_many_actions_in_list():
    mock = catalog.DiscreteMockEnv()
    try:
        Builder(mock).mask_actions(list(range(mock.n_actions + 1))).build()
        raise Exception("It should not be possible to mask actions with more actions than the environment provides")
    except AssertionError:
        pass


def test_mask_actions_builder_errors_action_index_out_of_bounds():
    mock = catalog.DiscreteMockEnv()
    try:
        Builder(mock).mask_actions(mock.n_actions + 1).build()
        raise Exception(
            "It should not be possible to mask an action whose index is higher than the number of actions provided by the environment"
        )
    except AssertionError:
        pass


def test_mask_actions_builder_errors_invalid_shape_too_many_actions():
    mock = catalog.DiscreteMockEnv()
    try:
        Builder(mock).mask_actions(np.full((mock.n_agents, mock.n_actions + 1), True)).build()
        raise Exception("It should not be possible to mask actions with an invalid input shape")
    except AssertionError:
        pass


def test_mask_actions_builder_errors_invalid_shape_too_many_agents():
    mock = catalog.DiscreteMockEnv()
    try:
        Builder(mock).mask_actions(np.full((mock.n_agents + 1, mock.n_actions), True)).build()
        raise Exception("It should not be possible to mask actions with an invalid input")
    except AssertionError:
        pass


def test_mask_actions_builder_errors_all_actions_masked():
    mock = catalog.DiscreteMockEnv()
    try:
        Builder(mock).mask_actions(np.full((mock.n_agents, mock.n_actions), False)).build()
        raise Exception("At least one action should remain available for each agent.")
    except AssertionError:
        pass


def test_mask_actions_builder_from_bool_mask():
    mock = catalog.DiscreteMockEnv()
    mask = [i % 2 == 0 for i in range(mock.n_actions)]
    _ = Builder(mock).mask_actions(mask).build()


def test_wrapper_reward_shape():
    mock = DiscreteMOMockEnv(1)
    env = Builder(mock).time_penalty(0.1).last_action().available_actions().build()

    assert mock.is_multi_objective == env.is_multi_objective
    assert mock.reward_space == env.reward_space


def test_builder_action_mask():
    env = catalog.DiscreteMockEnv()
    mask = np.full((env.n_agents, env.n_actions), True)
    mask[0, 0] = False
    mask[1, 1] = False
    new_env = marlenv.Builder(env).mask_actions(mask).build()
    assert env.extras_shape == new_env.extras_shape


def test_time_limit_set_state():
    env = TimeLimit(catalog.DiscreteMockEnv(end_game=50), 100)
    env.reset()

    states = []
    for i in range(50):
        states.append(env.get_state())
        env.step(env.action_space.sample())
        assert env._current_step == i + 1

    for i, s in enumerate(states):
        env.set_state(s)
        assert env._current_step == i


def test_last_action_set_state():
    env = LastAction(catalog.DiscreteMockEnv())
    env.reset()

    states = []
    one_hot_actions = [np.zeros((env.n_agents, env.n_actions), dtype=np.float32)]
    for _ in range(50):
        states.append(env.get_state())
        action = env.action_space.sample()
        env.step(action)
        one_hots = np.zeros((env.n_agents, env.n_actions), dtype=np.float32)
        index = np.arange(env.n_agents)
        one_hots[index, action] = 1.0
        one_hot_actions.append(one_hots)

    for i, s in enumerate(states):
        last_action = one_hot_actions[i]
        env.set_state(s)

        assert env.last_one_hot_actions is not None
        assert np.array_equal(env.last_one_hot_actions, last_action)


def test_wrong_extra_meanings():
    from marlenv.wrappers import RLEnvWrapper

    try:
        RLEnvWrapper(catalog.DiscreteMockEnv(), extra_meanings=["a", "b"])
        assert False, "It should not be possible to set extra meanings without setting the extra shape"
    except ValueError:
        pass


def test_extra_meanings():
    from marlenv.wrappers import RLEnvWrapper

    env = RLEnvWrapper(catalog.DiscreteMockEnv(extras_size=0), extra_shape=(2,), extra_meanings=["added extra 1", "added extra 2"])
    assert env.extras_shape == (2,)
    assert env.extras_meanings == ["added extra 1", "added extra 2"]


def test_wrapper_extra_names():
    env = catalog.DiscreteMockEnv(extras_size=0)
    env = TimeLimit(env, 100, add_extra=True)
    assert env.extras_meanings[-1] == "Time ratio"
    env = LastAction(env)
    assert env.extras_meanings[-1] == "Last action"
    assert env.extras_meanings == ["Time ratio"] + ["Last action"] * env.n_actions


def _test_delayed_rewards(env: MARLEnv):
    assert isinstance(env, DelayedReward)
    assert isinstance(env.wrapped, catalog.DiscreteMockEnv)
    expected = []
    for i in range(env.wrapped.end_game):
        if i < env.delay:
            expected.append(np.zeros(env.reward_space.shape, dtype=np.float32))
        elif i < env.wrapped.end_game - 1:
            expected.append(env.wrapped.reward_step)
        else:
            expected.append(env.wrapped.reward_step * (env.delay + 1))
    env.reset()
    for i in range(env.wrapped.end_game):
        step = env.random_step()
        assert np.array_equal(step.reward, expected[i])


def test_delayed_rewards():
    env = catalog.DiscreteMockEnv(reward_step=[1, 2, 3], end_game=5, n_agents=2)
    env = DelayedReward(env, 2)
    _test_delayed_rewards(env)


def test_delayed_rewards_from_builder():
    for delay in range(0, 10):
        for end_game in range(delay + 1, delay * 2):
            env = Builder(catalog.DiscreteMockEnv(reward_step=10, end_game=end_game, n_agents=2)).delay_rewards(delay).build()
            _test_delayed_rewards(env)


def test_potential_shaping():
    from marlenv.wrappers.potential_shaping import PotentialShaping

    class PS(PotentialShaping):
        def __init__(self, env: MARLEnv):
            self.phi = 10
            super().__init__(env)

        def reset(self, *, seed: int | None = None):
            self.phi = 10
            return super().reset()

        def compute_potential(self) -> float:
            return self.phi

        def step(self, action):
            self.phi = max(0, self.phi - 1)
            return super().step(action)

    EP_LENGTH = 20
    env = PS(catalog.DiscreteMockEnv(reward_step=0, end_game=EP_LENGTH))
    env.reset()
    step = None

    for i in range(10):
        step = env.random_step()
        assert step.reward.item() == -1

    for i in range(10):
        step = env.random_step()
        assert step.reward.item() == 0


def test_randomize_actions_full_randomization_replaces_with_valid_actions():

    class RecordingEnv(catalog.DiscreteMockEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.last_action = None

        def step(self, action):
            self.last_action = np.array(action, copy=True)
            return super().step(action)

    wrapped = RecordingEnv(4, n_actions=5)
    env = ActionRandomizer(wrapped, p=1.0)
    env.reset()

    action = np.zeros(env.n_agents, dtype=np.int32)
    env.step(action)

    assert wrapped.last_action is not None
    assert np.all(wrapped.last_action >= 0)
    assert np.all(wrapped.last_action < env.n_actions)


def test_randomize_actions_per_agent_probability():

    class RecordingEnv(catalog.DiscreteMockEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.last_action = None

        def step(self, action):
            self.last_action = np.array(action, copy=True)
            return super().step(action)

    wrapped = RecordingEnv(3, n_actions=4, end_game=100)
    env = ActionRandomizer(wrapped, p=[0.0, 1.0, 0.0])
    env.reset()

    action = np.array([1, 1, 1], dtype=np.int32)
    changed_second_agent = False
    for _ in range(100):
        env.step(action)
        assert wrapped.last_action is not None
        used = wrapped.last_action
        assert used[0] == action[0]
        assert used[2] == action[2]
        if used[1] != action[1]:
            changed_second_agent = True
            break

    assert changed_second_agent, "Agent with probability 1.0 should eventually get a different sampled action"


def test_randomize_actions_invalid_probability_length_raises():
    with pytest.raises(AssertionError):
        ActionRandomizer(catalog.DiscreteMockEnv(3), p=[0.1, 0.2])


def test_randomize_actions_invalid_probability_value_raises():
    with pytest.raises(AssertionError):
        ActionRandomizer(catalog.DiscreteMockEnv(2), p=[0.2, 1.2])


def test_randomized_actions_from_builder():
    Builder(catalog.DiscreteMockEnv(3)).randomize_actions(p=[0.0, 1.0, 0.0]).build()
    with pytest.raises(AssertionError):
        Builder(catalog.DiscreteMockEnv(3)).randomize_actions(p=[0.1, 0.2]).build()
    with pytest.raises(AssertionError):
        Builder(catalog.DiscreteMockEnv(2)).randomize_actions(p=[0.2, 1.2]).build()


def test_env_pool():
    envs = [
        catalog.DiscreteMockEnv(n_agents=2, n_actions=2),
        catalog.DiscreteMockEnv(n_agents=2, n_actions=2),
    ]
    env_pool = EnvPool(envs)
    found = [False, False]
    n_trials = 0
    while n_trials < 1000 and not all(found):
        n_trials += 1
        env_pool.reset()
        for i, env in enumerate(envs):
            if env_pool.wrapped == env:
                found[i] = True
    assert found[0] and found[1]


def test_incompatible_envs():
    with pytest.raises(AssertionError):
        EnvPool([catalog.DiscreteMockEnv(n_agents=2, n_actions=2), catalog.DiscreteMockEnv(n_agents=2, n_actions=3)])
    with pytest.raises(AssertionError):
        EnvPool([catalog.DiscreteMockEnv(n_agents=2, n_actions=2), catalog.DiscreteMockEnv(n_agents=3, n_actions=2)])
    with pytest.raises(AssertionError):
        EnvPool(
            [
                catalog.DiscreteMockEnv(n_agents=2, n_actions=2, extras_size=10),
                catalog.DiscreteMockEnv(n_agents=2, n_actions=2, extras_size=1),
            ]
        )
