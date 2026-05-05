from marlenv.catalog import DiscreteMockEnv


def test_episode_lengths():
    # Check rollout produces episode of the requested length (end_game)
    for length in (1, 5, 20):
        env = DiscreteMockEnv(n_agents=2, end_game=length)
        episode = env.rollout(lambda _: env.sample_action())
        assert len(episode) == length


def test_observations_shapes_and_sizes():
    n_agents = 3
    obs_size = 7
    n_actions = 4
    extras_size = 2

    env = DiscreteMockEnv(n_agents=n_agents, obs_size=obs_size, n_actions=n_actions, extras_size=extras_size)
    obs, state = env.reset()

    # Observation data shape: [n_agents, *observation_shape]
    assert obs.data.shape == (n_agents, obs_size)
    assert obs.shape == env.observation_shape
    assert env.observation_shape == (obs_size,)

    # Observation extras shape: [n_agents, *extras_shape]
    assert obs.extras.shape == (n_agents, extras_size)
    assert obs.extras_shape == env.extras_shape
    assert env.extras_shape == (extras_size,)
    assert obs.extras_size == env.extras_size
    assert env.extras_size == extras_size

    # Available actions shape
    assert obs.available_actions.shape == (n_agents, n_actions)
    assert env.n_actions == n_actions

    # Calling as_tensors (if torch is available) should keep consistent shapes;
    # we don't require torch here, but check the Observation convenience properties
    assert obs.n_agents == n_agents


def test_observation_extras_default_zero():
    # When extras_size is not provided, extras should be empty per-agent
    n_agents = 4
    env = DiscreteMockEnv(n_agents=n_agents)
    obs, _ = env.reset()

    assert obs.extras.shape == (n_agents, 0)
    assert obs.extras_shape == env.extras_shape
    assert env.extras_size == 0
    assert obs.extras_size == 0


def test_state_shapes_and_sizes():
    n_agents = 5
    agent_state_size = 2
    env = DiscreteMockEnv(n_agents=n_agents, agent_state_size=agent_state_size)
    obs, state = env.reset()

    # State data shape: flattened size = n_agents * agent_state_size
    expected_state_shape = (n_agents * agent_state_size,)
    assert state.shape == expected_state_shape
    assert env.state_shape == expected_state_shape
    assert state.data.shape == expected_state_shape

    # By default DiscreteMockEnv does not set state extras -> empty
    assert state.extras.shape == (0,)
    assert state.extras_shape == env.state_extra_shape
    # extras_size on State is the product of extras_shape (0 => 0)
    assert state.extras_size == 0
    # MARLEnv convenience method for state extras size
    assert env.state_extras_size == 0


def test_state_and_observation_consistency_multiple_resets():
    # Ensure repeated resets preserve declared shapes
    env = DiscreteMockEnv(n_agents=3, obs_size=6, agent_state_size=1, extras_size=1, n_actions=3)
    for seed in (None, 0, 42):
        obs, state = env.reset(seed=seed)
        assert obs.data.shape == (env.n_agents, env.observation_shape[0])
        assert obs.extras.shape == (env.n_agents, env.extras_shape[0])
        assert state.data.shape == env.state_shape
        assert state.extras.shape == env.state_extra_shape or state.extras.shape == (0,)


def test_state_extra_shape_size():
    env = DiscreteMockEnv()
    assert env.state_extra_shape == (0,)
    assert env.state_extras_size == 0
    obs, state = env.reset()
    assert obs.extras_size == 0
    assert state.extras_size == 0
    assert state.extras_shape == (0,)
