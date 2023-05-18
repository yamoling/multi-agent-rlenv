import rlenv


def test_gym_adapter():
    # Discrete action space
    env = rlenv.make("CartPole-v1")
    env.reset()

    # Continuous action space
    env = rlenv.make("Pendulum-v1")
    env.reset()


def test_smac_summary():
    env = rlenv.make("smac:3m")
    summary = env.summary()
    env2 = rlenv.from_summary(summary)
    assert env.name == env2.name
    assert env.n_agents == env2.n_agents
    assert env.n_actions == env2.n_actions
    assert env.observation_shape == env2.observation_shape
    assert env.state_shape == env2.state_shape
