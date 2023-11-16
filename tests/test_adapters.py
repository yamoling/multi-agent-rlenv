import rlenv


def test_gym_adapter():
    # Discrete action space
    env = rlenv.make("CartPole-v1")
    env.reset()

    # Continuous action space
    env = rlenv.make("Pendulum-v1")
    env.reset()


def test_smac_adapter():
    from rlenv.adapters import SMACAdapter
    from rlenv.models import DiscreteActionSpace

    env = SMACAdapter("3m")
    assert env.n_agents == 3
    assert isinstance(DiscreteActionSpace, env.action_space)
