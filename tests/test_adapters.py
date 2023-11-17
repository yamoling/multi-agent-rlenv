import rlenv


def test_gym_adapter():
    # Discrete action space
    env = rlenv.make("CartPole-v1")
    env.reset()

    # Continuous action space
    env = rlenv.make("Pendulum-v1")
    env.reset()


def test_smac_adapter():
    # Do not test this if starcrat is not installed (e.g. on CI)
    try:
        from rlenv.adapters import SMAC
        from rlenv.models import DiscreteActionSpace

        env = SMAC("3m")
        env.reset()
        assert env.n_agents == 3
        assert isinstance(env.action_space, DiscreteActionSpace)
    except (FileNotFoundError, ImportError):
        # File not found error if smac is installed but starcraft is not
        pass
