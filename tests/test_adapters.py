import rlenv


def test_gym_adapter():
    # Discrete action space
    env = rlenv.make("CartPole-v1")
    env.reset()

    # Continuous action space
    env = rlenv.make("Pendulum-v1")
    env.reset()


from rlenv.adapters import SMAC  # noqa: E402

# Only perform the tests if SMAC is installed.
if SMAC is not None:

    def test_smac_adapter():
        from rlenv.models import DiscreteActionSpace

        env = SMAC("3m")
        env.reset()
        assert env.n_agents == 3
        assert isinstance(env.action_space, DiscreteActionSpace)

    def test_smac_render():
        env = SMAC("3m")
        env.reset()
        env.render("human")

    def test_smac_pickle():
        import pickle

        env = SMAC("3m")
        env.reset()
        pickle.dumps(env)
