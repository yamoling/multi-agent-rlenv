import rlenv


def test_gym_adapter():
    # Discrete action space
    env = rlenv.make("CartPole-v1")
    env.reset()

    # Continuous action space
    env = rlenv.make("Pendulum-v1")
    env.reset()
