import rlenv
from pettingzoo.sisl import pursuit_v4, waterworld_v4
from rlenv.adapters import SMAC


def test_gym_adapter():
    # Discrete action space
    env = rlenv.make("CartPole-v1")
    env.reset()

    # Continuous action space
    env = rlenv.make("Pendulum-v1")
    env.reset()


def test_pettingzoo_adapter_discrete_action():
    # https://pettingzoo.farama.org/environments/sisl/pursuit/#pursuit
    env = rlenv.make(pursuit_v4.parallel_env())
    env.reset()
    env.step(env.action_space.sample())
    assert env.n_agents == 8
    assert env.n_actions == 5
    assert isinstance(env.action_space, rlenv.DiscreteActionSpace)


def test_pettingzoo_adapter_continuous_action():
    # https://pettingzoo.farama.org/environments/sisl/waterworld/
    env = rlenv.make(waterworld_v4.parallel_env())
    env.reset()
    env.step(env.action_space.sample())
    assert env.n_actions == 2
    assert env.n_agents == 2
    assert isinstance(env.action_space, rlenv.ContinuousActionSpace)


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
