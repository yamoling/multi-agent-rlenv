import pytest
from marlenv.env_pool import EnvPool
from marlenv import DiscreteMockEnv


def test_env_pool():
    envs = [
        DiscreteMockEnv(n_agents=2, n_actions=2),
        DiscreteMockEnv(n_agents=2, n_actions=2),
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
        EnvPool([DiscreteMockEnv(n_agents=2, n_actions=2), DiscreteMockEnv(n_agents=2, n_actions=3)])
    with pytest.raises(AssertionError):
        EnvPool([DiscreteMockEnv(n_agents=2, n_actions=2), DiscreteMockEnv(n_agents=3, n_actions=2)])
    with pytest.raises(AssertionError):
        EnvPool([DiscreteMockEnv(n_agents=2, n_actions=2, extras_size=10), DiscreteMockEnv(n_agents=2, n_actions=2, extras_size=1)])
