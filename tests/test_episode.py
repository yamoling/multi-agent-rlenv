import random
import pytest
from rlenv.models import EpisodeBuilder, Transition
from .mock_env import MockEnv

def test_returns():
    env = MockEnv(2)
    obs = env.reset()
    builder = EpisodeBuilder()
    n_steps = 20
    rewards = []
    for i in range(n_steps):
        done = i == n_steps - 1
        r = random.random()
        rewards.append(r)
        builder.add(Transition(obs, [0, 0], r, done, {}, obs))
    episode = builder.build()
    returns = episode.compute_returns(discount=1)
    for i, r in enumerate(returns):
        assert pytest.approx(sum(rewards[i:]), 1e-5) == r

