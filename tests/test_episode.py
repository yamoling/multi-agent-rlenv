import random
from rlenv.models import EpisodeBuilder, Transition
from .mock_env import MockEnv

def test_returns():
    env = MockEnv(2)
    obs = env.reset()
    builder = EpisodeBuilder()
    n_steps = 20
    gamma = 0.95
    rewards = []
    for i in range(n_steps):
        done = i == n_steps - 1
        r = random.random()
        rewards.append(r)
        builder.add(Transition(obs, [0, 0], r, done, {}, obs))
    episode = builder.build()
    returns = episode.compute_returns(discount=gamma)
    for i, r in enumerate(returns):
        G_t = rewards[-1]
        for j in range(len(rewards) - 2, i - 1, -1):
            G_t = rewards[j] + gamma * G_t
        assert abs(r - G_t) < 1e-6

