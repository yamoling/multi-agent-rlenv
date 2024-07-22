import numpy as np
from rlenv.models import EpisodeBuilder, Transition, Episode, RLEnv
from rlenv import wrappers, MockEnv


def generate_episode(env: RLEnv, with_probs: bool = False) -> Episode:
    obs = env.reset()
    episode = EpisodeBuilder()
    while not episode.is_finished:
        action = env.action_space.sample()
        probs = None
        if with_probs:
            probs = np.random.random(action.shape)
        next_obs, r, done, truncated, info = env.step(action)
        episode.add(Transition(obs, action, r, done, info, next_obs, truncated, probs))
        obs = next_obs
    return episode.build()


def test_episode_with_logprobs():
    env = wrappers.TimeLimit(MockEnv(2), 10)
    episode = generate_episode(env, with_probs=True)
    assert episode.actions_probs is not None
    assert len(episode.actions_probs) == 10


test_episode_with_logprobs()
