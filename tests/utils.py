from marlenv import MARLEnv, Episode, Transition
import numpy as np
from typing import Any, Optional


def generate_episode(env: MARLEnv[Any, Any], with_probs: bool = False, seed: Optional[int] = None) -> Episode:
    if seed is not None:
        env.seed(seed)
    obs, state = env.reset()
    episode = Episode.new(obs, state)
    while not episode.is_finished:
        action = env.sample_action()
        probs = None
        if with_probs:
            probs = np.random.random(action.shape)
        step = env.step(action)
        episode.add(Transition.from_step(obs, state, action, step, action_probs=probs))
        obs = step.obs
        state = step.state
    return episode
