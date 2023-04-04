#! /usr/bin/env python3
import rlenv
from rlenv.models import EpisodeBuilder

if __name__ == "__main__":
    env = (rlenv.Builder("CartPole-v0")
           .agent_id()
           .last_action()
           .time_limit(100)
           .force_actions({0: 3})
           .build()
           )
    done = False
    obs = env.reset()
    builder = EpisodeBuilder()
    while not done:
        actions = policy.choose_action(obs)
        obs_, reward, done, info = env.step(actions)
        t = rlenv.Transition(obs, actions, reward, done, info, obs_)
        builder.add(t)
    episode = builder.build()



