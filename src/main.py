#! /usr/bin/env python3
import rlenv

print(rlenv.__version__)
import rlenv
# From Gym
env = rlenv.make("CartPole-v1")

# From pettingzoo
from pettingzoo.sisl import pursuit_v4
env = rlenv.make(pursuit_v4.parallel_env())


# Building the environment with additional information
from pettingzoo.sisl import pursuit_v4
env = rlenv.Builder(pursuit_v4.parallel_env())\
    .with_agent_id()\
    .with_last_action()\
    .build()

assert env.extra_feature_shape == (13, )
