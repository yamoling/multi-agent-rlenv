#! /usr/bin/env python3
import rlenv

print(rlenv.__version__)
import rlenv
# From Gym

env = (rlenv.Builder("CartPole-v1")
       .agent_id()
       .pad("extra", 2)
       .build())


