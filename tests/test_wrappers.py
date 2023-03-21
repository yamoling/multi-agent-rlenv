import numpy as np
from rlenv import Builder
from .mock_env import MockEnv


def test_padding():
    env = (Builder(MockEnv(5))
           .pad("extra", 2)
           .build())
    assert env.extra_feature_shape == (2, )

    env = (Builder(MockEnv(5))
           .pad("obs", 2)
           .build()) 
    assert env.observation_shape == (12, )   


def test_agent_id():
    env = (Builder(MockEnv(5))
           .agent_id()
           .build())

    assert env.extra_feature_shape == (5, )
    obs = env.reset()
    assert np.array_equal(obs.extras, np.identity(5, dtype=np.float32))

def test_penalty_wrapper():
    env = Builder(MockEnv(1)).penalty(0.1).build()
    done = False
    while not done:
       _, reward, done, _ = env.step([0])
       assert reward == MockEnv.REWARD_STEP - 0.1


def test_time_limit_wrapper():
    MAX_T = 5
    env = Builder(MockEnv(1)).time_limit(MAX_T).build()
    done = False
    t = 0
    while not done:
        _, _, done, _ = env.step([0])
        t += 1
    assert t == MAX_T