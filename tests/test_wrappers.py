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

