from rlenv import Observation, Transition
from .mock_env import MockEnv
import numpy as np


def test_obs_eq():
    obs1 = Observation(
        data=np.arange(20),
        available_actions=np.ones(5),
        state=np.ones(10),
        extras=np.arange(5),
    )

    obs2 = Observation(
        data=np.arange(20),
        available_actions=np.ones(5),
        state=np.ones(10),
        extras=np.arange(5),
    )

    assert obs1 == obs2


def test_obs_eq_extras_none():
    obs1 = Observation(
        data=np.arange(20),
        available_actions=np.ones(5),
        state=np.ones(10),
        extras=None,
    )

    obs2 = Observation(
        data=np.arange(20),
        available_actions=np.ones(5),
        state=np.ones(10),
        extras=None,
    )

    assert obs1 == obs2


def test_obs_hash():
    obs1 = Observation(
        data=np.arange(20),
        available_actions=np.ones(5),
        state=np.ones(10),
        extras=np.arange(5),
    )

    obs2 = Observation(
        data=np.arange(20),
        available_actions=np.ones(5),
        state=np.ones(10),
        extras=np.arange(5),
    )

    assert hash(obs1) == hash(obs2)


def test_transition_eq():
    t1 = Transition(
        obs=Observation(
            data=np.arange(20),
            available_actions=np.ones(5),
            state=np.ones(10),
            extras=np.arange(5),
        ),
        action=np.ones(5),
        reward=1.0,
        done=False,
        info={},
        obs_=Observation(
            data=np.arange(20),
            available_actions=np.ones(5),
            state=np.ones(10),
            extras=np.arange(5),
        ),
        truncated=False,
    )

    t2 = Transition(
        obs=Observation(
            data=np.arange(20),
            available_actions=np.ones(5),
            state=np.ones(10),
            extras=np.arange(5),
        ),
        action=np.ones(5),
        reward=1.0,
        done=False,
        info={},
        obs_=Observation(
            data=np.arange(20),
            available_actions=np.ones(5),
            state=np.ones(10),
            extras=np.arange(5),
        ),
        truncated=False,
    )

    assert t1 == t2


def test_transition_hash():
    t1 = Transition(
        obs=Observation(
            data=np.arange(20),
            available_actions=np.ones(5),
            state=np.ones(10),
            extras=np.arange(5),
        ),
        action=np.ones(5),
        reward=1.0,
        done=False,
        info={},
        obs_=Observation(
            data=np.arange(20),
            available_actions=np.ones(5),
            state=np.ones(10),
            extras=np.arange(5),
        ),
        truncated=False,
    )

    t2 = Transition(
        obs=Observation(
            data=np.arange(20),
            available_actions=np.ones(5),
            state=np.ones(10),
            extras=np.arange(5),
        ),
        action=np.ones(5),
        reward=1.0,
        done=False,
        info={},
        obs_=Observation(
            data=np.arange(20),
            available_actions=np.ones(5),
            state=np.ones(10),
            extras=np.arange(5),
        ),
        truncated=False,
    )

    assert hash(t1) == hash(t2)


def test_rlenv_available_actions():
    env = MockEnv(4)
    assert np.all(env.available_actions() == 1)
