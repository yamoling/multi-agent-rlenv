from marlenv import Observation, Transition, MockEnv, MOMockEnv
import numpy as np


def test_obs_eq():
    obs1 = Observation(
        data=np.arange(20, dtype=np.float32),
        available_actions=np.ones(5, dtype=bool),
        state=np.ones(10, dtype=np.float32),
        extras=np.arange(5, dtype=np.float32),
    )

    obs2 = Observation(
        data=np.arange(20, dtype=np.float32),
        available_actions=np.full((5,), True),
        state=np.ones(10, dtype=np.float32),
        extras=np.arange(5, dtype=np.float32),
    )

    assert obs1 == obs2


def test_obs_not_eq():
    obs1 = Observation(
        data=np.arange(20, dtype=np.float32),
        available_actions=np.full((5,), True),
        state=np.ones(10, dtype=np.float32),
        extras=np.arange(5, dtype=np.float32),
    )

    obs2 = Observation(
        data=np.arange(20, dtype=np.float32) + 1,
        available_actions=np.full((5,), True),
        state=np.ones(10, dtype=np.float32),
        extras=np.arange(5, dtype=np.float32),
    )

    assert not obs1 == obs2


def test_obs_eq_extras_none():
    obs1 = Observation(
        data=np.arange(20, dtype=np.float32),
        available_actions=np.full((5,), True),
        state=np.ones(10, dtype=np.float32),
        extras=None,
    )

    obs2 = Observation(
        data=np.arange(20, dtype=np.float32),
        available_actions=np.full((5,), True),
        state=np.ones(10, dtype=np.float32),
        extras=None,
    )

    assert obs1 == obs2


def test_obs_hash():
    obs1 = Observation(
        data=np.arange(20, dtype=np.float32),
        available_actions=np.full((5,), True),
        state=np.ones(10, dtype=np.float32),
        extras=np.arange(5, dtype=np.float32),
    )

    obs2 = Observation(
        data=np.arange(20, dtype=np.float32),
        available_actions=np.full((5,), True),
        state=np.ones(10, dtype=np.float32),
        extras=np.arange(5, dtype=np.float32),
    )

    assert hash(obs1) == hash(obs2)


def test_transition_eq():
    t1 = Transition(
        obs=Observation(
            data=np.arange(20, dtype=np.float32),
            available_actions=np.full((5,), True),
            state=np.ones(10, dtype=np.float32),
            extras=np.arange(5, dtype=np.float32),
        ),
        action=np.ones(5, dtype=np.int64),
        reward=[1.0],
        done=False,
        info={},
        obs_=Observation(
            data=np.arange(20, dtype=np.float32),
            available_actions=np.full((5,), True),
            state=np.ones(10, dtype=np.float32),
            extras=np.arange(5, dtype=np.float32),
        ),
        truncated=False,
    )

    t2 = Transition(
        obs=Observation(
            data=np.arange(20, dtype=np.float32),
            available_actions=np.full((5,), True),
            state=np.ones(10, dtype=np.float32),
            extras=np.arange(5, dtype=np.float32),
        ),
        action=np.ones(5, dtype=np.int64),
        reward=[1.0],
        done=False,
        info={},
        obs_=Observation(
            data=np.arange(20, dtype=np.float32),
            available_actions=np.full((5,), True),
            state=np.ones(10, dtype=np.float32),
            extras=np.arange(5, dtype=np.float32),
        ),
        truncated=False,
    )

    assert t1 == t2


def test_transition_hash():
    t1 = Transition(
        obs=Observation(
            data=np.arange(20, dtype=np.float32),
            available_actions=np.full((5,), True),
            state=np.ones(10, dtype=np.float32),
            extras=np.arange(5, dtype=np.float32),
        ),
        action=np.ones(5, dtype=np.int64),
        reward=[1.0],
        done=False,
        info={},
        obs_=Observation(
            data=np.arange(20, dtype=np.float32),
            available_actions=np.full((5,), True),
            state=np.ones(10, dtype=np.float32),
            extras=np.arange(5, dtype=np.float32),
        ),
        truncated=False,
    )

    t2 = Transition(
        obs=Observation(
            data=np.arange(20, dtype=np.float32),
            available_actions=np.full((5,), True),
            state=np.ones(10, dtype=np.float32),
            extras=np.arange(5, dtype=np.float32),
        ),
        action=np.ones(5, dtype=np.int64),
        reward=[1.0],
        done=False,
        info={},
        obs_=Observation(
            data=np.arange(20, dtype=np.float32),
            available_actions=np.full((5,), True),
            state=np.ones(10, dtype=np.float32),
            extras=np.arange(5, dtype=np.float32),
        ),
        truncated=False,
    )

    t3 = Transition(
        obs=Observation(
            data=np.arange(20, dtype=np.float32),
            available_actions=np.full((5,), True),
            state=np.ones(10, dtype=np.float32),
            extras=np.arange(5, dtype=np.float32),
        ),
        action=np.ones(5, dtype=np.int64),
        reward=[1.0],
        done=False,
        info={},
        obs_=Observation(
            data=np.arange(20, dtype=np.float32),
            available_actions=np.full((5,), True),
            state=np.ones(10, dtype=np.float32),
            extras=np.arange(5, dtype=np.float32),
        ),
        truncated=True,
    )

    assert hash(t1) == hash(t2)
    assert hash(t1) != hash(t3)


def test_has_same_inouts():
    env = MockEnv(4)
    env2 = MockEnv(3)
    assert env.has_same_inouts(env)
    assert not env.has_same_inouts(env2)

    env = MOMockEnv(n_objectives=1)
    env2 = MOMockEnv(n_objectives=2)
    assert not env.has_same_inouts(env2)

    env = MockEnv(n_actions=5)
    env2 = MockEnv(n_actions=6)
    assert not env.has_same_inouts(env2)

    env = MockEnv(agent_state_size=1)
    env2 = MockEnv(agent_state_size=2)
    assert not env.has_same_inouts(env2)

    env = MockEnv(obs_size=5)
    env2 = MockEnv(obs_size=6)
    assert not env.has_same_inouts(env2)

    env = MockEnv(extras_size=2)
    env2 = MockEnv(extras_size=4)
    assert not env.has_same_inouts(env2)


def test_rlenv_available_actions():
    env = MockEnv(4)
    assert np.all(env.available_actions() == 1)


def test_multi_objective_env():
    N_AGENTS = 2
    N_OBJECTVES = 3
    env = MOMockEnv(N_AGENTS, N_OBJECTVES)
    assert env.reward_space.size == N_OBJECTVES
    assert env.n_agents == N_AGENTS
    assert env.n_actions == env.n_actions

    env.reset()
    reward = env.step([0] * N_AGENTS)[1]
    assert len(reward) == N_OBJECTVES


def test_is_multi_objective():
    env = MockEnv(4)
    assert not env.is_multi_objective

    env = MOMockEnv(4)
    assert env.is_multi_objective
