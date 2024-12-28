from marlenv import Observation, Transition, DiscreteMockEnv, DiscreteMOMockEnv, Builder, State, Episode
import numpy as np
from .utils import generate_episode


def test_obs_eq():
    obs1 = Observation(
        data=np.arange(20, dtype=np.float32),
        available_actions=np.ones(5, dtype=bool),
        extras=np.arange(5, dtype=np.float32),
    )

    obs2 = Observation(
        data=np.arange(20, dtype=np.float32),
        available_actions=np.full((5,), True),
        extras=np.arange(5, dtype=np.float32),
    )

    assert obs1 == obs2


def test_obs_not_eq():
    obs1 = Observation(
        data=np.arange(20, dtype=np.float32),
        available_actions=np.full((5,), True),
        extras=np.arange(5, dtype=np.float32),
    )

    obs2 = Observation(
        data=np.arange(20, dtype=np.float32) + 1,
        available_actions=np.full((5,), True),
        extras=np.arange(5, dtype=np.float32),
    )

    assert not obs1 == obs2


def test_obs_eq_extras_none():
    obs1 = Observation(
        data=np.arange(20, dtype=np.float32),
        available_actions=np.full((5,), True),
        extras=None,
    )

    obs2 = Observation(
        data=np.arange(20, dtype=np.float32),
        available_actions=np.full((5,), True),
        extras=None,
    )

    assert obs1 == obs2


def test_obs_hash():
    obs1 = Observation(
        data=np.arange(20, dtype=np.float32),
        available_actions=np.full((5,), True),
        extras=np.arange(5, dtype=np.float32),
    )

    obs2 = Observation(
        data=np.arange(20, dtype=np.float32),
        available_actions=np.full((5,), True),
        extras=np.arange(5, dtype=np.float32),
    )

    assert hash(obs1) == hash(obs2)


def test_transition_eq():
    t1 = Transition(
        obs=Observation(
            data=np.arange(20, dtype=np.float32),
            available_actions=np.full((5,), True),
            extras=np.arange(5, dtype=np.float32),
        ),
        state=State(np.ones(10, dtype=np.float32)),
        action=np.ones(5, dtype=np.int64),
        reward=1.0,
        done=False,
        info={},
        next_obs=Observation(
            data=np.arange(20, dtype=np.float32),
            available_actions=np.full((5,), True),
            extras=np.arange(5, dtype=np.float32),
        ),
        next_state=State(np.ones(10, dtype=np.float32)),
        truncated=False,
    )

    t2 = Transition(
        obs=Observation(
            data=np.arange(20, dtype=np.float32),
            available_actions=np.full((5,), True),
            extras=np.arange(5, dtype=np.float32),
        ),
        state=State(np.ones(10, dtype=np.float32)),
        action=np.ones(5, dtype=np.int64),
        reward=1.0,
        done=False,
        info={},
        next_obs=Observation(
            data=np.arange(20, dtype=np.float32),
            available_actions=np.full((5,), True),
            extras=np.arange(5, dtype=np.float32),
        ),
        next_state=State(np.ones(10, dtype=np.float32)),
        truncated=False,
    )

    assert t1 == t2


def test_transition_hash():
    t1 = Transition(
        obs=Observation(
            data=np.arange(20, dtype=np.float32),
            available_actions=np.full((5,), True),
            extras=np.arange(5, dtype=np.float32),
        ),
        state=State(np.ones(10, dtype=np.float32)),
        action=np.ones(5, dtype=np.int64),
        reward=1.0,
        done=False,
        info={},
        next_obs=Observation(
            data=np.arange(20, dtype=np.float32),
            available_actions=np.full((5,), True),
            extras=np.arange(5, dtype=np.float32),
        ),
        next_state=State(np.ones(10, dtype=np.float32)),
        truncated=False,
    )

    t2 = Transition(
        obs=Observation(
            data=np.arange(20, dtype=np.float32),
            available_actions=np.full((5,), True),
            extras=np.arange(5, dtype=np.float32),
        ),
        state=State(np.ones(10, dtype=np.float32)),
        action=np.ones(5, dtype=np.int64),
        reward=1.0,
        done=False,
        info={},
        next_obs=Observation(
            data=np.arange(20, dtype=np.float32),
            available_actions=np.full((5,), True),
            extras=np.arange(5, dtype=np.float32),
        ),
        next_state=State(np.ones(10, dtype=np.float32)),
        truncated=False,
    )

    t3 = Transition(
        obs=Observation(
            data=np.arange(20, dtype=np.float32),
            available_actions=np.full((5,), True),
            extras=np.arange(5, dtype=np.float32),
        ),
        state=State(np.ones(10, dtype=np.float32)),
        action=np.ones(5, dtype=np.int64),
        reward=1.0,
        done=False,
        info={},
        next_obs=Observation(
            data=np.arange(20, dtype=np.float32),
            available_actions=np.full((5,), True),
            extras=np.arange(5, dtype=np.float32),
        ),
        next_state=State(np.ones(10, dtype=np.float32) + 1),
        truncated=True,
    )

    assert hash(t1) == hash(t2)
    assert hash(t1) != hash(t3)


def test_transition_arbitrary_keys():
    t = Transition(
        obs=Observation(
            data=np.arange(20, dtype=np.float32),
            available_actions=np.full((5,), True),
            extras=np.arange(5, dtype=np.float32),
        ),
        state=State(np.ones(10, dtype=np.float32)),
        action=np.ones(5, dtype=np.int64),
        reward=1.0,
        done=False,
        info={},
        next_obs=Observation(
            data=np.arange(20, dtype=np.float32),
            available_actions=np.full((5,), True),
            extras=np.arange(5, dtype=np.float32),
        ),
        next_state=State(np.ones(10, dtype=np.float32)),
        truncated=False,
        arbitrary_key=17,
        other_key=[1, 2, 3],
    )

    assert t["arbitrary_key"] == 17
    assert t["other_key"] == [1, 2, 3]


def test_episode_arbitrary_keys():
    episode = Episode.new(
        Observation(np.ones(10, dtype=np.float32), np.full(5, True)),
        State(np.ones(10, dtype=np.float32)),
    )
    episode.add(
        Transition(
            obs=Observation(
                data=np.arange(20, dtype=np.float32),
                available_actions=np.full((5,), True),
                extras=np.arange(5, dtype=np.float32),
            ),
            state=State(np.ones(10, dtype=np.float32)),
            action=np.ones(5, dtype=np.int64),
            reward=1.0,
            done=False,
            info={},
            next_obs=Observation(
                data=np.arange(20, dtype=np.float32),
                available_actions=np.full((5,), True),
                extras=np.arange(5, dtype=np.float32),
            ),
            next_state=State(np.ones(10, dtype=np.float32)),
            truncated=False,
            arbitrary_key=17,
            other_key=[1, 2, 3],
        )
    )
    episode.add(
        Transition(
            obs=Observation(
                data=np.arange(20, dtype=np.float32),
                available_actions=np.full((5,), True),
                extras=np.arange(5, dtype=np.float32),
            ),
            state=State(np.ones(10, dtype=np.float32)),
            action=np.ones(5, dtype=np.int64),
            reward=1.0,
            done=True,
            info={},
            next_obs=Observation(
                data=np.arange(20, dtype=np.float32),
                available_actions=np.full((5,), True),
                extras=np.arange(5, dtype=np.float32),
            ),
            next_state=State(np.ones(10, dtype=np.float32)),
            truncated=False,
            arbitrary_key=18,
            other_key=[2, 3, 4],
        )
    )
    assert np.array_equal(episode["arbitrary_key"], [17, 18])
    assert np.array_equal(episode["other_key"], [[1, 2, 3], [2, 3, 4]])


def test_has_same_inouts():
    env = DiscreteMockEnv(4)
    env2 = DiscreteMockEnv(3)
    assert env.has_same_inouts(env)
    assert not env.has_same_inouts(env2)

    env = DiscreteMOMockEnv(n_objectives=1)
    env2 = DiscreteMOMockEnv(n_objectives=2)
    assert not env.has_same_inouts(env2)

    env = DiscreteMockEnv(n_actions=5)
    env2 = DiscreteMockEnv(n_actions=6)
    assert not env.has_same_inouts(env2)

    env = DiscreteMockEnv(agent_state_size=1)
    env2 = DiscreteMockEnv(agent_state_size=2)
    assert not env.has_same_inouts(env2)

    env = DiscreteMockEnv(obs_size=5)
    env2 = DiscreteMockEnv(obs_size=6)
    assert not env.has_same_inouts(env2)

    env = DiscreteMockEnv(extras_size=2)
    env2 = DiscreteMockEnv(extras_size=4)
    assert not env.has_same_inouts(env2)


def test_rlenv_available_actions():
    env = DiscreteMockEnv(4)
    assert np.all(env.available_actions() == 1)


def test_available_joint_actions():
    env = DiscreteMockEnv(n_agents=2, n_actions=3)
    res = env.available_joint_actions()
    assert len(res) == env.n_actions**env.n_agents

    possibilities = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    for joint_action in res:
        assert len(joint_action) == env.n_agents
        assert joint_action in possibilities


def test_available_joint_actions_masked():
    env = Builder(DiscreteMockEnv(n_agents=2, n_actions=3)).mask_actions(np.array([[0, 1, 1], [1, 0, 1]])).build()
    res = env.available_joint_actions()

    possibilities = [(1, 0), (1, 2), (2, 0), (2, 2)]
    for joint_action in res:
        assert len(joint_action) == env.n_agents
        assert joint_action in possibilities


def test_multi_objective_env():
    N_AGENTS = 2
    N_OBJECTVES = 3
    env = DiscreteMOMockEnv(N_AGENTS, N_OBJECTVES)
    assert env.reward_space.size == N_OBJECTVES
    assert env.n_agents == N_AGENTS
    assert env.n_actions == env.n_actions

    env.reset()
    reward = env.step([0] * N_AGENTS).reward
    assert len(reward) == N_OBJECTVES


def test_is_multi_objective():
    env = DiscreteMockEnv(4)
    assert not env.is_multi_objective

    env = DiscreteMOMockEnv(4)
    assert env.is_multi_objective


def test_state_hash():
    s1 = State(np.ones(10, dtype=np.float32))
    s2 = State(np.ones(10, dtype=np.float32))
    s3 = State(np.ones(10, dtype=np.float32) + 1)
    s4 = State(np.ones(10, dtype=np.float32), np.ones(10, dtype=np.float32))

    assert hash(s1) == hash(s2)
    assert hash(s1) != hash(s3)
    assert hash(s1) != hash(s4)


def test_state_eq():
    s1 = State(np.ones(10, dtype=np.float32))
    s2 = State(np.ones(10, dtype=np.float32))
    s3 = State(np.ones(10, dtype=np.float32) + 1)
    s4 = State(np.ones(10, dtype=np.float32), np.ones(10, dtype=np.float32))

    assert s1 == s2
    assert s1 != s3
    assert s1 != s4


def test_unpack_step():
    env = DiscreteMockEnv(4)
    obs, state = env.reset()
    action = np.array([0, 1, 2, 3])
    step = env.step(action)
    obs, state, reward, done, truncated, info = step
    assert isinstance(obs, Observation)
    assert isinstance(state, State)
    assert isinstance(reward, (int, np.ndarray, float))
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_env_replay():
    class PseudoRandomEnv(DiscreteMockEnv):
        def available_actions(self):
            availables = np.full((self.n_agents, self.n_actions), False, dtype=bool)
            for agent, available in enumerate(availables):
                available[(agent + self._seed) % self.n_actions] = True
            return availables

        def step(self, actions):
            return super().step(actions)

        def seed(self, seed_value: int):
            np.random.seed(seed_value)
            self._seed = seed_value

    env = PseudoRandomEnv()
    env.seed(0)
    episode = generate_episode(env)
    actions = episode.actions
    episode2 = env.replay(actions, seed=0)
    assert np.array_equal(episode.all_observations, episode2.all_observations)
    assert np.array_equal(episode.all_states, episode2.all_states)
    assert np.array_equal(episode.rewards, episode2.rewards)
    assert np.array_equal(episode.all_available_actions, episode2.all_available_actions)
