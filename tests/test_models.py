from copy import deepcopy
from importlib.util import find_spec

import numpy as np
import pytest

from marlenv import Builder, DiscreteSpace, Episode, MARLEnv, Observation, State, Transition
from marlenv.catalog import DiscreteMockEnv, DiscreteMOMockEnv

from .utils import generate_episode

HAS_PYTORCH = find_spec("torch") is not None


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


def test_transition_set_arbitrary_keys():
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
    )

    t["arbitrary_key"] = 17
    t["other_key"] = [1, 2, 3]

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

        def step(self, action):
            return super().step(action)

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


def test_env_extras_meanings():
    env = DiscreteMockEnv(4, extras_size=4)
    assert len(env.extras_meanings) == 4


def test_env_extras_size_from_1d_shape():
    env = DiscreteMockEnv(4, extras_size=4)
    assert env.extras_size == 4


def test_env_extras_size_is_zero_by_default():
    env = DiscreteMockEnv(4)
    assert env.extras_size == 0


def test_env_extras_size_from_multidim_shape():
    class TestClass(MARLEnv):
        def __init__(self):
            super().__init__(4, DiscreteSpace(5), (10,), (10,), extras_shape=(2, 3))

        def get_observation(self):
            raise NotImplementedError()

        def get_state(self):
            raise NotImplementedError()

        def step(self, action):
            raise NotImplementedError()

        def reset(self, *, seed: int | None = None):
            raise NotImplementedError()

    env = TestClass()
    assert env.extras_size == 6


def test_env_observation_size_from_1d_shape():
    env = DiscreteMockEnv(4, obs_size=7)
    assert env.observation_size == 7


def test_env_observation_size_from_multidim_shape():
    class TestClass(MARLEnv):
        def __init__(self):
            super().__init__(4, DiscreteSpace(5), (2, 3, 4), (10,))

        def get_observation(self):
            raise NotImplementedError()

        def get_state(self):
            raise NotImplementedError()

        def step(self, action):
            raise NotImplementedError()

        def reset(self, *, seed: int | None = None):
            raise NotImplementedError()

    env = TestClass()
    assert env.observation_size == 24


def test_env_observation_size_default_mock_shape():
    env = DiscreteMockEnv(4)
    assert env.observation_size == env.observation_shape[0]


def test_wrong_extras_meanings_length():
    class TestClass(MARLEnv):
        def __init__(self):
            super().__init__(4, DiscreteSpace(5), (10,), (10,), extras_shape=(5,), extras_meanings=["a", "b", "c"])

        def get_observation(self):
            raise NotImplementedError()

        def get_state(self):
            raise NotImplementedError()

        def step(self, action):
            raise NotImplementedError()

        def reset(self, *, seed: int | None = None):
            raise NotImplementedError()

    try:
        TestClass()
        assert False, "This should raise a ValueError because the length of extras_meanings is different from the actual number of extras"
    except ValueError:
        pass


def test_env_rollout():
    EP_LENGTH = 50
    env = DiscreteMockEnv(end_game=EP_LENGTH)
    episode = env.rollout(lambda x: env.sample_action())
    assert len(episode) == EP_LENGTH

    env = DiscreteMOMockEnv(end_game=EP_LENGTH)
    episode = env.rollout(lambda x: env.sample_action())
    assert len(episode) == EP_LENGTH


@pytest.mark.skipif(not HAS_PYTORCH, reason="torch is not installed")
def test_observation_as_tensor():
    import torch

    env = DiscreteMockEnv(4)
    obs = env.reset()[0]
    data, extras = obs.as_tensors()
    assert isinstance(data, torch.Tensor)
    assert data.shape == (env.n_agents, *env.observation_shape)
    assert data.dtype == torch.float32
    assert isinstance(extras, torch.Tensor)
    assert extras.shape == (env.n_agents, *env.extras_shape)
    assert extras.dtype == torch.float32


@pytest.mark.skipif(not HAS_PYTORCH, reason="torch is not installed")
def test_observation_as_tensor_with_batch():
    import torch

    env = DiscreteMockEnv(4)
    obs = env.reset()[0]
    data, extras = obs.as_tensors(batch_dim=True)
    assert isinstance(data, torch.Tensor)
    assert data.shape == (1, env.n_agents, *env.observation_shape)
    assert data.dtype == torch.float32
    assert isinstance(extras, torch.Tensor)
    assert extras.shape == (1, env.n_agents, *env.extras_shape)
    assert extras.dtype == torch.float32


@pytest.mark.skipif(not HAS_PYTORCH, reason="torch is not installed")
def test_observation_as_tensor_with_batch_with_available_actions():
    import torch

    env = DiscreteMockEnv(4)
    obs = env.reset()[0]
    data, extras, available_actions = obs.as_tensors(batch_dim=True, actions=True)
    assert isinstance(data, torch.Tensor)
    assert data.shape == (1, env.n_agents, *env.observation_shape)
    assert data.dtype == torch.float32
    assert isinstance(extras, torch.Tensor)
    assert extras.shape == (1, env.n_agents, *env.extras_shape)
    assert extras.dtype == torch.float32
    assert isinstance(available_actions, torch.Tensor)
    assert available_actions.shape == (1, env.n_agents, env.n_actions)
    assert available_actions.dtype == torch.bool


@pytest.mark.skipif(not HAS_PYTORCH, reason="torch is not installed")
def test_state_as_tensor():
    import torch

    env = DiscreteMockEnv(4)
    state = env.reset()[1]
    data, extras = state.as_tensors()
    assert isinstance(data, torch.Tensor)
    assert data.shape == env.state_shape
    assert data.dtype == torch.float32
    assert isinstance(extras, torch.Tensor)
    assert extras.shape == env.state_extra_shape
    assert extras.dtype == torch.float32


@pytest.mark.skipif(not HAS_PYTORCH, reason="torch is not installed")
def test_state_as_tensor_with_batch():
    import torch

    env = DiscreteMockEnv(4)
    state = env.reset()[1]
    data, extras = state.as_tensors(batch_dim=True)
    assert isinstance(data, torch.Tensor)
    assert data.shape == (1, *env.state_shape)
    assert data.dtype == torch.float32
    assert isinstance(extras, torch.Tensor)
    assert extras.shape == (1, *env.state_extra_shape)
    assert extras.dtype == torch.float32


def test_as_joint_1d_obs():
    """With 4 agents and obs shape (20,), the joint obs shape should be (80,)."""
    n_agents = 4
    obs_size = 20
    n_actions = 5
    n_extras = 3

    obs = Observation(
        data=np.ones((n_agents, obs_size), dtype=np.float32),
        available_actions=np.ones((n_agents, n_actions), dtype=bool),
        extras=np.zeros((n_agents, n_extras), dtype=np.float32),
    )

    joint = obs.as_joint()

    # The underlying array has a leading agent dimension of 1
    assert joint.data.shape == (1, n_agents * obs_size)
    assert joint.available_actions.shape == (1, n_agents * n_actions)
    assert joint.extras.shape == (1, n_agents * n_extras)
    # As seen by the single joint agent, the obs shape is (80,)
    assert joint.shape == (n_agents * obs_size,)
    assert joint.n_agents == 1


def test_as_joint_image_obs():
    """With 2 agents each receiving an RGB (3, 80, 80) image, the joint obs shape should be (6, 80, 80)."""
    n_agents = 2
    C, H, W = 3, 80, 80
    n_actions = 4
    n_extras = 2

    obs = Observation(
        data=np.ones((n_agents, C, H, W), dtype=np.float32),
        available_actions=np.ones((n_agents, n_actions), dtype=bool),
        extras=np.zeros((n_agents, n_extras), dtype=np.float32),
    )

    joint = obs.as_joint()

    # The underlying array has a leading agent dimension of 1
    assert joint.data.shape == (1, n_agents * C, H, W)
    assert joint.available_actions.shape == (1, n_agents * n_actions)
    assert joint.extras.shape == (1, n_agents * n_extras)
    # As seen by the single joint agent, the obs shape is (6, 80, 80)
    assert joint.shape == (n_agents * C, H, W)
    assert joint.n_agents == 1


def test_as_joint_preserves_data_values():
    """Values in the joint observation should be the concatenation of per-agent data."""
    obs = Observation(
        data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        available_actions=np.array([[True, False], [False, True]], dtype=bool),
        extras=np.array([[10.0], [20.0]], dtype=np.float32),
    )
    joint = obs.as_joint()
    assert np.array_equal(joint.data[0], np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    assert np.array_equal(joint.available_actions[0], np.array([True, False, False, True], dtype=bool))
    assert np.array_equal(joint.extras[0], np.array([10.0, 20.0], dtype=np.float32))


def test_original_obs_is_unchanged():
    """Calling as_joint should not modify the original observation."""
    obs = Observation(
        data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        available_actions=np.array([[True, False], [False, True]], dtype=bool),
        extras=np.array([[10.0], [20.0]], dtype=np.float32),
    )
    obs_copy = deepcopy(obs)
    _ = obs.as_joint()
    assert np.array_equal(obs.data, obs_copy.data)
    assert np.array_equal(obs.available_actions, obs_copy.available_actions)
    assert np.array_equal(obs.extras, obs_copy.extras)


def test_as_joint_no_extras():
    """as_joint works correctly when no extras are provided (extras default to empty)."""
    n_agents = 3
    obs_size = 10
    n_actions = 5

    obs = Observation(
        data=np.arange(n_agents * obs_size, dtype=np.float32).reshape(n_agents, obs_size),
        available_actions=np.ones((n_agents, n_actions), dtype=bool),
    )

    joint = obs.as_joint()

    assert joint.shape == (n_agents * obs_size,)
    assert joint.extras_shape == (0,)
    assert joint.n_agents == 1


def test_step_equality():
    env1 = DiscreteMockEnv(n_agents=2)
    env2 = DiscreteMockEnv(n_agents=3)
    env3 = DiscreteMockEnv(n_agents=2)

    env1.reset()
    env2.reset()
    env3.reset()

    a1 = env1.sample_action()
    s1 = env1.step(a1)
    s3 = env3.step(a1)
    assert s1 == s3
    s2 = env2.random_step()
    assert s1 != s2
    assert s1 == s1

    s4 = env1.step(a1)
    assert s1 != s4
