import pickle
import marlenv
from dataclasses import asdict

from marlenv import DiscreteMockEnv


def test_registry():
    env = DiscreteMockEnv(4)
    serialized = pickle.dumps(env)
    restored_env = pickle.loads(serialized)
    assert restored_env.n_agents == env.n_agents
    assert restored_env.observation_shape == env.observation_shape
    assert restored_env.state_shape == env.state_shape
    assert restored_env.extra_shape == env.extra_shape
    assert restored_env.n_actions == env.n_actions


def test_registry_gym():
    env = marlenv.make("CartPole-v1")
    restored_env = pickle.loads(pickle.dumps(env))
    assert restored_env.n_agents == env.n_agents
    assert restored_env.observation_shape == env.observation_shape
    assert restored_env.state_shape == env.state_shape
    assert restored_env.extra_shape == env.extra_shape
    assert restored_env.n_actions == env.n_actions


def test_registry_wrapper():
    env = marlenv.Builder(DiscreteMockEnv(4)).agent_id().time_limit(10).build()
    restored_env = pickle.loads(pickle.dumps(env))
    assert restored_env.n_agents == env.n_agents
    assert restored_env.observation_shape == env.observation_shape
    assert restored_env.state_shape == env.state_shape
    assert restored_env.extra_shape == env.extra_shape
    assert restored_env.n_actions == env.n_actions
