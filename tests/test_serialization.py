import pickle
import rlenv
import json
from dataclasses import asdict
from serde.json import to_json

from rlenv import MockEnv


def test_registry():
    env = MockEnv(4)
    serialized = pickle.dumps(env)
    restored_env = pickle.loads(serialized)
    assert restored_env.n_agents == env.n_agents
    assert restored_env.observation_shape == env.observation_shape
    assert restored_env.state_shape == env.state_shape
    assert restored_env.extra_feature_shape == env.extra_feature_shape
    assert restored_env.n_actions == env.n_actions


def test_registry_gym():
    env = rlenv.make("CartPole-v1")
    restored_env = pickle.loads(pickle.dumps(env))
    assert restored_env.n_agents == env.n_agents
    assert restored_env.observation_shape == env.observation_shape
    assert restored_env.state_shape == env.state_shape
    assert restored_env.extra_feature_shape == env.extra_feature_shape
    assert restored_env.n_actions == env.n_actions


def test_registry_wrapper():
    env = rlenv.Builder(MockEnv(4)).agent_id().time_limit(10).build()
    restored_env = pickle.loads(pickle.dumps(env))
    assert restored_env.n_agents == env.n_agents
    assert restored_env.observation_shape == env.observation_shape
    assert restored_env.state_shape == env.state_shape
    assert restored_env.extra_feature_shape == env.extra_feature_shape
    assert restored_env.n_actions == env.n_actions


def test_action_space_json():
    space = rlenv.DiscreteActionSpace(1, 4)
    json.dumps(asdict(space))
    to_json(space)
    space = rlenv.ContinuousActionSpace(1, 4)
    json.dumps(asdict(space))
    to_json(space)


def test_env_serialization_json():
    env = MockEnv(4)
    to_json(env)
    json.dumps(asdict(env))


def test_gym_adapter_json():
    env = rlenv.make("CartPole-v1")
    data = to_json(env)
    assert "observation_shape" in data
    assert "state_shape" in data
    assert "extra_feature_shape" in data
    assert "n_actions" in data
