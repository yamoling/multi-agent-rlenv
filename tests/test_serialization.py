import pickle
import marlenv
import numpy as np
import orjson

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


def test_env_json_serialization():
    env = DiscreteMockEnv(4)
    _ = orjson.dumps(env, option=orjson.OPT_SERIALIZE_NUMPY)


def test_wrappers_serializable():
    env = DiscreteMockEnv(4)
    env = marlenv.Builder(env).agent_id().available_actions().time_limit(10).last_action().time_penalty(5).blind(0.2).build()
    _ = orjson.dumps(env, option=orjson.OPT_SERIALIZE_NUMPY)


def test_serialize_observation():
    env = DiscreteMockEnv(4)
    obs = env.get_observation()
    _ = orjson.dumps(obs, option=orjson.OPT_SERIALIZE_NUMPY)


def test_serialize_state():
    env = DiscreteMockEnv(4)
    state = env.get_state()
    _ = orjson.dumps(state, option=orjson.OPT_SERIALIZE_NUMPY)


def test_serialize_step():
    env = DiscreteMockEnv(4)
    obs, state = env.reset()
    action = np.array([0, 1, 2, 3])
    step = env.step(action)
    _ = orjson.dumps(step, option=orjson.OPT_SERIALIZE_NUMPY)


def test_serialize_transition():
    env = DiscreteMockEnv(4)
    obs, state = env.reset()
    action = np.array([0, 1, 2, 3])
    step = env.step(action)
    transition = marlenv.Transition.from_step(obs, state, action, step)
    _ = orjson.dumps(transition, option=orjson.OPT_SERIALIZE_NUMPY)


def test_serialize_episode():
    env = DiscreteMockEnv(4, end_game=10)
    obs, state = env.reset()
    episode = marlenv.Episode.new(obs, state)
    action = np.array([0, 1, 2, 3])
    for _ in range(10):
        step = env.step(action)
        transition = marlenv.Transition.from_step(obs, state, action, step)
        episode.add(transition)

    _ = orjson.dumps(episode, option=orjson.OPT_SERIALIZE_NUMPY)
