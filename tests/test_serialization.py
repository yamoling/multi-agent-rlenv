import pickle

import numpy as np
import orjson
import pytest

import marlenv
from marlenv import DiscreteMockEnv, wrappers
from marlenv.utils import Schedule


def test_registry():
    env = DiscreteMockEnv(4)
    serialized = pickle.dumps(env)
    restored_env = pickle.loads(serialized)
    assert restored_env.n_agents == env.n_agents
    assert restored_env.observation_shape == env.observation_shape
    assert restored_env.state_shape == env.state_shape
    assert restored_env.extras_shape == env.extras_shape
    assert restored_env.n_actions == env.n_actions


@pytest.mark.skipif(not marlenv.adapters.HAS_GYM, reason="Gymnasium is not installed")
def test_registry_gym():
    env = marlenv.make("CartPole-v1")
    restored_env = pickle.loads(pickle.dumps(env))
    assert restored_env.n_agents == env.n_agents
    assert restored_env.observation_shape == env.observation_shape
    assert restored_env.state_shape == env.state_shape
    assert restored_env.extras_shape == env.extras_shape
    assert restored_env.n_actions == env.n_actions


def test_registry_wrapper():
    env = marlenv.Builder(DiscreteMockEnv(4)).agent_id().time_limit(10).build()
    restored_env = pickle.loads(pickle.dumps(env))
    assert restored_env.n_agents == env.n_agents
    assert restored_env.observation_shape == env.observation_shape
    assert restored_env.state_shape == env.state_shape
    assert restored_env.extras_shape == env.extras_shape
    assert restored_env.n_actions == env.n_actions


def test_env_json_serialization():
    env = DiscreteMockEnv(4)
    _ = orjson.dumps(env, option=orjson.OPT_SERIALIZE_NUMPY)


def test_serialize_episode_fields():
    env = DiscreteMockEnv(4, end_game=10)
    obs, state = env.reset()
    episode = marlenv.Episode.new(obs, state)
    action = np.array([0, 1, 2, 3])
    for _ in range(10):
        step = env.step(action)
        transition = marlenv.Transition.from_step(obs, state, action, step)
        episode.add(transition)

    serialized = orjson.dumps(episode, option=orjson.OPT_SERIALIZE_NUMPY)
    episode = orjson.loads(serialized)
    fields = [
        "all_observations",
        "all_extras",
        "actions",
        "rewards",
        "all_available_actions",
        "all_states",
        "all_states_extras",
        "metrics",
        "episode_len",
        "other",
        "is_done",
        "is_truncated",
    ]
    for field in fields:
        assert field in episode


def serde_and_check_key_values(env: object):
    serialized = orjson.dumps(env, option=orjson.OPT_SERIALIZE_NUMPY)
    deserialized = orjson.loads(serialized)
    checked_keys = []
    for key, value in env.__dict__.items():
        if key.startswith("_"):
            continue
        checked_keys.append(key)
        assert key in deserialized
        match value:
            case int() | float() | str() | bool() | list() | dict():
                assert deserialized[key] == value
            case np.ndarray():
                assert np.all(deserialized[key] == value)
    assert len(checked_keys) > 0


def test_serialize_blind():
    env = DiscreteMockEnv(4)
    serde_and_check_key_values(wrappers.Blind(env, 0.2))


def test_serialize_time_limit():
    env = DiscreteMockEnv(4)
    serde_and_check_key_values(wrappers.TimeLimit(env, 10))


def test_serialize_time_penalty():
    env = DiscreteMockEnv(4)
    serde_and_check_key_values(wrappers.TimePenalty(env, 0.2))


def test_serialize_agent_id():
    env = DiscreteMockEnv(4)
    serde_and_check_key_values(wrappers.AgentId(env))


def test_serialize_last_action():
    env = DiscreteMockEnv(4)
    serde_and_check_key_values(wrappers.LastAction(env))


def test_serialize_available_actions():
    env = DiscreteMockEnv(4)
    serde_and_check_key_values(wrappers.AvailableActions(env))


def test_serialize_video():
    env = DiscreteMockEnv(4)
    serde_and_check_key_values(wrappers.VideoRecorder(env))


def test_serialize_centralised():
    env = DiscreteMockEnv(4)
    serde_and_check_key_values(wrappers.Centralized(env))


def test_serialize_pad_extras():
    env = DiscreteMockEnv(4)
    serde_and_check_key_values(wrappers.PadExtras(env, 5))


def test_serialize_pad_observation():
    env = DiscreteMockEnv(4)
    serde_and_check_key_values(wrappers.PadObservations(env, 5))


def test_wrappers_serializable():
    env = DiscreteMockEnv(4)
    env = marlenv.Builder(env).agent_id().available_actions().time_limit(10).last_action().time_penalty(5).blind(0.2).build()

    serde_and_check_key_values(env)


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


@pytest.mark.skipif(not marlenv.adapters.HAS_GYM, reason="Gymnasium is not installed")
def test_json_serialize_gym():
    env = marlenv.make("CartPole-v1")
    serde_and_check_key_values(env)


@pytest.mark.skipif(not marlenv.adapters.HAS_PETTINGZOO, reason="PettingZoo is not installed")
def test_json_serialize_pettingzoo():
    from pettingzoo.sisl import pursuit_v4

    env = marlenv.adapters.PettingZoo(pursuit_v4.parallel_env())
    serde_and_check_key_values(env)


@pytest.mark.skipif(not marlenv.adapters.HAS_SMAC, reason="SMAC is not installed")
def test_json_serialize_smac():
    env = marlenv.adapters.SMAC("3m")
    serde_and_check_key_values(env)


class C:
    def __call__(self, t):
        return t + 1


def test_serialize_schedule():
    s = Schedule.linear(0, 1, 10)
    orjson.dumps(s)
    b = pickle.dumps(s)
    s2 = pickle.loads(b)
    assert s == s2

    s = Schedule.exp(1, 16, 5)
    orjson.dumps(s)
    b = pickle.dumps(s)
    s2 = pickle.loads(b)
    assert s == s2

    s = Schedule.constant(50)
    orjson.dumps(s)
    b = pickle.dumps(s)
    s2 = pickle.loads(b)
    assert s == s2

    s = Schedule.arbitrary(lambda t: t + 1)
    b = orjson.dumps(s)
    try:
        pickle.dumps(s)
        assert False, "Should not be able to pickle arbitrary schedules because of the callable lambda"
    except AttributeError:
        pass

    s = Schedule.arbitrary(C())
    orjson.dumps(s)
    b = pickle.dumps(s)
    s2 = pickle.loads(b)
    assert s == s2
