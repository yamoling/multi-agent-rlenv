import pickle
import numpy as np
import orjson
import pytest
import os
from copy import deepcopy

import marlenv
from marlenv import DiscreteMockEnv


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


def test_wrappers_serializable():
    env = DiscreteMockEnv(4)
    env = marlenv.Builder(env).agent_id().available_actions().time_limit(10).last_action().time_penalty(5).blind(0.2).build()
    as_bytes = orjson.dumps(env, option=orjson.OPT_SERIALIZE_NUMPY)
    deserialized = orjson.loads(as_bytes)

    def check_key_values(env: object, deserialized: dict):
        for key, value in env.__dict__.items():
            if key.startswith("_"):
                continue
            assert key in deserialized
            if key == "wrapped":
                check_key_values(value, deserialized[key])

    check_key_values(env, deserialized)


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


@pytest.mark.skipif(not marlenv.adapters.HAS_OVERCOOKED, reason="Overcooked is not installed")
def test_deepcopy_overcooked():
    env = marlenv.adapters.Overcooked.from_layout("scenario4")
    env2 = deepcopy(env)
    assert env == env2


@pytest.mark.skipif(not marlenv.adapters.HAS_OVERCOOKED, reason="Overcooked is not installed")
def test_pickle_overcooked():
    env = marlenv.adapters.Overcooked.from_layout("scenario1_s", horizon=60)
    serialized = pickle.dumps(env)
    restored = pickle.loads(serialized)
    assert env == restored

    env.reset()
    restored.reset()

    for _ in range(50):
        actions = env.sample_action()
        step = env.step(actions)
        step_restored = restored.step(actions)
        assert step == step_restored


@pytest.mark.skipif(not marlenv.adapters.HAS_OVERCOOKED, reason="Overcooked is not installed")
def test_unpickling_from_blank_process():
    from marlenv.adapters import Overcooked
    import pickle
    import subprocess
    import tempfile

    env = Overcooked.from_layout("large_room")
    env_file = tempfile.NamedTemporaryFile("wb", delete=False)
    pickle.dump(env, env_file)
    env_file.close()

    # Write the python file

    f = tempfile.NamedTemporaryFile("w", delete=False)
    f.write("""
import pickle
import sys

with open(sys.argv[1], "rb") as f:
    env = pickle.load(f)

env.reset()""")
    f.close()
    try:
        output = subprocess.run(f"python {f.name} {env_file.name}", shell=True, capture_output=True)
        assert output.returncode == 0, output.stderr.decode("utf-8")
    finally:
        os.remove(f.name)
        os.remove(env_file.name)
