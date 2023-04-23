import rlenv
from rlenv import wrappers
from .mock_env import MockEnv


def test_registry():
    env = MockEnv(4)
    summary = env.summary()
    rlenv.register(MockEnv)
    restored_env = rlenv.from_summary(summary)
    assert restored_env.n_agents == env.n_agents
    assert restored_env.observation_shape == env.observation_shape
    assert restored_env.state_shape == env.state_shape
    assert restored_env.extra_feature_shape == env.extra_feature_shape
    assert restored_env.n_actions == env.n_actions
    assert env.summary() == restored_env.summary()


def test_registry_gym():
    env = rlenv.make("CartPole-v1")
    summary = env.summary()
    restored_env = rlenv.from_summary(summary)
    assert restored_env.n_agents == env.n_agents
    assert restored_env.observation_shape == env.observation_shape
    assert restored_env.state_shape == env.state_shape
    assert restored_env.extra_feature_shape == env.extra_feature_shape
    assert restored_env.n_actions == env.n_actions
    assert env.summary() == restored_env.summary()

def test_unregistered_environment():
    from rlenv.exceptions import UnknownEnvironmentException
    env = MockEnv(4)
    rlenv.registry.ENV_REGISTRY.pop(MockEnv.__name__)
    summary = env.summary()
    try:
        rlenv.from_summary(summary)
        assert False
    except UnknownEnvironmentException:
        assert True


def test_registry_wrapper():
    env = rlenv.Builder(MockEnv(4)).agent_id().time_limit(10).build()
    rlenv.register(MockEnv)
    summary = env.summary()
    restored_env = rlenv.from_summary(summary)
    assert restored_env.n_agents == env.n_agents
    assert restored_env.observation_shape == env.observation_shape
    assert restored_env.state_shape == env.state_shape
    assert restored_env.extra_feature_shape == env.extra_feature_shape
    assert restored_env.n_actions == env.n_actions
    assert env.summary() == restored_env.summary()

def test_env_to_summary():
    env = MockEnv(4)
    summary = env.summary()
    assert summary["name"] == "MockEnv"
    assert summary["n_agents"] == 4
    assert summary["obs_shape"] == (MockEnv.OBS_SIZE, )
    assert summary["state_shape"] == (0, )
    assert summary["extras_shape"] == (0, )
    assert summary["n_actions"] == MockEnv.N_ACTIONS
    assert summary[MockEnv.__name__] == { "n_agents": 4 }

def test_env_from_summary():
    env = MockEnv(4)
    summary = env.summary()
    env2 = MockEnv.from_summary(summary)
    assert env2.n_agents == 4
    assert env2.observation_shape == (MockEnv.OBS_SIZE, )
    assert env2.state_shape == (0, )
    assert env2.extra_feature_shape == (0, )
    assert env2.n_actions == MockEnv.N_ACTIONS


def test_wrapper_to_summary():
    env = MockEnv(4)
    env = wrappers.AgentIdWrapper(env)
    env = wrappers.PadExtras(env, n_added=5)
    summary = env.summary()
    assert summary["name"] == "MockEnv"
    assert summary["n_agents"] == env.n_agents
    assert summary["obs_shape"] == (MockEnv.OBS_SIZE, )
    assert summary["state_shape"] == (0, )
    assert summary["extras_shape"] == (env.n_agents + 5, )
    assert summary["n_actions"] == MockEnv.N_ACTIONS
    assert summary["wrappers"] == [wrappers.AgentIdWrapper.__name__, wrappers.PadExtras.__name__]
    assert summary[MockEnv.__name__] == { "n_agents": 4 }
    assert summary[wrappers.AgentIdWrapper.__name__] == {}
    assert summary[wrappers.PadExtras.__name__] == { "n_added": 5 }

def test_wrapper_from_summary():
    env = MockEnv(4)
    env = wrappers.AgentIdWrapper(env)
    env = wrappers.PadExtras(env, n_added=5)
    summary = env.summary()
    env = MockEnv.from_summary(summary)
    env = wrappers.from_summary(env, summary)

    assert env.n_agents == 4
    assert env.observation_shape == (MockEnv.OBS_SIZE, )
    assert env.state_shape == (0, )
    assert env.extra_feature_shape == (env.n_agents + 5, )
    assert env.n_actions == MockEnv.N_ACTIONS
    assert env.name == "MockEnv"