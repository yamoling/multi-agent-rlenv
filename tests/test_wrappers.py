import numpy as np
from rlenv import Builder
import rlenv
from .mock_env import MockEnv


def test_padding():
    PAD_SIZE = 2
    env = Builder(MockEnv(5)).pad("extra", PAD_SIZE).build()
    assert env.extra_feature_shape == (PAD_SIZE,)

    env = Builder(MockEnv(5)).pad("obs", PAD_SIZE).build()
    assert env.observation_shape == (MockEnv.OBS_SIZE + PAD_SIZE,)


def test_available_actions():
    N_AGENTS = 5
    env = Builder(MockEnv(N_AGENTS)).available_actions().build()

    assert env.extra_feature_shape == (5,)
    obs = env.reset()
    assert np.array_equal(obs.extras, np.ones((N_AGENTS, MockEnv.N_ACTIONS), dtype=np.float32))


def test_agent_id():
    env = Builder(MockEnv(5)).agent_id().build()

    assert env.extra_feature_shape == (5,)
    obs = env.reset()
    assert np.array_equal(obs.extras, np.identity(5, dtype=np.float32))


def test_penalty_wrapper():
    env = Builder(MockEnv(1)).time_penalty(0.1).build()
    done = False
    while not done:
        _, reward, done, *_ = env.step([0])
        assert reward == MockEnv.REWARD_STEP - 0.1


def test_time_limit_wrapper():
    MAX_T = 5
    env = Builder(MockEnv(1)).time_limit(MAX_T).build()
    assert env.extra_feature_shape == (0,)
    stop = False
    t = 0
    while not stop:
        _, _, done, truncated, _ = env.step([0])
        stop = done or truncated
        t += 1
    assert t == MAX_T


def test_time_limit_wrapper_with_extra():
    MAX_T = 5
    env = Builder(MockEnv(5)).time_limit(MAX_T, add_extra=True).build()
    assert env.extra_feature_shape == (1,)
    obs = env.reset()
    assert obs.extras.shape == (5, 1)
    stop = False
    t = 0
    while not stop:
        obs, _, done, truncated, _ = env.step([0])
        stop = done or truncated
        t += 1
    assert t == MAX_T
    assert np.all(obs.extras[:] == 1)


def test_force_actions():
    forced_actions = {0: 1, 3: 4}
    env = Builder(MockEnv(5)).force_actions(forced_actions).build()
    obs = env.reset()
    done = False
    while not done:
        for agent, forced_action in forced_actions.items():
            for action in range(env.n_actions):
                if action == forced_action:
                    assert obs.available_actions[agent, action] == 1
                else:
                    assert obs.available_actions[agent, action] == 0
        obs, _, done, *_ = env.step([0, 1, 2, 3, 4])


def test_restore_custom_wrapper():
    import rlenv
    from rlenv import wrappers, RLEnv

    class MyWrapper(wrappers.RLEnvWrapper):
        def __init__(self, env: RLEnv, **kwargs):
            super().__init__(env)
            self._kwargs = kwargs

        def kwargs(self):
            return self._kwargs

    rlenv.register(MockEnv)
    rlenv.register_wrapper(MyWrapper)
    env = MyWrapper(MockEnv(1), a=1, b=2)
    s = env.summary()
    restored = rlenv.from_summary(s)
    assert isinstance(restored, MyWrapper)
    assert restored.kwargs() == {"a": 1, "b": 2}
    assert restored.n_agents == 1


def test_blind_wrapper():
    def test(env: rlenv.RLEnv):
        obs = env.reset()
        assert np.any(obs.data != 0)
        obs, r, done, truncated, info = env.step(env.action_space.sample())
        assert np.all(obs.data == 0)

    env = rlenv.Builder(MockEnv(5)).blind(p=1).build()
    test(env)
    env = rlenv.wrappers.BlindWrapper(MockEnv(5), p=1)
    test(env)
