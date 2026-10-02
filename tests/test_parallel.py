import numpy as np
import pytest

from marlenv import Builder, ParallelMARLEnv
from marlenv.catalog import DiscreteMockEnv, DiscreteMOMockEnv, EnvPool


def make_env(end_games=(1, 3, 5), **kwargs):
    return ParallelMARLEnv([DiscreteMockEnv(end_game=n, **kwargs) for n in end_games])


def test_reset_shapes():
    env = make_env(n_agents=2, obs_size=7, n_actions=4, extras_size=3)
    obs, state = env.reset()
    assert obs.data.shape == (3, 2, 7)
    assert obs.extras.shape == (3, 2, 3)
    assert obs.available_actions.shape == (3, 2, 4)
    assert state.data.shape == (3, *env.state_shape)
    assert env.mask.all()
    assert obs[1].data.shape == (2, 7)
    assert env.sample_action().shape == (3, 2)


def test_invalid_construction():
    with pytest.raises(ValueError):
        ParallelMARLEnv([])
    env = DiscreteMockEnv()
    with pytest.raises(ValueError):
        ParallelMARLEnv([env, env])
    with pytest.raises(ValueError):
        ParallelMARLEnv([DiscreteMockEnv(n_actions=3), DiscreteMockEnv(n_actions=4)])


def test_step_before_reset_and_wrong_number_of_actions():
    env = make_env()
    with pytest.raises(RuntimeError):
        env.step(np.zeros((3, 4), dtype=np.int64))
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.zeros((2, 4), dtype=np.int64))


def test_same_results_as_independent_envs():
    end_games = (2, 4)
    parallel = make_env(end_games)
    independents = [DiscreteMockEnv(end_game=n) for n in end_games]
    obs, _ = parallel.reset(seed=10)
    for i, env in enumerate(independents):
        ind_obs, _ = env.reset(seed=10 + i)
        assert ind_obs == obs[i]
    for _ in range(4):
        actions = parallel.sample_action()
        step = parallel.step(actions)
        for i, env in enumerate(independents):
            if not step.mask[i]:
                continue
            ind_step = env.step(actions[i])
            assert ind_step == step.steps[i]
            assert ind_step.obs == step.obs[i]
            assert ind_step.state == step.state[i]


def test_mask_and_sticky_flags():
    env = make_env((1, 3))
    env.reset()
    actions = np.zeros((2, 4), dtype=np.int64)

    step = env.step(actions)
    assert step.mask.tolist() == [True, True]
    assert step.done.tolist() == [True, False]
    assert not step.is_terminal
    assert env.mask.tolist() == [False, True]
    frozen_obs = step.obs.data[0].copy()

    step = env.step(actions)
    assert step.mask.tolist() == [False, True]
    assert step.steps[0] is None and step.steps[1] is not None
    assert step.infos[0] == {}
    assert step.reward[0] == 0 and step.reward[1] == 1
    assert step.done.tolist() == [True, False]
    assert np.array_equal(step.obs.data[0], frozen_obs)
    assert len(env.envs[0].actions_history) == 1  # pyright: ignore[reportAttributeAccessIssue]

    step = env.step(actions)
    assert step.done.tolist() == [True, True]
    assert step.is_terminal
    with pytest.raises(RuntimeError):
        env.step(actions)

    env.reset()
    assert env.mask.all()


def test_truncation():
    envs = [Builder(DiscreteMockEnv(end_game=10)).time_limit(n, add_extra=False).build() for n in (1, 2)]
    env = ParallelMARLEnv(envs)
    env.reset()
    step = env.step(env.sample_action())
    assert step.truncated.tolist() == [True, False]
    assert step.done.tolist() == [False, False]
    assert not step.is_terminal
    step = env.step(env.sample_action())
    assert step.truncated.tolist() == [True, True]
    assert step.is_terminal


def test_multi_objective_rewards():
    env = ParallelMARLEnv([DiscreteMOMockEnv(n_objectives=3, end_game=n) for n in (1, 2)])
    env.reset()
    env.step(env.sample_action())
    step = env.step(env.sample_action())
    assert step.reward.shape == (2, 3)
    assert np.all(step.reward[0] == 0) and np.all(step.reward[1] == 1)


def test_env_pool_to_parallel():
    envs = [DiscreteMockEnv(end_game=n) for n in (1, 3)]
    parallel = EnvPool(envs).to_parallel()
    assert parallel.envs == envs
    episodes = parallel.rollout(lambda obs: np.zeros((len(obs), obs.n_agents), dtype=np.int64))
    assert [len(e) for e in episodes] == [1, 3]


def test_rollout():
    env = make_env((1, 3, 5))
    episodes = env.rollout(lambda obs: np.zeros((len(obs), obs.n_agents), dtype=np.int64))
    assert [len(e) for e in episodes] == [1, 3, 5]
