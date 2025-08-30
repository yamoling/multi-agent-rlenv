from marlenv.catalog.deepsea import DeepSea, LEFT, RIGHT


def test_env():
    env = DeepSea(20)
    assert env.n_actions == 2
    assert env.action_space.is_discrete
    assert env.observation_shape == (2,)
    assert env.n_agents == 1


def test_reset():
    env = DeepSea(20)
    obs, state = env.reset()
    assert obs.shape == (2,)
    assert state.shape == (2,)

    assert obs.data[0][0] == 0
    assert state.data[0] == 0
    assert obs.data[0][1] == 0
    assert state.data[1] == 0


def test_step():
    env = DeepSea(20)
    env.reset()
    step = env.step([RIGHT])
    obs = step.obs
    assert obs.data[0][0] == 1
    assert obs.data[0][1] == 1
    assert step.reward.item() < 0

    step = env.step([LEFT])
    obs = step.obs
    assert obs.data[0][0] == 2
    assert obs.data[0][1] == 0
    assert step.reward.item() == 0.0

    step = env.step([LEFT])
    obs = step.obs
    assert obs.data[0][0] == 3
    assert obs.data[0][1] == 0
    assert step.reward.item() == 0.0
