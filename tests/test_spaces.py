import numpy as np
from rlenv.models import DiscreteActionSpace, ContinuousActionSpace


def test_discrete_action_space():
    s = DiscreteActionSpace(2, 3)
    available_actions = np.array([[1, 1, 0], [1, 0, 1]])
    for _ in range(100):
        actions = s.sample(available_actions)
        assert actions.shape == (2,)
        assert actions[0] in [0, 1]
        assert actions[1] in [0, 2]

        actions = s.sample()
        assert actions.shape == (2,)
        assert actions[0] in [0, 1, 2]
        assert actions[1] in [0, 1, 2]


def test_continuous_action_space():
    s = ContinuousActionSpace(2, 3, low=[0.0, -1.0, 0.0], high=[1.0, 1.0, 2.0])
    for _ in range(100):
        actions = s.sample()
        assert actions.shape == (2, 3)
        assert np.all(actions[:, 0] >= 0) and np.all(actions[:, 0] < 1)
        assert np.all(actions[:, 1] >= -1) and np.all(actions[:, 1] < 1)
        assert np.all(actions[:, 2] >= 0) and np.all(actions[:, 2] < 2)


def test_action_names():
    s = DiscreteActionSpace(2, 3, action_names=["a", "b", "c"])
    assert s.action_names == ["a", "b", "c"]
    s = ContinuousActionSpace(2, 3, action_names=["a", "b", "c"])
    assert s.action_names == ["a", "b", "c"]
    s = DiscreteActionSpace(2, 3)
    assert s.action_names == ["Action 0", "Action 1", "Action 2"]
    s = ContinuousActionSpace(2, 3)
    assert s.action_names == ["Action 0", "Action 1", "Action 2"]


def test_action_names_wrong_number_of_actions():
    try:
        DiscreteActionSpace(2, 5, action_names=["a", "b", "c"])
        assert False
    except AssertionError:
        pass

    try:
        DiscreteActionSpace(2, 3, action_names=["a", "c"])
        assert False
    except AssertionError:
        pass

    try:
        ContinuousActionSpace(2, 5, action_names=["a", "b", "c"])
        assert False
    except AssertionError:
        pass

    try:
        ContinuousActionSpace(2, 3, action_names=["a", "b"])
        assert False
    except AssertionError:
        pass
