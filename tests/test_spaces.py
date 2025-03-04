import numpy as np
from marlenv.models import DiscreteActionSpace, ContinuousActionSpace, DiscreteSpace, MultiDiscreteSpace, ContinuousSpace


def test_discrete_action_space():
    s = DiscreteActionSpace(2, 3)
    available_actions = np.array([[True, True, False], [True, False, True]])
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
    s = ContinuousActionSpace(2, low=[0.0, -1.0, 0.0], high=[1.0, 1.0, 2.0])
    for _ in range(100):
        actions = s.sample()
        assert actions.shape == (2, 3)
        assert np.all(actions[:, 0] >= 0) and np.all(actions[:, 0] < 1)
        assert np.all(actions[:, 1] >= -1) and np.all(actions[:, 1] < 1)
        assert np.all(actions[:, 2] >= 0) and np.all(actions[:, 2] < 2)


def test_action_names():
    s = DiscreteActionSpace(2, 3, ["a", "b", "c"])
    assert s.action_names == ["a", "b", "c"]
    s = ContinuousActionSpace(2, [0, 0, 0], [1, 1, 1], action_names=["a", "b", "c"])
    assert s.action_names == ["a", "b", "c"]
    s = DiscreteActionSpace(2, 3)
    assert s.action_names == ["Action 0", "Action 1", "Action 2"]
    s = ContinuousActionSpace(2, [0, 1, 0], [1, 2, 1])
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
        ContinuousActionSpace(2, [0] * 5, [1] * 5, action_names=["a", "b", "c"])
        assert False
    except AssertionError:
        pass

    try:
        ContinuousActionSpace(2, [0] * 3, [1] * 3, action_names=["a", "b"])
        assert False
    except AssertionError:
        pass


def test_discrete_space_sample():
    s = DiscreteSpace(3)
    for _ in range(100):
        action = s.sample()
        assert action >= 0 and action < 3

    mask = np.array([True, False, True])
    for _ in range(100):
        action = s.sample(mask)
        assert action in [0, 2]


def test_multi_discrete_space():
    s = MultiDiscreteSpace(DiscreteSpace(5), DiscreteSpace(10))
    for _ in range(100):
        action = s.sample()
        assert action.shape == (2,)
        assert action[0] >= 0 and action[0] < 5
        assert action[1] >= 0 and action[1] < 10

    mask = [
        np.array([True, False, True, False, True]),
        np.array([True, True, False, False, True, True, False, True, False, True]),
    ]
    for _ in range(100):
        action = s.sample(mask)
        assert action[0] in [0, 2, 4]
        assert action[1] in [0, 1, 4, 5, 7, 9]


def test_wrong_continuous_space():
    try:
        ContinuousSpace([0, 1], [0, 1, 2])
    except AssertionError:
        pass

    try:
        ContinuousSpace([0, 1], [0, 0])
    except AssertionError:
        pass


def test_continuous_space():
    s = ContinuousSpace(low=[0.0, -1.0, 0.0], high=[1.0, 1.0, 2.0])
    for _ in range(100):
        action = s.sample()
        assert action.shape == (3,)
        assert np.all(action >= [0.0, -1.0, 0.0]) and np.all(action < [1.0, 1.0, 2.0])

    s = ContinuousSpace(low=[[0.0, 0.5], [-1, -1]], high=[[1.0, 1.0], [1.0, 1.0]])
    for _ in range(100):
        action = s.sample()
        assert action.shape == (2, 2)
        assert np.all(action >= [[0.0, 0.5], [-1, -1]]) and np.all(action < [[1.0, 1.0], [1.0, 1.0]])


def test_eq_spaces():
    s1 = DiscreteSpace(3)
    s2 = DiscreteSpace(3)
    s3 = DiscreteSpace(4)
    assert s1 == s2
    assert s1 != s3

    s4 = DiscreteActionSpace(2, 5)
    s5 = DiscreteActionSpace(2, 5)
    s6 = DiscreteActionSpace(2, 4)
    assert s4 == s5
    assert s4 != s6
