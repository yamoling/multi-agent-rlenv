import numpy as np
from marlenv.models import DiscreteSpace, MultiDiscreteSpace, ContinuousSpace
import itertools


def test_discrete_action_space():
    s = DiscreteSpace(3).repeat(2)
    available_actions = np.array([[True, True, False], [True, False, True]])
    for _ in range(100):
        actions = s.sample(mask=available_actions)
        assert actions.shape == (2,)
        assert actions[0] in [0, 1]
        assert actions[1] in [0, 2]

        actions = s.sample()
        assert actions.shape == (2,)
        assert actions[0] in [0, 1, 2]
        assert actions[1] in [0, 1, 2]


def test_continuous_action_space():
    s = ContinuousSpace(low=[0.0, -1.0, 0.0], high=[1.0, 1.0, 2.0]).repeat(2)
    for _ in range(100):
        actions = s.sample()
        assert actions.shape == (2, 3)
        assert np.all(actions[:, 0] >= 0) and np.all(actions[:, 0] < 1)
        assert np.all(actions[:, 1] >= -1) and np.all(actions[:, 1] < 1)
        assert np.all(actions[:, 2] >= 0) and np.all(actions[:, 2] < 2)


def test_is_discrete():
    s = DiscreteSpace(3)
    assert s.is_discrete
    s = MultiDiscreteSpace(DiscreteSpace(3), DiscreteSpace(4))
    assert s.is_discrete
    s = ContinuousSpace(low=[0.0, -1.0, 0.0], high=[1.0, 1.0, 2.0])
    assert not s.is_discrete
    s = ContinuousSpace(low=[[0.0, 0.5], [-1, -1]], high=[[1.0, 1.0], [1.0, 1.0]])
    assert not s.is_discrete


def test_is_continuous():
    s = DiscreteSpace(3)
    assert not s.is_continuous
    s = MultiDiscreteSpace(DiscreteSpace(3), DiscreteSpace(4))
    assert not s.is_continuous
    s = ContinuousSpace(low=[0.0, -1.0, 0.0], high=[1.0, 1.0, 2.0])
    assert s.is_continuous
    s = ContinuousSpace(low=[[0.0, 0.5], [-1, -1]], high=[[1.0, 1.0], [1.0, 1.0]])
    assert s.is_continuous


def test_action_names():
    s = DiscreteSpace(3, ["a", "b", "c"])
    assert s.labels == ["a", "b", "c"]
    s = ContinuousSpace([0, 0, 0], [1, 1, 1], ["a", "b", "c"])
    assert s.labels == ["a", "b", "c"]
    s = DiscreteSpace.action(3)
    assert s.labels == ["Action 0", "Action 1", "Action 2"]
    s = ContinuousSpace([0, 1, 0], [1, 2, 1])
    assert s.labels == ["Dim 0", "Dim 1", "Dim 2"]


def test_action_names_wrong_number_of_actions():
    try:
        DiscreteSpace(5, ["a", "b", "c"])
        assert False, "The number of labels does not match the number of actions"
    except AssertionError:
        pass

    try:
        DiscreteSpace(3, ["a", "c"])
        assert False, "The number of labels does not match the number of actions"
    except AssertionError:
        pass

    try:
        ContinuousSpace([0.0] * 5, [1.0] * 5, ["a", "b", "c"])
        assert False, "The number of labels does not match the number of dimensions"
    except AssertionError:
        pass

    try:
        ContinuousSpace([0.0] * 3, [1.0] * 3, ["a", "b"])
        assert False, "The number of labels does not match the number of dimensions"
    except AssertionError:
        pass


def test_discrete_space_sample():
    s = DiscreteSpace(3)
    for _ in range(100):
        action = s.sample()
        assert action >= 0 and action < 3

    mask = np.array([True, False, True])
    for _ in range(100):
        action = s.sample(mask=mask)
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


def test_continuous_space_clamp():
    s = ContinuousSpace(low=[0.0, -1.0, 0.0], high=[1.0, 1.0, 2.0])
    action = np.array([2.0, -2.0, 3.0])
    clamped_action = s.clamp(action)
    assert clamped_action[0] == 1.0
    assert clamped_action[1] == -1.0
    assert clamped_action[2] == 2.0


def test_continuous_space_clamp_inf():
    s = ContinuousSpace(low=[0.0, -1.0, 0.0], high=None)
    action = np.array([2.0, -2.0, 3.0])
    clamped_action = s.clamp(action)
    assert clamped_action[0] == 2.0
    assert clamped_action[1] == -1.0
    assert clamped_action[2] == 3.0


def test_continuous_space_clamp_batch():
    s = ContinuousSpace(low=[0.0, -1.0, 0.0], high=[1.0, 1.0, 2.0])
    actions = np.array(list(itertools.product(range(-10, 10), repeat=3)))
    clamped_actions = np.array([s.clamp(action) for action in actions])
    assert clamped_actions.shape == (len(actions), 3)
    assert np.all(clamped_actions >= [0.0, -1.0, 0.0]) and np.all(clamped_actions <= [1.0, 1.0, 2.0])


def test_eq_spaces():
    s1 = DiscreteSpace(3)
    s2 = DiscreteSpace(3)
    s3 = DiscreteSpace(4)
    assert s1 == s2
    assert s1 != s3

    s4 = DiscreteSpace(5)
    s5 = DiscreteSpace(5)
    s6 = DiscreteSpace(4)
    assert s4 == s5
    assert s4 != s6
