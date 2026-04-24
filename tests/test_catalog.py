import pytest

from marlenv import catalog
from marlenv.utils import dummy_function, dummy_type

try:
    catalog.lle()
    skip_lle = False
except ImportError:
    skip_lle = True

try:
    catalog.overcooked()
    skip_overcooked = False
except ImportError:
    skip_overcooked = True


@pytest.mark.skipif(skip_lle, reason="LLE is not installed")
def test_lle():
    catalog.lle().level(1)


@pytest.mark.skipif(skip_overcooked, reason="Overcooked is not installed")
def test_overcooked():
    catalog.overcooked().from_layout("scenario4")


def test_dummy_type():
    try:
        x = dummy_type("")
        x.abc
        assert False, "Expected ImportError upon usage because dummy_type is not installed"
    except ImportError:
        pass

    try:
        x = dummy_type("")
        x.abc()  # type: ignore
        assert False, "Expected ImportError upon usage because dummy_type is not installed"
    except ImportError:
        pass


def test_dummy_function():
    try:
        f = dummy_function("")
        f()
        assert False, "Expected ImportError upon usage because dummy_function is not installed"
    except ImportError:
        pass


def test_m_steps_matrix_optimal_path():
    from marlenv.catalog.m_steps_matrix import Action

    env = catalog.MStepsMatrix(10)
    actions = [Action.TOP_LEFT] * 9 + [Action.BOTTOM_RIGHT]
    for _ in range(5):
        env.reset()
        score = 0
        for action in actions:
            step = env.step(action.to_tuple())
            score += step.reward.item()
        assert score == 13


def test_m_steps_matrix_suboptimal_path_left():
    from marlenv.catalog.m_steps_matrix import Action

    env = catalog.MStepsMatrix(10)
    env.reset()
    actions = [Action.TOP_LEFT] * 9
    last_actions = [Action.TOP_LEFT, Action.TOP_RIGHT, Action.BOTTOM_LEFT]
    for last in last_actions:
        env.reset()
        score = 0
        for action in actions:
            step = env.step(action.to_tuple())
            score += step.reward.item()
        step = env.step(last.to_tuple())
        score += step.reward.item()
        assert score == 10


def test_m_steps_matrix_suboptimal_path_right():
    from marlenv.catalog.m_steps_matrix import Action

    env = catalog.MStepsMatrix(10)
    env.reset()
    actions = [Action.BOTTOM_RIGHT] * 9
    for last in Action:
        env.reset()
        score = 0
        for action in actions:
            step = env.step(action.to_tuple())
            score += step.reward.item()
        step = env.step(last.to_tuple())
        score += step.reward.item()
        assert score == 10


def test_m_steps_other_paths():
    from marlenv.catalog.m_steps_matrix import Action

    env = catalog.MStepsMatrix(10)
    env.reset()
    assert env.step(Action.TOP_RIGHT.to_tuple()).done
    env.reset()
    assert env.step(Action.BOTTOM_LEFT.to_tuple()).done

    # Test left path
    for i in range(8):
        for last_action in [Action.TOP_RIGHT, Action.BOTTOM_LEFT, Action.BOTTOM_RIGHT]:
            env.reset()
            actions = [Action.TOP_LEFT] * (i + 1)
            for action in actions:
                step = env.step(action.to_tuple())
                assert not step.is_terminal
            step = env.step(last_action.to_tuple())
            assert step.done
            assert step.reward.item() == 0
    # Test right path
    for i in range(8):
        for last_action in [Action.TOP_LEFT, Action.TOP_RIGHT, Action.BOTTOM_LEFT]:
            env.reset()
            actions = [Action.BOTTOM_RIGHT] * (i + 1)
            for action in actions:
                step = env.step(action.to_tuple())
                assert not step.is_terminal
            step = env.step(last_action.to_tuple())
            assert step.done
            assert step.reward.item() == 0
