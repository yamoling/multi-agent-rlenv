import pytest
from marlenv import catalog
from marlenv.utils import dummy_type, dummy_function

skip_lle = not catalog.HAS_LLE
skip_overcooked = not catalog.HAS_OVERCOOKED


@pytest.mark.skipif(skip_lle, reason="LLE is not installed")
def test_lle():
    catalog.LLE.level(1)


@pytest.mark.skipif(skip_overcooked, reason="Overcooked is not installed")
def test_overcooked():
    catalog.Overcooked.from_layout("scenario4")


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
