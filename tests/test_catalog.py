import pytest
from marlenv import catalog
from marlenv.utils import dummy_type, dummy_function

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
