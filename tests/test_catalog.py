import pytest
from marlenv import catalog
from marlenv.utils import DummyClass, dummy_function

skip_lle = isinstance(catalog.LLE, DummyClass)
skip_overcooked = isinstance(catalog.Overcooked, DummyClass)


@pytest.mark.skipif(skip_lle, reason="LLE is not installed")
def test_lle():
    catalog.LLE.level(1)


@pytest.mark.skipif(skip_overcooked, reason="Overcooked is not installed")
def test_overcooked():
    catalog.Overcooked.from_layout("scenario4")


def test_dummy_class():
    try:
        x = DummyClass("")
        x.abc
        assert False, "Expected ImportError upon usage because DummyClass is not installed"
    except ImportError:
        pass

    try:
        x = DummyClass("")
        x.abc()
        assert False, "Expected ImportError upon usage because DummyClass is not installed"
    except ImportError:
        pass


def test_dummy_function():
    try:
        f = dummy_function("")
        f()
        assert False, "Expected ImportError upon usage because dummy_function is not installed"
    except ImportError:
        pass
