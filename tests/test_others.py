import marlenv


def test_version():
    assert hasattr(marlenv, "__version__")
    x, y, z = marlenv.__version__.split(".")
