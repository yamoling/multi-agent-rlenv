from typing import Optional, Any
from types import SimpleNamespace


def _raise_error(module_name: str, package_name: Optional[str] = None):
    raise ImportError(
        f"The optional dependency `{module_name}` is not installed.\nInstall the `{package_name}` package (e.g. pip install {package_name})."
    )


def dummy_type(module_name: str, package_name: Optional[str] = None):
    class DummyType(type):
        def __getattr__(cls, _) -> Any:
            _raise_error(module_name, package_name)

        def __call__(self, *args: Any, **kwds: Any) -> Any:
            _raise_error(module_name, package_name)

    class DummyClass(SimpleNamespace, metaclass=DummyType):
        def __getattr__(self, _) -> Any:
            _raise_error(module_name, package_name)

    return DummyClass


def dummy_function(module_name: str, package_name: Optional[str] = None):
    def fail(*args, **kwargs) -> Any:
        _raise_error(module_name, package_name)

    return fail
