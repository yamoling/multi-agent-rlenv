from types import SimpleNamespace
from typing import Any, Optional


def _raise_error(module_name: str, package_name: str | None = None):
    if package_name is None:
        package_name = module_name
    raise ImportError(
        f"The optional dependency `{module_name}` is not installed.\nInstall the `{package_name}` package (e.g. pip install {package_name})."
    )


def dummy_type(module_name: str, package_name: str | None = None):
    class DummyType(type):
        def __getattr__(cls, _) -> Any:
            _raise_error(module_name, package_name)

        def __call__(self, *args: Any, **kwds: Any) -> Any:
            _raise_error(module_name, package_name)

    class DummyClass(SimpleNamespace, metaclass=DummyType):
        def __getattr__(self, _) -> Any:
            _raise_error(module_name, package_name)

    return DummyClass


def dummy_function(module_name: str, package_name: str | None = None):
    def fail(*args, **kwargs) -> Any:
        _raise_error(module_name, package_name)

    return fail
