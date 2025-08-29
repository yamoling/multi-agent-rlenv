from typing import Optional, Any


class DummyClass:
    def __init__(self, module_name: str, package_name: Optional[str] = None):
        self.module_name = module_name
        if package_name is None:
            self.package_name = module_name
        else:
            self.package_name = package_name

    def _raise_error(self):
        raise ImportError(
            f"The optional dependency `{self.module_name}` is not installed.\nInstall the `{self.package_name}` package (e.g. pip install {self.package_name})."
        )

    def __getattr__(self, _):
        self._raise_error()

    def __call__(self, *args, **kwargs):
        self._raise_error()


def dummy_function(module_name: str, package_name: Optional[str] = None):
    dummy = DummyClass(module_name, package_name)

    def fail(*args, **kwargs) -> Any:
        dummy()

    return fail
