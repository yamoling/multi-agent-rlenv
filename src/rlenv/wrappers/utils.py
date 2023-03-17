from typing import Type
import inspect

from .rlenv_wrapper import RLEnvWrapper, RLEnv

def _get_all_wrappers() -> dict[str, Type[RLEnvWrapper]]:
    from rlenv import wrappers
    classes = inspect.getmembers(wrappers, inspect.isclass)
    return {name: cls for name, cls in classes if issubclass(cls, RLEnvWrapper) and cls is not RLEnvWrapper}

def from_summary(env: RLEnv, summary: dict) -> RLEnvWrapper:
    """Create a wrapper from its summary"""
    wrappers_list: list[str] = summary["wrappers"]
    for wrapper_name in wrappers_list:
        wrapper_class = ALL_WRAPPERS[wrapper_name]
        env = wrapper_class.from_summary(env, summary)
    return env

def register(wrapper_class: Type[RLEnvWrapper]):
    """Register a wrapper class"""
    ALL_WRAPPERS[wrapper_class.__name__] = wrapper_class

ALL_WRAPPERS: dict[str, Type[RLEnvWrapper]] = _get_all_wrappers()
