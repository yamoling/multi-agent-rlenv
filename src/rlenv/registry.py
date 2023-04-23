import inspect
from typing import Type
from gymnasium.error import NameNotFound
from .models import RLEnv
from .wrappers import RLEnvWrapper
from . import wrappers
from .exceptions import UnknownEnvironmentException

def _get_all_wrappers() -> dict[str, Type[RLEnvWrapper]]:
    classes = inspect.getmembers(wrappers, inspect.isclass)
    return {name: cls for name, cls in classes if issubclass(cls, RLEnvWrapper) and cls is not RLEnvWrapper} 


ENV_REGISTRY: dict[str, Type[RLEnv]] = {}
WRAPPER_REGISTRY: dict[str, Type[RLEnvWrapper]] = _get_all_wrappers()


def from_summary(summary: dict[str, ]) -> RLEnv:
    try:
        clss = ENV_REGISTRY[summary["name"]]
        env = clss.from_summary(summary)
    except KeyError: # If the env is not registered, check if it is a gym env
        import rlenv
        try:
            env = rlenv.make(summary["name"])
        except NameNotFound: # If that fails, then raise an error
            raise UnknownEnvironmentException(summary["name"])
    wrappers_list: list[str] = summary.get("wrappers", [])
    for wrapper_name in wrappers_list:
        wrapper_class = WRAPPER_REGISTRY[wrapper_name]
        env = wrapper_class.from_summary(env, summary)
    return env



def register(env: Type[RLEnv]):
    ENV_REGISTRY[env.__name__] = env

def register_wrapper(wrapper: Type[RLEnvWrapper]):
    WRAPPER_REGISTRY[wrapper.__name__] = wrapper


