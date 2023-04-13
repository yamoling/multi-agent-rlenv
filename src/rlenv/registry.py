from typing import Type
from .models import RLEnv

registry: dict[str, Type[RLEnv]] = {}

def from_summary(summary: dict[str, ]) -> RLEnv:
    try:
        clss = registry[summary["name"]]
        return clss.from_summary(summary)
    except KeyError:
        # If the env is not registered, check if it is a gym env
        import rlenv
        return rlenv.make(summary["name"])

    


def register(env: Type[RLEnv]):
    registry[env.__name__] = env


