from .pymarl_adapter import PymarlAdapter
from typing import Any

__all__ = ["PymarlAdapter"]
try:
    from .gym_adapter import Gym

    __all__.append("Gym")
except ImportError:
    Gym = Any

try:
    from .pettingzoo_adapter import PettingZoo

    __all__.append("PettingZoo")
except ImportError:
    PettingZoo = Any

try:
    from .smac_adapter import SMAC

    __all__.append("SMAC")
except ImportError:
    SMAC = Any
