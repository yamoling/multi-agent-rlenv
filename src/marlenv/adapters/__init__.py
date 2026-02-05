from .pymarl_adapter import PymarlAdapter
from marlenv.utils import dummy_function

try:
    from .gym_adapter import Gym, make

    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    make = dummy_function("gymnasium")

try:
    from .pettingzoo_adapter import PettingZoo

    HAS_PETTINGZOO = True
except ImportError:
    HAS_PETTINGZOO = False

try:
    from .smac_adapter import SMAC

    HAS_SMAC = True
except ImportError:
    HAS_SMAC = False


__all__ = [
    "PymarlAdapter",
    "Gym",
    "make",
    "PettingZoo",
    "SMAC",
    "HAS_GYM",
    "HAS_PETTINGZOO",
    "HAS_SMAC",
]
