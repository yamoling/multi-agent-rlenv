from importlib.util import find_spec
from .pymarl_adapter import PymarlAdapter
from marlenv.utils import DummyClass, dummy_function

HAS_GYM = False
if find_spec("gymnasium") is not None:
    from .gym_adapter import Gym, make

    HAS_GYM = True
else:
    Gym = DummyClass("gymnasium")
    make = dummy_function("gymnasium")

HAS_PETTINGZOO = False
if find_spec("pettingzoo") is not None:
    from .pettingzoo_adapter import PettingZoo

    HAS_PETTINGZOO = True

HAS_SMAC = False
if find_spec("smac") is not None:
    from .smac_adapter import SMAC

    HAS_SMAC = True

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
