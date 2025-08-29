from importlib.util import find_spec
from .pymarl_adapter import PymarlAdapter
from marlenv.utils import dummy_type, dummy_function

HAS_GYM = find_spec("gymnasium") is not None
if HAS_GYM:
    from .gym_adapter import Gym, make
else:
    Gym = dummy_type("gymnasium")
    make = dummy_function("gymnasium")

HAS_PETTINGZOO = find_spec("pettingzoo") is not None
if HAS_PETTINGZOO:
    from .pettingzoo_adapter import PettingZoo
else:
    PettingZoo = dummy_type("pettingzoo")

HAS_SMAC = find_spec("smac") is not None
if HAS_SMAC:
    from .smac_adapter import SMAC
else:
    SMAC = dummy_type("smac", "https://github.com/oxwhirl/smac.git")

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
