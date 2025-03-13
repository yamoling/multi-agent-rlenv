from importlib.util import find_spec
from .pymarl_adapter import PymarlAdapter

HAS_GYM = False
if find_spec("gymnasium") is not None:
    from .gym_adapter import Gym

    HAS_GYM = True

HAS_PETTINGZOO = False
if find_spec("pettingzoo") is not None:
    from .pettingzoo_adapter import PettingZoo

    HAS_PETTINGZOO = True

HAS_SMAC = False
if find_spec("smac") is not None:
    from .smac_adapter import SMAC

    HAS_SMAC = True

HAS_OVERCOOKED = False
if find_spec("overcooked_ai_py.mdp") is not None:
    import numpy

    # Overcooked assumes a version of numpy <2.0 where np.Inf is available.
    setattr(numpy, "Inf", numpy.inf)
    from .overcooked_adapter import Overcooked

    HAS_OVERCOOKED = True

__all__ = [
    "PymarlAdapter",
    "Gym",
    "PettingZoo",
    "SMAC",
    "Overcooked",
    "HAS_GYM",
    "HAS_PETTINGZOO",
    "HAS_SMAC",
    "HAS_OVERCOOKED",
]
