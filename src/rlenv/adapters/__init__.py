__all__ = []
try:
    from .gym_adapter import Gym

    __all__.append("Gym")
except ImportError:
    Gym = None

try:
    from .pettingzoo_adapter import PettingZoo

    __all__.append("PettingZoo")
except ImportError:
    PettingZoo = None

try:
    from .smac_adapter import SMAC

    __all__.append("SMAC")
except ImportError:
    SMAC = None
