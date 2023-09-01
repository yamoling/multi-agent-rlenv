try:
    from .gym_adapter import GymAdapter
except ImportError:
    GymAdapter = None

try:
    from .pettingzoo_adapter import PettingZooAdapter
except ImportError:
    PettingZooAdapter = None

try:
    from .smac_adapter import SMACAdapter
except ImportError:
    SMACAdapter = None
