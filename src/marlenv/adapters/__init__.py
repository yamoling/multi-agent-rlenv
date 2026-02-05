"""
Adapters for external RL libraries.

This submodule provides optional wrappers that normalize third-party APIs into
`MARLEnv`. Adapters are imported lazily via `try/except` so the base install
remains lightweight. The availability flags (`HAS_GYM`, `HAS_PETTINGZOO`,
`HAS_SMAC`) reflect whether the corresponding extra was installed.

Install extras to enable adapters with `uv` or `pip`:
- `multi-agent-rlenv[all]` for all optional dependencies
- `multi-agent-rlenv[gym]` for Gymnasium
- `multi-agent-rlenv[pettingzoo]` for PettingZoo
- `multi-agent-rlenv[smac]` for SMAC
"""

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
