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
- `multi-agent-rlenv[smacv2]` for SMACv2
"""

from marlenv.utils import dummy_function

from .pymarl_adapter import PymarlAdapter


def _is_missing_optional_dependency(error: ImportError, *module_names: str) -> bool:
    """
    Whether `error` is one of `module_names` simply not being installed.

    Returns False for any other import failure -- a circular import, a missing core
    dependency, a typo inside the adapter -- so that a genuine bug is raised loudly
    instead of being silently downgraded to "the optional extra is not installed",
    which would make every dependent test skip rather than fail.
    """
    if not isinstance(error, ModuleNotFoundError):
        return False
    return (error.name or "").split(".")[0] in module_names


try:
    from .gym_adapter import Gym, ToGym, make

    HAS_GYM = True
except ImportError as error:
    if not _is_missing_optional_dependency(error, "gymnasium"):
        raise
    HAS_GYM = False
    make = dummy_function("gymnasium")

try:
    from .pettingzoo_adapter import PettingZoo

    HAS_PETTINGZOO = True
except ImportError as error:
    if not _is_missing_optional_dependency(error, "pettingzoo", "gymnasium"):
        raise
    HAS_PETTINGZOO = False

try:
    from .smac_adapter import SMAC

    HAS_SMAC = True
except ImportError as error:
    if not _is_missing_optional_dependency(error, "smac", "pysc2"):
        raise
    HAS_SMAC = False


try:
    from .smacv2_adapter import SMACv2

    HAS_SMACv2 = True
except ImportError as error:
    if not _is_missing_optional_dependency(error, "smacv2", "pysc2"):
        raise
    HAS_SMACv2 = False


__all__ = [
    "HAS_GYM",
    "HAS_PETTINGZOO",
    "HAS_SMAC",
    "SMAC",
    "Gym",
    "HAS_SMACv2",
    "PettingZoo",
    "PymarlAdapter",
    "SMACv2",
    "ToGym",
    "make",
]
