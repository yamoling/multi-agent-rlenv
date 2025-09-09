from marlenv.adapters import SMAC
from .deepsea import DeepSea


__all__ = [
    "SMAC",
    "DeepSea",
    "lle",
    "overcooked",
]


def lle():
    from lle import LLE  # pyright: ignore[reportMissingImports]

    return LLE


def overcooked():
    from overcooked import Overcooked  # pyright: ignore[reportMissingImports]

    return Overcooked
