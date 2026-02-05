from marlenv.adapters import SMAC
from .deepsea import DeepSea
from .matrix_game import MatrixGame
from .coordinated_grid import CoordinatedGrid


__all__ = ["SMAC", "DeepSea", "lle", "overcooked", "MatrixGame", "connect_n", "CoordinatedGrid"]


def lle():
    from lle import LLE  # pyright: ignore[reportMissingImports]

    return LLE


def overcooked():
    from overcooked import Overcooked  # pyright: ignore[reportMissingImports]

    return Overcooked


def connect_n():
    from .connectn import ConnectN

    return ConnectN
