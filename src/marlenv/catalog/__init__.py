from importlib.util import find_spec
from ..utils.import_placeholders import dummy_type
from marlenv.adapters import SMAC
from .deepsea import DeepSea


HAS_LLE = find_spec("lle") is not None
if HAS_LLE:
    from lle import LLE  # pyright: ignore[reportMissingImports]
else:
    LLE = dummy_type("lle", "laser-learning-environment")

HAS_OVERCOOKED = find_spec("overcooked") is not None
if HAS_OVERCOOKED:
    from overcooked import Overcooked  # pyright: ignore[reportMissingImports]
else:
    Overcooked = dummy_type("overcooked", "overcooked")

__all__ = [
    "Overcooked",
    "SMAC",
    "LLE",
    "DeepSea",
    "HAS_LLE",
    "HAS_OVERCOOKED",
]
