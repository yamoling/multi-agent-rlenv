from importlib.util import find_spec
from ..utils.import_placeholders import DummyClass
from marlenv.adapters import SMAC
from .deepsea import DeepSea


spec = find_spec("lle")
if spec is not None:
    from lle import LLE  # pyright: ignore[reportMissingImports]
else:
    LLE = DummyClass("lle", "laser-learning-environment")

spec = find_spec("overcooked")
if spec is not None:
    from overcooked import Overcooked  # pyright: ignore[reportMissingImports]
else:
    Overcooked = DummyClass("overcooked", "overcooked")

__all__ = ["Overcooked", "SMAC", "LLE", "DeepSea"]
