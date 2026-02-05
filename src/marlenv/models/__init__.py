"""
Core data models for the `marlenv` API.

This package defines the typed containers and interfaces shared across adapters,
wrappers, and environments:
- `MARLEnv`: the abstract environment contract.
- `Observation` / `State`: structured inputs to agents and state tracking.
- `Step` / `Transition` / `Episode`: execution results and replayable logs.
- `Space` variants: action/reward space definitions.
"""

from .spaces import DiscreteSpace, ContinuousSpace, MultiDiscreteSpace, Space
from .observation import Observation
from .step import Step
from .state import State
from .env import MARLEnv
from .transition import Transition
from .episode import Episode


__all__ = [
    "Step",
    "State",
    "DiscreteSpace",
    "ContinuousSpace",
    "Observation",
    "MARLEnv",
    "Transition",
    "Episode",
    "MultiDiscreteSpace",
    "Space",
]
