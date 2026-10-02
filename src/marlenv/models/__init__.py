"""
Core data models for the `marlenv` API.

This package defines the typed containers and interfaces shared across adapters,
wrappers, and environments:
- `MARLEnv`: the abstract environment contract.
- `Observation` / `State`: structured inputs to agents and state tracking.
- `Step` / `Transition` / `Episode`: execution results and replayable logs.
- `Space` variants: action/reward space definitions.
- `ParallelMARLEnv` / `ParallelStep`: batched execution of several environments.
"""

from .env import ContinuousMARLEnv, DiscreteMARLEnv, MARLEnv
from .episode import Episode
from .observation import Observation
from .parallel import ParallelMARLEnv, ParallelObservation, ParallelState, ParallelStep
from .spaces import ContinuousSpace, DiscreteSpace, MultiDiscreteSpace, Space
from .state import State
from .step import Step
from .transition import Transition

__all__ = [
    "ContinuousMARLEnv",
    "ContinuousSpace",
    "DiscreteMARLEnv",
    "DiscreteSpace",
    "Episode",
    "MARLEnv",
    "MultiDiscreteSpace",
    "Observation",
    "ParallelMARLEnv",
    "ParallelObservation",
    "ParallelState",
    "ParallelStep",
    "Space",
    "State",
    "Step",
    "Transition",
]
