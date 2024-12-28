"""
`marlenv` is a strongly typed library for multi-agent and multi-objective reinforcement learning.

It aims to
    - provide a simple and consistent interface for reinforcement learning environments
    - provide fundamental models such as `Observation`s, `Episode`s, `Transition`s, ...
    - work with gymnasium, pettingzoo and SMAC out of the box
    - work with multi-objective environments
    - provide helpful wrappers to add intrinsic rewards, agent ids, record videos, ...


A design choice is taht almost every class is a dataclass. This makes it easy to
serialize and deserialize classes, for instance to json with the `orjson` library.
"""

__version__ = "3.1.1"

from . import models
from . import wrappers
from . import adapters
from .models import spaces


from .env_builder import make, Builder
from .models import (
    MARLEnv,
    State,
    Step,
    Observation,
    Episode,
    Transition,
    DiscreteSpace,
    ContinuousSpace,
    ActionSpace,
    DiscreteActionSpace,
    ContinuousActionSpace,
)
from .mock_env import DiscreteMockEnv, DiscreteMOMockEnv

__all__ = [
    "models",
    "wrappers",
    "adapters",
    "spaces",
    "make",
    "Builder",
    "MARLEnv",
    "Step",
    "State",
    "Observation",
    "Episode",
    "Transition",
    "ActionSpace",
    "DiscreteSpace",
    "ContinuousSpace",
    "DiscreteActionSpace",
    "ContinuousActionSpace",
    "DiscreteMockEnv",
    "DiscreteMOMockEnv",
]
