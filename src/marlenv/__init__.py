"""
`marlenv` is a strongly typed library for multi-agent and multi-objective reinforcement learning.

It aims to
    - provide a simple and consistent interface for reinforcement learning environments
    - provide fundamental models such as `Observation`s, `Episode`s, `Transition`s, ...
    - work with gymnasium, pettingzoo and SMAC out of the box
    - work with multi-objective environments
    - provide helpful wrappers to add intrinsic rewards, agent ids, record videos, ...
"""

__version__ = "1.2.0"

from . import models
from . import wrappers
from . import adapters
from .models import spaces


from .env_builder import make, Builder
from .models import (
    MARLEnv,
    # MOMARLEnv,
    Observation,
    Episode,
    EpisodeBuilder,
    Transition,
    DiscreteSpace,
    ContinuousSpace,
    ActionSpace,
    DiscreteActionSpace,
    ContinuousActionSpace,
)
from .mock_env import MockEnv, MOMockEnv

__all__ = [
    "models",
    "wrappers",
    "adapters",
    "spaces",
    "make",
    "Builder",
    "MARLEnv",
    # "MOMARLEnv",
    "Observation",
    "Episode",
    "EpisodeBuilder",
    "Transition",
    "ActionSpace",
    "DiscreteSpace",
    "ContinuousSpace",
    "DiscreteActionSpace",
    "ContinuousActionSpace",
    "MockEnv",
    "MOMockEnv",
]
