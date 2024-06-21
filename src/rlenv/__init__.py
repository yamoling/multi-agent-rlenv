"""
RLEnv is a strongly typed library for multi-agent and multi-objective reinforcement learning.

RLEnv
    - provides a simple and consistent interface for reinforcement learning environments
    - provides fundamental models such as `Observation`s, `Episode`s, `Transition`s, ...
    - works with gymnasium, pettingzoo and SMAC out of the box
    - provides helpful wrappers to add intrinsic rewards, agent ids, record videos, ...
"""

__version__ = "1.0.4"

from . import models
from . import wrappers
from . import adapters
from .models import spaces


from .env_builder import make, Builder
from .models import (
    RLEnv,
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
from .mock_env import MockEnv

__all__ = [
    "models",
    "wrappers",
    "adapters",
    "spaces",
    "make",
    "Builder",
    "RLEnv",
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
]
