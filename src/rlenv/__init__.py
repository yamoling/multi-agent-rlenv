"""
RLEnv is a strongly typed library for reinforcement learning.

RLEnv 
    - works with gym(-nasium), pettingzoo and SMAC 
    - provides helpful wrappers to add intrinsic rewards, agent ids, record videos, ...
"""
__version__ = "0.5.2"

from . import models
from . import wrappers
from . import adapters


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
