"""
RLEnv is a strongly typed library for reinforcement learning.

RLEnv 
    - works with gym(-nasium), pettingzoo and SMAC 
    - provides helpful wrappers to add intrinsic rewards, agent ids, record videos, ...
"""
__version__ = "0.5.1"

from . import models
from . import wrappers
from . import adapters

from .env_builder import make, Builder
from .models import (
    RLEnv,
    Observation,
    Episode,
    Transition,
    ActionSpace,
    DiscreteActionSpace,
    ContinuousActionSpace,
)

__all__ = [
    "models",
    "wrappers",
    "adapters",
    "make",
    "Builder",
    "RLEnv",
    "Observation",
    "Episode",
    "Transition",
    "ActionSpace",
    "DiscreteActionSpace",
    "ContinuousActionSpace",
]
