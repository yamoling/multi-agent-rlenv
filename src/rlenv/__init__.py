"""
RLEnv is a strongly typed library for reinforcement learning.

RLEnv 
    - works with gym(-nasium), pettingzoo and SMAC 
    - provides helpful wrappers to add extrinsic rewards, agent ids, record videos, ...
"""
__version__ = "0.2.6"

from . import models
from . import wrappers
from . import adapters

from .env_factory import make, Builder
from .models import RLEnv, Observation, Episode, Transition
