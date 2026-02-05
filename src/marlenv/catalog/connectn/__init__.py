"""
Connect-N game environment.

Inspiration from: https://github.com/Gualor/connect4-montecarlo
"""

from .board import GameBoard
from .env import ConnectN


__all__ = ["ConnectN", "GameBoard"]
