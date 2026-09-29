"""Random maze environment for comparing static and liquid (LTC) networks."""

from .game import DOWN, LEFT, RIGHT, UP, MazeGame
from .maze import DIRECTIONS, generate_maze, shortest_path

__all__ = [
    "DIRECTIONS",
    "DOWN",
    "LEFT",
    "RIGHT",
    "UP",
    "MazeGame",
    "generate_maze",
    "shortest_path",
]
