"""Headless maze game state. No pygame here, so agents can drive it directly."""

from __future__ import annotations

import random
import time

from .maze import DIRECTIONS, generate_maze, shortest_path

UP, RIGHT, DOWN, LEFT = range(4)


class MazeGame:
    """One agent in a random maze, starting top-left with the goal bottom-right.

    ``new_maze()`` starts a fresh maze and ``step(action)`` moves one tile. The
    timer starts on the first step and stops when the goal is reached.
    """

    def __init__(self, rows: int = 30, cols: int = 40, seed: int | None = None):
        if rows < 2 or cols < 2:
            raise ValueError("rows and cols must be at least 2")
        self.rows = rows
        self.cols = cols
        # Draws one seed per maze: a fixed `seed` reproduces the whole sequence of
        # mazes, while every retry still gets a different maze.
        self._seed_rng = random.Random(seed)
        self.new_maze()

    def new_maze(self, seed: int | None = None) -> None:
        """Generate a new maze and reset the agent. Pass `seed` to replay one maze."""
        self.maze_seed = self._seed_rng.randrange(2**32) if seed is None else seed
        self.grid = generate_maze(self.rows, self.cols, random.Random(self.maze_seed))
        self.start = (1, 1)
        self.goal = (2 * self.rows - 1, 2 * self.cols - 1)
        self.pos = self.start
        self.heading = RIGHT
        self.moves = 0
        self.wall_hits = 0
        self.solved = False
        self._start_time: float | None = None
        self._end_time: float | None = None

    def step(self, action: int) -> bool:
        """Face `action` (UP/RIGHT/DOWN/LEFT) and try to move one tile. Returns True if moved."""
        if action not in range(4):
            raise ValueError(f"invalid action {action!r}; expected 0-3")
        if self.solved:
            return False
        if self._start_time is None:
            self._start_time = time.perf_counter()
        self.heading = action
        dr, dc = DIRECTIONS[action]
        target = (self.pos[0] + dr, self.pos[1] + dc)
        if self.grid[target]:
            self.wall_hits += 1
            return False
        self.pos = target
        self.moves += 1
        if self.pos == self.goal:
            self.solved = True
            self._end_time = time.perf_counter()
        return True

    @property
    def elapsed(self) -> float:
        """Seconds since the first step, frozen once the goal is reached."""
        if self._start_time is None:
            return 0.0
        end = self._end_time if self._end_time is not None else time.perf_counter()
        return end - self._start_time

    def shortest_path_length(self) -> int:
        """Minimum number of moves from start to goal in the current maze."""
        return len(shortest_path(self.grid, self.start, self.goal)) - 1
