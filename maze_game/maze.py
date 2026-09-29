"""Random maze generation and solving on a block grid.

A maze with ``rows x cols`` cells is a boolean array of shape
``(2 * rows + 1, 2 * cols + 1)`` where ``True`` is a wall. Cell ``(r, c)`` sits at
grid position ``(2r + 1, 2c + 1)`` and the tiles between cells are either walls or
open passages. Every tile is the same size, so an agent moves one tile per step.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np

# (d_row, d_col) for up, right, down, left; the index is the action id.
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def generate_maze(rows: int, cols: int, rng: random.Random | None = None) -> np.ndarray:
    """Return a random perfect maze: exactly one path between any two cells.

    Uses an iterative randomized depth-first search (the "recursive backtracker"),
    which gives long winding corridors with many dead ends.
    """
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be at least 1")
    rng = rng or random.Random()
    grid = np.ones((2 * rows + 1, 2 * cols + 1), dtype=bool)
    visited = [[False] * cols for _ in range(rows)]

    start = (rng.randrange(rows), rng.randrange(cols))
    visited[start[0]][start[1]] = True
    grid[2 * start[0] + 1, 2 * start[1] + 1] = False
    stack = [start]
    while stack:
        r, c = stack[-1]
        unvisited = [
            (r + dr, c + dc)
            for dr, dc in DIRECTIONS
            if 0 <= r + dr < rows and 0 <= c + dc < cols and not visited[r + dr][c + dc]
        ]
        if not unvisited:
            stack.pop()
            continue
        nr, nc = rng.choice(unvisited)
        visited[nr][nc] = True
        grid[r + nr + 1, c + nc + 1] = False  # knock down the wall between the two cells
        grid[2 * nr + 1, 2 * nc + 1] = False
        stack.append((nr, nc))
    return grid


def shortest_path(
    grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]] | None:
    """Return the tiles on a shortest path from start to goal (both included), or None."""
    parents = {start: None}
    queue = deque([start])
    while queue:
        tile = queue.popleft()
        if tile == goal:
            path = []
            while tile is not None:
                path.append(tile)
                tile = parents[tile]
            return path[::-1]
        for dr, dc in DIRECTIONS:
            nxt = (tile[0] + dr, tile[1] + dc)
            if (
                0 <= nxt[0] < grid.shape[0]
                and 0 <= nxt[1] < grid.shape[1]
                and not grid[nxt]
                and nxt not in parents
            ):
                parents[nxt] = tile
                queue.append(nxt)
    return None
