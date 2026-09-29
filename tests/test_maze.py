import os
import random
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from maze_game import DIRECTIONS, DOWN, LEFT, RIGHT, UP, MazeGame, generate_maze, shortest_path
from maze_game.app import MazeApp

ACTION_TO_KEY = {UP: pygame.K_UP, RIGHT: pygame.K_RIGHT, DOWN: pygame.K_DOWN, LEFT: pygame.K_LEFT}


def actions_along(path):
    return [DIRECTIONS.index((b[0] - a[0], b[1] - a[1])) for a, b in zip(path, path[1:])]


class GenerateMazeTest(unittest.TestCase):
    def test_is_a_perfect_maze(self):
        for seed in range(20):
            rows, cols = 7 + seed % 5, 9 + seed % 3
            grid = generate_maze(rows, cols, random.Random(seed))
            self.assertEqual(grid.shape, (2 * rows + 1, 2 * cols + 1))
            # Border and the pillars between cells are always walls.
            self.assertTrue(grid[0].all() and grid[-1].all() and grid[:, 0].all() and grid[:, -1].all())
            self.assertTrue(grid[::2, ::2].all())
            # Every cell is open and reachable, and cells + passages form a tree.
            self.assertFalse(grid[1::2, 1::2].any())
            self.assertEqual((~grid).sum(), 2 * rows * cols - 1)
            for r in range(rows):
                for c in range(cols):
                    self.assertIsNotNone(shortest_path(grid, (1, 1), (2 * r + 1, 2 * c + 1)))

    def test_seed_is_reproducible(self):
        a = generate_maze(20, 20, random.Random(5))
        b = generate_maze(20, 20, random.Random(5))
        c = generate_maze(20, 20, random.Random(6))
        self.assertTrue((a == b).all())
        self.assertFalse((a == c).all())


class MazeGameTest(unittest.TestCase):
    def test_solving_along_shortest_path(self):
        game = MazeGame(10, 12, seed=1)
        path = shortest_path(game.grid, game.start, game.goal)
        for action in actions_along(path):
            self.assertTrue(game.step(action))
        self.assertTrue(game.solved)
        self.assertEqual(game.moves, game.shortest_path_length())
        self.assertEqual(game.wall_hits, 0)
        finished = game.elapsed
        self.assertFalse(game.step(RIGHT))  # no moves after the goal
        self.assertEqual(game.elapsed, finished)  # timer is frozen

    def test_wall_blocks_but_turns(self):
        game = MazeGame(5, 5, seed=0)
        self.assertEqual(game.elapsed, 0.0)
        self.assertFalse(game.step(UP))  # the top border is always a wall
        self.assertEqual(game.pos, game.start)
        self.assertEqual(game.heading, UP)
        self.assertEqual(game.wall_hits, 1)
        self.assertGreater(game.elapsed, 0.0)  # timer started on the first step

    def test_new_maze_resets_and_changes(self):
        game = MazeGame(15, 15, seed=3)
        first = game.grid.copy()
        game.step(RIGHT)
        game.new_maze()
        self.assertEqual((game.pos, game.moves, game.wall_hits, game.solved), (game.start, 0, 0, False))
        self.assertEqual(game.elapsed, 0.0)
        self.assertFalse((game.grid == first).all())

    def test_seed_reproduces_sequence_and_single_maze(self):
        a, b = MazeGame(12, 12, seed=7), MazeGame(12, 12, seed=7)
        for _ in range(3):
            self.assertEqual(a.maze_seed, b.maze_seed)
            self.assertTrue((a.grid == b.grid).all())
            a.new_maze()
            b.new_maze()
        replay = MazeGame(12, 12)
        replay.new_maze(seed=a.maze_seed)
        self.assertTrue((replay.grid == a.grid).all())

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            MazeGame(1, 10)
        with self.assertRaises(ValueError):
            MazeGame(5, 5).step(4)


class MazeAppTest(unittest.TestCase):
    def setUp(self):
        pygame.init()

    def tearDown(self):
        pygame.quit()

    def test_play_solve_and_instant_retry(self):
        app = MazeApp(MazeGame(8, 10, seed=2))
        path = shortest_path(app.game.grid, app.game.start, app.game.goal)
        for action in actions_along(path):
            app.handle_key(ACTION_TO_KEY[action])
            app.draw()
        self.assertTrue(app.game.solved)
        self.assertEqual(app.solved_count, 1)
        self.assertEqual(app.last_time, app.best_time)
        self.assertEqual(app.last_shortest, len(path) - 1)
        seed = app.game.maze_seed
        app.handle_key(pygame.K_SPACE)
        self.assertFalse(app.game.solved)
        self.assertNotEqual(app.game.maze_seed, seed)
        app.draw()

    def test_space_does_not_restart_mid_run_but_r_does(self):
        app = MazeApp(MazeGame(8, 10, seed=2))
        seed = app.game.maze_seed
        app.handle_key(pygame.K_SPACE)
        self.assertEqual(app.game.maze_seed, seed)
        app.handle_key(pygame.K_r)
        self.assertNotEqual(app.game.maze_seed, seed)


if __name__ == "__main__":
    unittest.main()
