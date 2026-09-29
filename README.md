# Static vs. Liquid Neural Networks in a Maze

CPT_S 437 project — Cole Godfrey, Young Uk Kim, Kaiden Penney, Blake Turman.

**Research question:** Can biologically inspired artificial neurons improve self-operating motion
dynamics within a reinforcement learning paradigm? We compare a static neural network against a
Liquid Time-Constant (LTC) network on the same maze navigation task and parameter budget.

## Maze game

A simple 2D black-and-white Pygame maze. Every start and every retry generates a new random maze.
You are the black triangle (top-left); reach the gray tile (bottom-right).

```bash
pip install -r requirements.txt
python play.py                      # 30 x 40 cell maze
python play.py --rows 40 --cols 60  # bigger maze (tiles shrink to fit)
python play.py --seed 42            # same sequence of mazes every launch
```

| Key                  | Action                                  |
| -------------------- | --------------------------------------- |
| Arrow keys / WASD    | Move (hold to keep moving)              |
| R                    | New maze at any time                    |
| R / Enter / Space    | New maze after finishing (instant retry) |
| Esc                  | Quit                                    |

The timer starts on your first move and stops at the goal. The top bar shows the current time and
moves, plus your last time, best time, and number of mazes solved this session. The finish banner
also shows the shortest possible number of moves. The window title shows the current maze's seed.

### How mazes are built

Mazes are *perfect* (exactly one path between any two points), generated with a randomized
depth-first search, which gives long winding corridors and many dead ends. A maze of `rows x cols`
cells is stored as a `(2*rows + 1, 2*cols + 1)` boolean NumPy array where `True` is a wall; the
agent moves one tile per step, so going from one cell to the next takes two steps.

## Using the maze from code (for agents)

`maze_game.MazeGame` holds all game state and has no Pygame dependency, so training can run headless:

```python
from maze_game import MazeGame, UP, RIGHT, DOWN, LEFT

game = MazeGame(rows=30, cols=40, seed=0)  # seed -> reproducible sequence of mazes
game.grid        # np.ndarray[bool], True = wall
game.pos         # (row, col) of the agent in grid
game.goal        # (row, col) of the goal
game.heading     # last action attempted (the way the triangle points)

moved = game.step(RIGHT)   # False if a wall blocked the move
game.solved, game.moves, game.wall_hits, game.elapsed
game.shortest_path_length()  # optimal move count, e.g. for moves / optimal efficiency

game.new_maze()            # next maze in the sequence
game.new_maze(seed=1234)   # replay one specific maze
```

For a fair static-vs-LTC comparison, evaluate both models on the same list of maze seeds.

## Tests

```bash
python -m unittest
```
