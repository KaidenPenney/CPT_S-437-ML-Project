# Liquid vs. Static Networks in a Maze (CPT_S 437)

Cole Godfrey, Young Uk Kim, Kaiden Penney, Blake Turman

An agent always moves forward through a procedurally generated maze and only
chooses to steer left, go straight, or steer right. It sees the maze through
distance sensors. We train a Closed-form Continuous-time liquid network (CfC)
and two baselines with the same parameter budget, a feedforward MLP and a
vanilla RNN, using recurrent PPO (actor-critic), then compare them on mazes
they never saw during training.

## Setup

```
pip install -r requirements.txt
```

## Files

| File | Purpose |
| --- | --- |
| `config.py` | Game physics, sensors, rewards, and evaluation-set settings |
| `maze_game.py` | Gymnasium environment (new random maze every episode) |
| `models.py` | MLP, RNN, CfC (plus optional GRU/LSTM) actor-critics, parameter-matched |
| `train_models.py` | Recurrent PPO trainer |
| `evaluate.py` | Metrics on the held-out evaluation mazes |
| `plot_results.py` | Training curves, mean ± std over seeds |
| `watch_agent.py` | Watch a trained agent play |
| `play_maze.py` | Play the maze yourself with the arrow keys |

## Running the experiment

```
# Train all three models on 3 seeds (each run saves to runs/<arch>_seed<k>/)
python train_models.py --arch mlp rnn cfc --seed 0 1 2 --timesteps 3000000

# Held-out evaluation and training curves
python evaluate.py
python plot_results.py --metric success_rate

# Watch one
python watch_agent.py runs/cfc_seed0/model.pt --eval-mazes
```

Each run uses one CPU thread, so on a multi-core machine you can start several
in separate terminals, for example one per `--seed`. On our test machine a 3M-step
run took about 4 minutes (MLP), 6 minutes (RNN) and 9 minutes (CfC).

## Design notes

- **Observations** (13 values): 7 ray-cast distance sensors (-90° to +90°), forward
  and sideways speed, turn rate, compass heading (cos, sin), and a bump sensor.
  Set `OBS_INCLUDE_POSITION = True` in `config.py` to also give the absolute
  position and the vector to the goal.
- **Memory**: the hidden state is carried across agent decisions and reset at the
  start of each episode. Training uses backpropagation through time over 64-step
  sequences.
- **Equal parameters**: each model's hidden size is chosen automatically so the
  total parameter count is about `--param-budget` (default 6000).
- **Unseen mazes**: the 200 evaluation mazes come from fixed seeds, and the
  training environments reject any maze with the same layout.
- **Reward**: progress along the maze's shortest path to the goal, +10 for reaching it,
  small penalties per step and for touching walls. The progress term is
  potential-based shaping, so it does not change which policy is optimal.
