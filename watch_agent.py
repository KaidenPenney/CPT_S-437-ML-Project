"""
Watch a trained agent drive through mazes (for qualitative smoothness checks).

Examples:
    python watch_agent.py runs/cfc_seed0/model.pt
    python watch_agent.py runs/mlp_seed0/model.pt --eval-mazes --deterministic
"""
import argparse

import torch
from torch.distributions import Categorical

from evaluate import load_model
from maze_game import MazeGame, eval_maze_seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--eval-mazes", action="store_true", help="use the held-out evaluation mazes")
    parser.add_argument("--deterministic", action="store_true", help="take the most likely action")
    args = parser.parse_args()

    arch, model = load_model(args.checkpoint)
    env = MazeGame(visualize=True)
    maze_seeds = eval_maze_seeds() if args.eval_mazes else [None] * args.episodes

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=maze_seeds[episode])
        state = model.initial_state(1)
        start = torch.ones(1, 1)
        done = False
        while not done:
            with torch.no_grad():
                logits, _, state = model(torch.tensor(obs).view(1, 1, -1), state, start)
            start = torch.zeros(1, 1)
            if args.deterministic:
                action = int(logits[0, 0].argmax())
            else:
                action = int(Categorical(logits=logits[0, 0]).sample())
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        print(f"{arch} episode {episode}: {'reached goal' if env.reached_goal else 'timed out'} "
              f"after {env.steps} ticks")


if __name__ == "__main__":
    main()
