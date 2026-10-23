"""watch a trained agent: python watch_agent.py [--arch mlp|rnn|cfc] [--seed 0|1|2]."""
import argparse
from pathlib import Path

import torch
from torch.distributions import Categorical

from evaluate import load_model
from maze_game import MazeGame, eval_maze_seeds
import pygame


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", nargs="?")
    parser.add_argument("--arch", choices=["mlp", "rnn", "cfc"], default="cfc")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--eval-mazes", action="store_true", help="use the held-out evaluation mazes")
    parser.add_argument("--deterministic", action="store_true", help="take the most likely action")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = Path(__file__).resolve().parent / "runs" / "baseline" / f"{args.arch}_seed{args.seed}" / "model.pt"
        if not checkpoint.exists():
            checkpoint = Path(__file__).resolve().parent / "pretrained" / f"{args.arch}_seed{args.seed}.pt"
        if not checkpoint.exists():
            parser.error(f"no checkpoint for {args.arch} seed {args.seed}")

    arch, model = load_model(checkpoint)
    env = MazeGame(visualize=True)
    maze_seeds = eval_maze_seeds() if args.eval_mazes or args.checkpoint is None else [None] * args.episodes

    for episode in range(args.episodes):
        maze_seed = maze_seeds[episode % len(maze_seeds)]
        pygame.display.set_caption(f"{arch.upper()} | {Path(checkpoint).parent.name} | maze {maze_seed}")
        obs, _ = env.reset(seed=maze_seed)
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
