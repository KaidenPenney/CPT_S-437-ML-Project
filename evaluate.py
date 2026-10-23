"""
Evaluate trained agents on the held-out mazes (never generated during training).

For every runs/<arch>_seed<k>/model.pt, runs one episode per evaluation maze and
reports, per architecture (mean +/- std over training seeds):
  success      fraction of mazes completed within the time limit
  spl          success weighted by path length (shortest path / path taken, 0 on failure)
  ticks        physics ticks to reach the goal (successful episodes only)
  distance     pixels traveled (successful episodes only)
  collisions   physics ticks spent touching a wall
  switch_rate  fraction of decisions where the steering action changed (lower = smoother)
  turn_jerk    mean |change in angular velocity| between decisions, deg/tick (lower = smoother)

Examples:
    python evaluate.py
    python evaluate.py --runs-dir runs --policy deterministic
"""
import argparse
import csv
import glob
import os
import warnings
from collections import defaultdict

import numpy as np
import torch
from torch.distributions import Categorical

from maze_game import MazeGame, eval_maze_seeds
from models import build_model


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu")
    model = build_model(
        checkpoint["arch"], checkpoint["obs_size"], checkpoint["num_actions"],
        hidden_size=checkpoint["hidden_size"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return checkpoint["arch"], model


@torch.no_grad()
def evaluate_model(model, maze_seeds, deterministic, torch_seed=0):
    """Runs all mazes in one batch. Returns a list of per-episode metric dicts."""
    torch.manual_seed(torch_seed)
    envs = [MazeGame() for _ in maze_seeds]
    obs = torch.tensor(np.stack([env.reset(seed=s)[0] for env, s in zip(envs, maze_seeds)]))
    shortest = [env.goal_distance() for env in envs]
    n = len(envs)
    state = model.initial_state(n)
    starts = torch.ones(n)
    active = np.ones(n, dtype=bool)
    previous_action = np.full(n, -1)
    previous_turn = np.zeros(n)
    switches = np.zeros(n)
    jerk = np.zeros(n)
    decisions = np.zeros(n)
    results = [None] * n

    while active.any():
        logits, _, state = model(obs.unsqueeze(0), state, starts.unsqueeze(0))
        starts = torch.zeros(n)
        if deterministic:
            actions = logits[0].argmax(-1)
        else:
            actions = Categorical(logits=logits[0]).sample()

        next_obs = obs.clone()
        for i in np.flatnonzero(active):
            action = int(actions[i])
            if previous_action[i] >= 0:
                switches[i] += action != previous_action[i]
            previous_action[i] = action
            o, _, terminated, truncated, info = envs[i].step(action)
            decisions[i] += 1
            jerk[i] += abs(envs[i].angular_velocity - previous_turn[i])
            previous_turn[i] = envs[i].angular_velocity
            next_obs[i] = torch.tensor(o)
            if terminated or truncated:
                stats = info["episode_stats"]
                success = stats["success"] > 0
                results[i] = {
                    "maze_seed": maze_seeds[i],
                    "success": float(success),
                    "spl": shortest[i] / max(shortest[i], stats["distance_traveled"]) if success else 0.0,
                    "ticks": stats["ticks"] if success else np.nan,
                    "distance": stats["distance_traveled"] if success else np.nan,
                    "collisions": stats["collision_ticks"],
                    "cells_visited": stats["cells_visited"],
                    "switch_rate": switches[i] / max(1, decisions[i] - 1),
                    "turn_jerk": jerk[i] / decisions[i],
                }
                active[i] = False
        obs = next_obs
    return results


METRICS = ("success", "spl", "ticks", "distance", "collisions", "switch_rate", "turn_jerk")


def main():
    # nanmean of a metric with no successful episodes is nan; that is expected.
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="Degrees of freedom")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--checkpoints", nargs="+", help="explicit checkpoint paths, including pretrained/*.pt")
    parser.add_argument("--threads", type=int, default=1, help="torch CPU threads")
    parser.add_argument("--policy", choices=["stochastic", "deterministic", "both"], default="both")
    parser.add_argument("--num-mazes", type=int, default=None, help="use only the first N eval mazes")
    parser.add_argument("--out", default=None, help="per-episode CSV (default: <runs-dir>/eval_episodes.csv)")
    args = parser.parse_args()
    if args.threads < 1 or (args.num_mazes is not None and args.num_mazes < 1):
        parser.error("threads and num-mazes must be positive")
    torch.set_num_threads(args.threads)

    maze_seeds = eval_maze_seeds()[:args.num_mazes]
    policies = ["stochastic", "deterministic"] if args.policy == "both" else [args.policy]
    checkpoints = args.checkpoints or sorted(glob.glob(os.path.join(args.runs_dir, "*", "model.pt")))
    if not checkpoints:
        raise SystemExit(f"No checkpoints found under {args.runs_dir}/*/model.pt")

    rows = []
    # per_run[(policy, arch)] -> list of per-run mean metric dicts (one per training seed)
    per_run = defaultdict(list)
    for path in checkpoints:
        run_name = (os.path.basename(os.path.dirname(path)) if os.path.basename(path) == "model.pt"
                    else os.path.splitext(os.path.basename(path))[0])
        arch, model = load_model(path)
        for policy in policies:
            episodes = evaluate_model(model, maze_seeds, deterministic=(policy == "deterministic"))
            for episode in episodes:
                rows.append({"run": run_name, "arch": arch, "policy": policy, **episode})
            run_means = {m: np.nanmean([e[m] for e in episodes]) for m in METRICS}
            per_run[(policy, arch)].append(run_means)
            print(f"{run_name:>16} {policy:>13}: success {run_means['success']:.3f}  spl {run_means['spl']:.3f}")

    out_path = args.out or os.path.join(args.runs_dir, "eval_episodes.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(maze_seeds)} held-out mazes; mean +/- std over training seeds")
    header = f"{'policy':>13} {'arch':>5} {'seeds':>5} " + " ".join(f"{m:>17}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for (policy, arch), runs in sorted(per_run.items()):
        cells = []
        for m in METRICS:
            values = np.array([r[m] for r in runs])
            cells.append(f"{np.nanmean(values):9.3f} +/- {np.nanstd(values):5.3f}")
        print(f"{policy:>13} {arch:>5} {len(runs):>5} " + " ".join(f"{c:>17}" for c in cells))
    print(f"\nPer-episode results written to {out_path}")


if __name__ == "__main__":
    main()
