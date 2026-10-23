"""
Plot training curves (mean +/- std over seeds) from runs/<arch>_seed<k>/progress.csv.

Example:
    python plot_results.py --runs-dir runs --metric success_rate
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--metric", default="success_rate",
                        help="column of progress.csv, e.g. success_rate, mean_return, mean_collision_ticks")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    curves = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(args.runs_dir, "*_seed*", "progress.csv"))):
        arch = re.match(r"(.+)_seed\d+", os.path.basename(os.path.dirname(path))).group(1)
        curves[arch].append(pd.read_csv(path)[["timesteps", args.metric]])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arch, frames in sorted(curves.items()):
        # align seeds by training step; logging can start at different updates.
        aligned = pd.concat([f.set_index("timesteps")[args.metric] for f in frames], axis=1).dropna()
        steps = aligned.index.to_numpy()
        values = aligned.to_numpy().T
        mean = values.mean(0)
        std = values.std(0, ddof=1) if len(frames) > 1 else np.zeros_like(mean)
        ax.plot(steps, mean, label=f"{arch} (n={len(frames)})")
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("agent decisions")
    ax.set_ylabel(args.metric)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = args.out or os.path.join(args.runs_dir, f"{args.metric}.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
