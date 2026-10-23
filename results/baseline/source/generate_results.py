"""summarize the completed baseline experiment and export figures.

    python generate_results.py --runs-dir runs/baseline --out-dir results/baseline
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cpts437-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cpts437-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch

from evaluate import load_model
from maze_game import MazeGame, eval_maze_seeds, eval_maze_keys
import config

ARCHES = ("mlp", "rnn", "cfc")
COLORS = {"mlp": "#31688e", "rnn": "#bf6b33", "cfc": "#387855"}
METRICS = ("success", "spl", "ticks", "distance", "collisions", "cells_visited", "switch_rate", "turn_jerk")


def collect(runs_dir):
    """require three seeds and both policies on the complete shared maze set."""
    episodes, training = [], []
    expected_mazes = set(eval_maze_seeds())
    for arch in ARCHES:
        for seed in range(3):
            name = f"{arch}_seed{seed}"
            run_dir = runs_dir / name
            frame = pd.read_csv(run_dir / "eval_episodes.csv")
            assert set(frame["run"]) == {name}
            assert set(frame["arch"]) == {arch}
            assert set(frame["policy"]) == {"stochastic", "deterministic"}
            assert len(frame) == 2 * len(expected_mazes)
            for policy, group in frame.groupby("policy"):
                assert set(group["maze_seed"]) == expected_mazes, (name, policy)
                assert not group["maze_seed"].duplicated().any()
            assert frame["success"].isin([0, 1]).all()
            assert frame["spl"].between(0, 1).all()
            assert frame.loc[frame.success == 0, "spl"].eq(0).all()
            assert frame.loc[frame.success == 0, ["ticks", "distance"]].isna().all().all()
            assert frame.loc[frame.success == 1, ["ticks", "distance"]].notna().all().all()
            assert frame["switch_rate"].between(0, 1).all()
            assert frame["collisions"].between(0, config.EPISODE_LIMIT).all()
            frame["seed"] = seed
            episodes.append(frame)
            progress = pd.read_csv(run_dir / "progress.csv")
            settings = json.loads((run_dir / "settings.json").read_text())
            checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
            actual_steps = settings["args"]["timesteps"] // (settings["args"]["num_envs"] * settings["args"]["rollout_steps"])
            actual_steps *= settings["args"]["num_envs"] * settings["args"]["rollout_steps"]
            assert int(progress.iloc[-1].timesteps) == actual_steps
            assert settings["seed"] == seed and settings["arch"] == arch
            assert settings["config"] == json.loads(json.dumps({k: getattr(config, k) for k in dir(config) if k.isupper()}))
            training.append({"run": name, "arch": arch, "seed": seed,
                             "hidden_size": checkpoint["hidden_size"], "parameters": checkpoint["num_params"],
                             "timesteps": actual_steps, "training_seconds": float(progress.iloc[-1].time_s),
                             "checkpoint_sha256": hashlib.sha256((run_dir / "model.pt").read_bytes()).hexdigest()})
    return pd.concat(episodes, ignore_index=True), pd.DataFrame(training)


def summarize(episodes):
    """average mazes within each run before measuring variation across training seeds."""
    keys = ["policy", "arch", "seed", "run"]
    per_run = episodes.groupby(keys, sort=False)[list(METRICS)].mean().reset_index()
    counts = episodes.groupby(keys)["success"].agg(["sum", "count"]).reset_index()
    per_run = per_run.merge(counts.rename(columns={"sum": "successes", "count": "mazes"}), on=keys)
    summary = per_run.groupby(["policy", "arch"])[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return per_run, summary.reset_index()


def save_figure(fig, out_dir, name):
    fig.savefig(out_dir / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_training(runs_dir, out_dir):
    panels = [("success_rate", "Training completion (%)", 100), ("mean_return", "Episode return", 1),
              ("mean_collision_ticks", "Collision ticks / episode", 1), ("entropy", "Policy entropy (nats)", 1)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, (metric, label, scale) in zip(axes.flat, panels):
        for arch in ARCHES:
            frames = [pd.read_csv(runs_dir / f"{arch}_seed{seed}" / "progress.csv") for seed in range(3)]
            aligned = pd.concat([f.set_index("timesteps")[metric].rename(seed) for seed, f in enumerate(frames)], axis=1).dropna()
            mean, std = aligned.mean(axis=1) * scale, aligned.std(axis=1, ddof=1) * scale
            x = aligned.index.to_numpy() / 1e6
            for column in aligned:
                ax.plot(x, aligned[column] * scale, color=COLORS[arch], alpha=0.14, lw=0.6)
            ax.plot(x, mean, color=COLORS[arch], label=arch.upper(), lw=1.5)
            lower, upper = mean - std, mean + std
            if metric == "success_rate":
                lower, upper = lower.clip(lower=0), upper.clip(upper=100)
            ax.fill_between(x, lower, upper, color=COLORS[arch], alpha=0.12, lw=0)
        ax.set(xlabel="Training decisions (millions)", ylabel=label)
        if metric == "success_rate":
            ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, ncols=3)
    fig.suptitle("Training on fresh mazes | mean and sample SD across 3 seeds", fontsize=13)
    save_figure(fig, out_dir, "training_curves")


def plot_evaluation(per_run, out_dir):
    panels = [("success", "Completion (%)", 100), ("spl", "Success-weighted path efficiency", 1),
              ("collisions", "Collision ticks / episode", 1), ("ticks", "Completion ticks (successes only)", 1),
              ("switch_rate", "Steering switches (%)", 100), ("turn_jerk", "Mean turn-rate change (deg/tick)", 1)]
    for policy in ("stochastic", "deterministic"):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
        for ax, (metric, label, scale) in zip(axes.flat, panels):
            for i, arch in enumerate(ARCHES):
                values = per_run.loc[(per_run.policy == policy) & (per_run.arch == arch)].sort_values("seed")[metric] * scale
                ax.bar(i, values.mean(), yerr=values.std(ddof=1), color=COLORS[arch], alpha=0.65, width=0.6, capsize=4)
                ax.scatter(i + np.linspace(-0.12, 0.12, len(values)), values, c=COLORS[arch], edgecolor="white", linewidth=0.7, zorder=3)
            ax.set_xticks(range(3), [a.upper() for a in ARCHES])
            ax.set_ylabel(label)
            ax.set_ylim(bottom=0)
            if metric == "success":
                ax.set_ylim(0, 100)
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle(f"{policy.capitalize()} policy | 200 held-out mazes per seed\nBars: seed mean; error bars: sample SD; dots: training seeds 0, 1, 2", fontsize=12)
        save_figure(fig, out_dir, f"evaluation_{policy}")


@torch.no_grad()
def plot_trajectories(episodes, runs_dir, out_dir):
    """replay seed-zero deterministic agents on common easy and difficult examples."""
    base = episodes[(episodes.seed == 0) & (episodes.policy == "deterministic")]
    scores = base.groupby("maze_seed")["success"].sum().sort_index()
    easy = int(scores.idxmax())
    difficult = int(scores.drop(easy).idxmin())
    examples = [easy, difficult]
    fig, axes = plt.subplots(3, 2, figsize=(10, 11), layout="constrained")
    checks = []
    for row, arch in enumerate(ARCHES):
        _, model = load_model(runs_dir / f"{arch}_seed0" / "model.pt")
        for col, maze_seed in enumerate(examples):
            env = MazeGame()
            obs, _ = env.reset(seed=maze_seed)
            state, starts = model.initial_state(1), torch.ones(1, 1)
            trace = [(env.x, env.y)]
            while True:
                logits, _, state = model(torch.tensor(obs).view(1, 1, -1), state, starts)
                starts.zero_()
                obs, _, terminated, truncated, _ = env.step(int(logits[0, 0].argmax()))
                trace.append((env.x, env.y))
                if terminated or truncated:
                    break
            expected = base[(base.arch == arch) & (base.maze_seed == maze_seed)].iloc[0]
            assert int(env.reached_goal) == int(expected.success)
            assert env.collision_ticks == expected.collisions
            if env.reached_goal:
                assert env.steps == expected.ticks
            checks.append({"arch": arch, "maze_seed": maze_seed, "success": bool(env.reached_goal),
                           "ticks": env.steps, "collision_ticks": env.collision_ticks})
            ax = axes[row, col]
            for y, maze_row in enumerate(env.maze):
                for x, walls in enumerate(maze_row):
                    left, top = x * config.CELL_SIZE, y * config.CELL_SIZE
                    right, bottom = left + config.CELL_SIZE, top + config.CELL_SIZE
                    if walls[0]: ax.plot([left, right], [top, top], color="#555555", lw=0.9)
                    if walls[3]: ax.plot([left, left], [top, bottom], color="#555555", lw=0.9)
                    if x == config.MAZE_COLS - 1 and walls[1]: ax.plot([right, right], [top, bottom], color="#555555", lw=0.9)
                    if y == config.MAZE_ROWS - 1 and walls[2]: ax.plot([left, right], [bottom, bottom], color="#555555", lw=0.9)
            goal = env.goal_rect
            ax.add_patch(Rectangle((goal.x, goal.y), goal.width, goal.height, color="#d8e9d8"))
            points = np.asarray(trace)
            segments = np.stack([points[:-1], points[1:]], axis=1)
            collection = LineCollection(segments, cmap="viridis", linewidth=1.2, alpha=0.8)
            collection.set_array(np.linspace(0, 1, len(segments)))
            ax.add_collection(collection)
            ax.scatter(*trace[0], c="black", marker="o", s=20, zorder=4)
            ax.scatter(*trace[-1], c="#b23b32", marker="x", s=35, zorder=4)
            status = "completed" if env.reached_goal else "timed out"
            ax.set_title(f"{arch.upper()} | maze {maze_seed} | {status}\n{env.steps:,} ticks; {env.collision_ticks:,} collision ticks", fontsize=10)
            ax.set(xlim=(-10, config.SCREEN_WIDTH + 10), ylim=(config.SCREEN_HEIGHT + 10, -10), aspect="equal")
            ax.axis("off")
    fig.suptitle("Seed 0, deterministic replay | path color: early (purple) to late (yellow)\nCommon mazes with highest and lowest completion counts across architectures", fontsize=12)
    save_figure(fig, out_dir, "trajectories")
    (out_dir / "trajectory_examples.json").write_text(json.dumps(checks, indent=2) + "\n")
    return examples


def write_report(episodes, per_run, summary, training, manifest, examples, out_dir):
    def cell(row, metric, scale=1, digits=2):
        return f"{row[metric + '_mean'] * scale:.{digits}f} ± {row[metric + '_std'] * scale:.{digits}f}"

    lines = ["# Maze baseline results", "",
             f"MLP, RNN, and CfC trained with seeds 0, 1, and 2. Each run used {manifest['actual_timesteps']:,} agent decisions "
             f"({manifest['requested_timesteps']:,} requested; the trainer uses complete 4,096-decision rollouts).", "",
             f"Each checkpoint was evaluated on the same 200 distinct held-out mazes with both policies: {len(episodes):,} evaluation episodes total. "
             "Values are means ± sample standard deviations across the three training seeds. Maze metrics are averaged within each seed first.", ""]
    for policy in ("stochastic", "deterministic"):
        lines += [f"## {policy.capitalize()} policy", "", "| Model | Completion (%) | Path efficiency (SPL) | Collision ticks | Steering switches (%) | Turn-rate change |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for arch in ARCHES:
            row = summary[(summary.policy == policy) & (summary.arch == arch)].iloc[0]
            lines.append(f"| {arch.upper()} | {cell(row, 'success', 100, 1)} | {cell(row, 'spl', digits=3)} | {cell(row, 'collisions', digits=1)} | {cell(row, 'switch_rate', 100, 1)} | {cell(row, 'turn_jerk', digits=3)} |")
        lines += ["", f"![{policy} evaluation](evaluation_{policy}.png)", ""]
    lines += ["## Completion by training seed", "", "| Model | Seed | Sampled: completed / 200 | Deterministic: completed / 200 |", "| --- | ---: | ---: | ---: |"]
    for arch in ARCHES:
        for seed in range(3):
            subset = per_run[(per_run.arch == arch) & (per_run.seed == seed)].set_index("policy")
            lines.append(f"| {arch.upper()} | {seed} | {int(subset.loc['stochastic', 'successes'])} | {int(subset.loc['deterministic', 'successes'])} |")
    lines += ["", "## Completion time and distance", "", "Successful episodes only. Models can succeed on different subsets of mazes.", "",
              "| Policy | Model | Physics ticks | Distance (pixels) |", "| --- | --- | ---: | ---: |"]
    for policy in ("stochastic", "deterministic"):
        for arch in ARCHES:
            row = summary[(summary.policy == policy) & (summary.arch == arch)].iloc[0]
            lines.append(f"| {policy} | {arch.upper()} | {cell(row, 'ticks', digits=1)} | {cell(row, 'distance', digits=1)} |")
    lines += ["", "## Training", "", "| Model | Hidden units | Parameters | Mean training time (seconds) |", "| --- | ---: | ---: | ---: |"]
    for arch in ARCHES:
        group = training[training.arch == arch]
        lines.append(f"| {arch.upper()} | {int(group.iloc[0].hidden_size)} | {int(group.iloc[0].parameters):,} | {group.training_seconds.mean():.1f} |")
    lines += ["", "The parameter target was 6,000. Runs used one CPU thread each, with three jobs running concurrently.", "",
              "![Training curves](training_curves.png)", "",
              "Curves align by training step. Thin lines show individual seeds; bands show sample standard deviations. "
              "Training episode metrics use the trainer's rolling window of 100 completed episodes.", "", "## Observations", ""]
    for policy in ("stochastic", "deterministic"):
        group = summary[summary.policy == policy].set_index("arch")
        difference = 100 * (group.loc['cfc', 'success_mean'] - group.loc['rnn', 'success_mean'])
        lines.append(f"CfC minus RNN mean completion: {difference:+.1f} percentage points with the {policy} policy.")
        lines.append("")
    lines += ["Three training seeds provide a baseline comparison; they do not establish a general architecture advantage. "
              "Low switching or turn-rate change can also describe an agent that circles or gets stuck, so read those metrics alongside completion.", "",
              "## Example trajectories", "", f"Deterministic seed-0 replays on mazes {examples[0]} and {examples[1]}. "
              "These are the first mazes with the highest and lowest seed-0 completion counts across the three architectures. "
              "All six replays were checked against the evaluation outcomes and collision counts.", "",
              "![Example trajectories](trajectories.png)", "", "## Metric definitions", "",
              "Completion is reaching the goal within 6,000 physics ticks. Collision ticks count time touching walls across all episodes. "
              "SPL is zero on failure and otherwise uses the environment's estimated doorway path to the goal center divided by traveled distance, capped at one. "
              "The goal region is reached before its center, so this is an approximate path-efficiency measure. "
              "Steering switches count action changes per decision transition. The raw `turn_jerk` column measures mean absolute turn-rate change between decisions in degrees per tick; it is a smoothness proxy.", "",
              "## Reproduction", "", "```sh", ".venv/bin/python run_experiment.py --resume",
              ".venv/bin/python generate_results.py --runs-dir runs/baseline --out-dir results/baseline", "```", "",
              "The manifest records package versions, source hashes, and the base commit. `source/` contains the experiment code. "
              "`training/` contains settings, logs, and learning curves for every run. "
              "Checkpoints are in `runs/baseline/<model>_seed<seed>/model.pt`; their hashes are in `training_runs.csv`. "
              "Training excludes all 200 evaluation layouts. Sampled evaluation uses action RNG seed 0 and one episode per maze per checkpoint. "
              "The experiment used the repository's existing training settings without tuning against these results.", "",
              "Raw data: [episodes](eval_episodes.csv), [per-run metrics](per_run_metrics.csv), [summary](summary.csv), "
              "[training runs](training_runs.csv), [manifest](manifest.json).", ""]
    (out_dir / "RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/baseline"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    manifest = json.loads((args.out_dir / "manifest.json").read_text())
    for name, expected_hash in manifest["source_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected_hash, f"source changed: {name}"
    assert len(eval_maze_keys()) == 200
    episodes, training = collect(args.runs_dir)
    assert set(training.timesteps) == {manifest["actual_timesteps"]}
    per_run, summary = summarize(episodes)
    episodes.to_csv(args.out_dir / "eval_episodes.csv", index=False)
    training.to_csv(args.out_dir / "training_runs.csv", index=False)
    per_run.to_csv(args.out_dir / "per_run_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    plot_training(args.runs_dir, args.out_dir)
    plot_evaluation(per_run, args.out_dir)
    examples = plot_trajectories(episodes, args.runs_dir, args.out_dir)
    write_report(episodes, per_run, summary, training, manifest, examples, args.out_dir)
    for name in ("run_experiment.py", "generate_results.py"):
        shutil.copy2(name, args.out_dir / "source" / name)
    print(summary[["policy", "arch", "success_mean", "success_std"]].to_string(index=False))
    print(f"Validated {len(training)} runs and {len(episodes)} evaluation episodes.")


if __name__ == "__main__":
    main()
