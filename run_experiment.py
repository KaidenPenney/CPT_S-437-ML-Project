"""train and evaluate the nine baseline runs, then build a reproducible report.

    python run_experiment.py
    python run_experiment.py --resume

completed runs are reused only with --resume and matching saved settings.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
ARCHES = ("mlp", "rnn", "cfc")
SEEDS = (0, 1, 2)
SOURCE_FILES = ("config.py", "maze_game.py", "models.py", "train_models.py", "evaluate.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/baseline"))
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1 or args.timesteps < 8192:
        parser.error("jobs must be positive; timesteps must be at least 8192")
    os.chdir(ROOT)
    args.runs_dir = args.runs_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            parser.error(f"{manifest_path} exists; choose a new out-dir or use --resume")
        manifest = json.loads(manifest_path.read_text())
        if manifest["source_sha256"] != sources or manifest["requested_timesteps"] != args.timesteps:
            parser.error("saved experiment differs from current source or training budget")
    else:
        if any((args.runs_dir / f"{arch}_seed{seed}").exists() for arch in ARCHES for seed in SEEDS):
            parser.error("run directories already exist without a manifest; choose a new runs-dir")
        manifest = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": sources,
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "packages": {name: version(name) for name in ("torch", "ncps", "gymnasium", "pygame", "numpy", "pandas", "matplotlib")},
            "architectures": ARCHES,
            "training_seeds": SEEDS,
            "requested_timesteps": args.timesteps,
            "actual_timesteps": args.timesteps // 4096 * 4096,
            "evaluation_mazes": 200,
            "evaluation_action_seed": 0,
            "policies": ["stochastic", "deterministic"],
            "threads_per_run": 1,
            "parallel_jobs": args.jobs,
            "runs_dir": os.path.relpath(args.runs_dir, ROOT),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    # keep the implementation used for these results.
    source_dir = args.out_dir / "source"
    source_dir.mkdir(exist_ok=True)
    for name in SOURCE_FILES + ("run_experiment.py",):
        shutil.copy2(ROOT / name, source_dir / name)

    def run_one(arch, seed):
        name = f"{arch}_seed{seed}"
        run_dir = args.runs_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = run_dir / "model.pt"
        if checkpoint.exists() and args.resume:
            settings = json.loads((run_dir / "settings.json").read_text())
            if settings["args"]["timesteps"] != args.timesteps or settings["arch"] != arch or settings["seed"] != seed:
                raise ValueError(f"Saved settings mismatch: {name}")
        else:
            print(f"Training {name}", flush=True)
            command = [sys.executable, "-u", "train_models.py", "--arch", arch, "--seed", str(seed),
                       "--timesteps", str(args.timesteps), "--threads", "1", "--out-dir", str(args.runs_dir)]
            with (run_dir / "train.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Evaluating {name}", flush=True)
        command = [sys.executable, "-u", "evaluate.py", "--checkpoints", str(checkpoint),
                   "--policy", "both", "--threads", "1", "--out", str(run_dir / "eval_episodes.csv")]
        with (run_dir / "eval.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        archive = args.out_dir / "training" / name
        archive.mkdir(parents=True, exist_ok=True)
        for filename in ("progress.csv", "settings.json", "train.log", "eval.log"):
            shutil.copy2(run_dir / filename, archive / filename)
        print(f"Finished {name}", flush=True)

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, arch, seed) for seed in SEEDS for arch in ARCHES]
        for future in as_completed(futures):
            future.result()
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["orchestration_elapsed_s"] = round(time.monotonic() - start, 2)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    subprocess.run([sys.executable, "generate_results.py", "--runs-dir", str(args.runs_dir),
                    "--out-dir", str(args.out_dir)], check=True)
    print(f"Results: {args.out_dir / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
