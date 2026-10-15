"""
Recurrent PPO (actor-critic) training for the maze agents.

stable_baselines3's PPO feeds the policy one observation at a time and never
gives it a hidden state, so a recurrent network trained with it has no memory.
This trainer keeps each environment's hidden state across steps, stores the state
at the start of every training sequence, and trains with backpropagation through
time over those sequences.

Examples:
    python train_models.py --arch cfc --seed 0
    python train_models.py --arch mlp rnn cfc --seed 0 1 2 --timesteps 3000000
"""
import argparse
import csv
import json
import os
import time
from collections import deque

import numpy as np
import torch
from torch.distributions import Categorical

import config
from maze_game import MazeGame, eval_maze_keys, observation_size
from models import ARCHITECTURES, DEFAULT_PARAM_BUDGET, build_model, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", nargs="+", default=["mlp", "rnn", "cfc"], choices=ARCHITECTURES)
    parser.add_argument("--seed", nargs="+", type=int, default=[0])
    parser.add_argument("--timesteps", type=int, default=3_000_000, help="agent decisions per run")
    parser.add_argument("--param-budget", type=int, default=DEFAULT_PARAM_BUDGET)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=256, help="steps per env per update")
    parser.add_argument("--seq-len", type=int, default=64, help="BPTT sequence length")
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=1, help="torch CPU threads per run")
    parser.add_argument("--out-dir", default="runs")
    return parser.parse_args()


class EpisodeTracker:
    def __init__(self, window=100):
        self.returns = deque(maxlen=window)
        self.stats = deque(maxlen=window)
        self.count = 0

    def add(self, episode_return, stats):
        self.returns.append(episode_return)
        self.stats.append(stats)
        self.count += 1

    def summary(self):
        if not self.stats:
            return {}
        return {
            "episodes": self.count,
            "mean_return": float(np.mean(self.returns)),
            "success_rate": float(np.mean([s["success"] for s in self.stats])),
            "mean_ticks": float(np.mean([s["ticks"] for s in self.stats])),
            "mean_collision_ticks": float(np.mean([s["collision_ticks"] for s in self.stats])),
            "mean_cells_visited": float(np.mean([s["cells_visited"] for s in self.stats])),
        }


def train(arch, seed, args):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(args.threads)

    run_dir = os.path.join(args.out_dir, f"{arch}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    excluded = eval_maze_keys()
    envs = [
        MazeGame(seed=seed * 10_000 + i, excluded_mazes=excluded)
        for i in range(args.num_envs)
    ]
    obs_size, num_actions = observation_size(), 3

    model = build_model(arch, obs_size, num_actions, args.param_budget)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    print(f"[{arch} seed {seed}] hidden={model.hidden_size} params={count_parameters(model)}")

    n_envs, n_steps, seq_len = args.num_envs, args.rollout_steps, args.seq_len
    assert n_steps % seq_len == 0, "rollout-steps must be a multiple of seq-len"
    n_seqs = n_steps // seq_len
    num_updates = args.timesteps // (n_envs * n_steps)

    obs = torch.tensor(np.stack([env.reset()[0] for env in envs]))
    state = model.initial_state(n_envs)
    episode_starts = torch.ones(n_envs)
    episode_returns = np.zeros(n_envs)
    tracker = EpisodeTracker()

    log_path = os.path.join(run_dir, "progress.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = None
    start_time = time.time()

    for update in range(1, num_updates + 1):
        lr = args.lr * (1.0 - (update - 1) / num_updates)
        for group in optimizer.param_groups:
            group["lr"] = lr

        buf_obs = torch.zeros(n_steps, n_envs, obs_size)
        buf_starts = torch.zeros(n_steps, n_envs)
        buf_actions = torch.zeros(n_steps, n_envs, dtype=torch.long)
        buf_logprobs = torch.zeros(n_steps, n_envs)
        buf_values = torch.zeros(n_steps, n_envs)
        buf_rewards = torch.zeros(n_steps, n_envs)
        buf_seq_states = torch.zeros(n_seqs, n_envs, model.state_size)

        # ---- Collect experience ----
        model.eval()
        with torch.no_grad():
            for t in range(n_steps):
                if t % seq_len == 0:
                    buf_seq_states[t // seq_len] = state
                buf_obs[t] = obs
                buf_starts[t] = episode_starts

                logits, value, state = model(obs.unsqueeze(0), state, episode_starts.unsqueeze(0))
                dist = Categorical(logits=logits[0])
                action = dist.sample()
                buf_actions[t] = action
                buf_logprobs[t] = dist.log_prob(action)
                buf_values[t] = value[0]

                next_obs, next_starts = [], []
                truncated_envs, truncated_obs = [], []
                for i, env in enumerate(envs):
                    o, r, terminated, truncated, info = env.step(int(action[i]))
                    episode_returns[i] += r
                    buf_rewards[t, i] = r
                    if terminated or truncated:
                        tracker.add(episode_returns[i], info["episode_stats"])
                        episode_returns[i] = 0.0
                        if truncated:
                            # Time limit is not a real ending: bootstrap from the value of the last state.
                            truncated_envs.append(i)
                            truncated_obs.append(o)
                        o, _ = env.reset()
                    next_obs.append(o)
                    next_starts.append(float(terminated or truncated))

                if truncated_envs:
                    idx = torch.tensor(truncated_envs)
                    _, final_value, _ = model(
                        torch.tensor(np.stack(truncated_obs)).unsqueeze(0),
                        state[idx],
                        torch.zeros(1, len(truncated_envs)),
                    )
                    buf_rewards[t, idx] += args.gamma * final_value[0]

                obs = torch.tensor(np.stack(next_obs))
                episode_starts = torch.tensor(next_starts)

            _, last_value, _ = model(obs.unsqueeze(0), state, episode_starts.unsqueeze(0))
            last_value = last_value[0]

        # ---- Generalized advantage estimation ----
        advantages = torch.zeros(n_steps, n_envs)
        last_gae = torch.zeros(n_envs)
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_nonterminal = 1.0 - episode_starts
                next_value = last_value
            else:
                next_nonterminal = 1.0 - buf_starts[t + 1]
                next_value = buf_values[t + 1]
            delta = buf_rewards[t] + args.gamma * next_value * next_nonterminal - buf_values[t]
            last_gae = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + buf_values

        # ---- PPO update over sequences of length seq_len ----
        # (n_steps, n_envs, ...) -> (seq_len, n_seqs * n_envs, ...); sequence j*n_envs+i
        # is env i's j-th chunk and starts from buf_seq_states[j, i].
        def to_sequences(x):
            x = x.reshape(n_seqs, seq_len, n_envs, *x.shape[2:])
            return x.transpose(0, 1).reshape(seq_len, n_seqs * n_envs, *x.shape[3:])

        seq_obs = to_sequences(buf_obs)
        seq_starts = to_sequences(buf_starts)
        seq_actions = to_sequences(buf_actions)
        seq_logprobs = to_sequences(buf_logprobs)
        seq_advantages = to_sequences(advantages)
        seq_returns = to_sequences(returns)
        seq_init_states = buf_seq_states.reshape(n_seqs * n_envs, model.state_size)

        model.train()
        total_seqs = n_seqs * n_envs
        seqs_per_batch = max(1, total_seqs // args.num_minibatches)
        clip_fractions, approx_kls = [], []
        for _ in range(args.epochs):
            perm = torch.randperm(total_seqs)
            for start in range(0, total_seqs, seqs_per_batch):
                mb = perm[start:start + seqs_per_batch]
                logits, values, _ = model(seq_obs[:, mb], seq_init_states[mb], seq_starts[:, mb])
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(seq_actions[:, mb])
                entropy = dist.entropy().mean()

                mb_adv = seq_advantages[:, mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                log_ratio = new_logprobs - seq_logprobs[:, mb]
                ratio = log_ratio.exp()
                policy_loss = -torch.min(
                    ratio * mb_adv,
                    ratio.clamp(1 - args.clip, 1 + args.clip) * mb_adv,
                ).mean()
                value_loss = 0.5 * (values - seq_returns[:, mb]).pow(2).mean()
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kls.append(((ratio - 1) - log_ratio).mean().item())
                    clip_fractions.append(((ratio - 1).abs() > args.clip).float().mean().item())

        # ---- Logging ----
        timesteps = update * n_envs * n_steps
        row = {
            "timesteps": timesteps,
            "time_s": round(time.time() - start_time, 1),
            **tracker.summary(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "approx_kl": float(np.mean(approx_kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
        }
        if log_writer is None and "episodes" in row:
            log_writer = csv.DictWriter(log_file, fieldnames=list(row.keys()))
            log_writer.writeheader()
        if log_writer is not None:
            log_writer.writerow(row)
            log_file.flush()
        if update % 10 == 0 or update == num_updates:
            fps = timesteps / (time.time() - start_time)
            print(
                f"[{arch} seed {seed}] steps {timesteps:>9} | "
                f"return {row.get('mean_return', float('nan')):7.2f} | "
                f"success {row.get('success_rate', float('nan')):.2f} | "
                f"cells {row.get('mean_cells_visited', float('nan')):5.1f} | "
                f"entropy {row['entropy']:.3f} | {fps:.0f} steps/s",
                flush=True,
            )

    log_file.close()
    torch.save(
        {
            "arch": arch,
            "hidden_size": model.hidden_size,
            "obs_size": obs_size,
            "num_actions": num_actions,
            "num_params": count_parameters(model),
            "state_dict": model.state_dict(),
        },
        os.path.join(run_dir, "model.pt"),
    )
    with open(os.path.join(run_dir, "settings.json"), "w") as f:
        settings = {
            "arch": arch,
            "seed": seed,
            "args": vars(args),
            "config": {k: getattr(config, k) for k in dir(config) if k.isupper()},
        }
        json.dump(settings, f, indent=2)
    print(f"[{arch} seed {seed}] saved to {run_dir}")


if __name__ == "__main__":
    args = parse_args()
    for seed in args.seed:
        for arch in args.arch:
            train(arch, seed, args)
