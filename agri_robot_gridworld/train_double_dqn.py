# train_double_dqn.py
import os
import time
import csv
import argparse
import shutil
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from config import make_config
from agri_env import AgriRobotEnv, Action
from dqn_double import DoubleDQNAgent, DQNConfig


def _action_group(name: str) -> str:
    # Group directional actions like MOVE_N/MOVE_S under MOVE, etc.
    # Also handles e.g. REFILL_WEED -> REFILL and DROPOFF_PRODUCE -> DROPOFF.
    return str(name).split("_", 1)[0]


def save_action_bars_png(
    out_path: str,
    *,
    labels: List[str],
    counts: np.ndarray,
    title: str,
    as_percent: bool = True,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    total = float(np.sum(counts))
    if total <= 0.0:
        total = 1.0

    if bool(as_percent):
        y = (counts.astype(np.float64) / total) * 100.0
        ylabel = "Percent of actions"
    else:
        y = counts.astype(np.float64)
        ylabel = "Count"

    plt.figure(figsize=(8.0, 3.6))
    x = np.arange(len(labels))
    plt.bar(x, y)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def save_action_group_lines_png(
    out_path: str,
    *,
    episodes: List[int],
    labels: List[str],
    fracs: List[np.ndarray],
    title: str,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if len(episodes) == 0 or len(fracs) == 0:
        return

    y = np.stack(fracs).astype(np.float64, copy=False) * 100.0
    x = np.array(episodes, dtype=np.int32)

    plt.figure(figsize=(9.5, 4.5))
    for j, lab in enumerate(labels):
        plt.plot(x, y[:, j], linewidth=2.0, label=lab)
    plt.xlabel("Episode")
    plt.ylabel("Percent of actions")
    plt.title(title)
    plt.grid(True, alpha=0.2)
    plt.legend(ncol=4, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def _env_snapshot(env: AgriRobotEnv) -> Dict[str, object]:
    return {
        "robot_x": int(env.robot_x),
        "robot_y": int(env.robot_y),
        "day": int(env.day),
        "actions_left": int(env.actions_left_today),
        "money": float(env.money),
        "produce_count": int(env.produce_count),
        "pending": float(env.produce_value_pending),
        "weed_tank": int(env.weed_tank),
        "fert_tank": int(env.fert_tank),
        "water_tank": int(env.water_tank),
        "disabled": int(env.disabled_today),
        "terrain": env.terrain.copy(),
        "saturation": env.saturation.copy(),
        "plant_stage": env.plant_stage.copy(),
        "weed": env.weed.copy(),
        "fertilized": env.fertilized.copy(),
    }


def record_greedy_episode(
    env: AgriRobotEnv,
    agent: DoubleDQNAgent,
    *,
    seed: int,
    max_steps: int,
) -> Tuple[Dict[str, np.ndarray], float]:
    """Record one greedy episode into the NPZ replay format consumed by main.py."""
    obs, _ = env.reset(seed=int(seed))
    done = False
    steps = 0

    actions: List[int] = []
    rewards: List[float] = []
    snaps: List[Dict[str, object]] = []

    while (not done) and steps < int(max_steps):
        mask = env.valid_action_mask()
        a = int(agent.act(obs, mask=mask, greedy=True))
        obs, r, term, trunc, info = env.step(a)
        done = bool(term or trunc)
        actions.append(int(a))
        rewards.append(float(r))
        snaps.append(_env_snapshot(env))
        steps += 1

    out: Dict[str, np.ndarray] = {
        "action": np.array(actions, dtype=np.int32),
        "reward": np.array(rewards, dtype=np.float32),
        "robot_x": np.array([s["robot_x"] for s in snaps], dtype=np.int32),
        "robot_y": np.array([s["robot_y"] for s in snaps], dtype=np.int32),
        "day": np.array([s["day"] for s in snaps], dtype=np.int32),
        "actions_left": np.array([s["actions_left"] for s in snaps], dtype=np.int32),
        "money": np.array([s["money"] for s in snaps], dtype=np.float32),
        "produce_count": np.array([s["produce_count"] for s in snaps], dtype=np.int32),
        "pending": np.array([s["pending"] for s in snaps], dtype=np.float32),
        "weed_tank": np.array([s["weed_tank"] for s in snaps], dtype=np.int32),
        "fert_tank": np.array([s["fert_tank"] for s in snaps], dtype=np.int32),
        "water_tank": np.array([s["water_tank"] for s in snaps], dtype=np.int32),
        "disabled": np.array([s["disabled"] for s in snaps], dtype=np.int32),
        "terrain": np.stack([s["terrain"] for s in snaps]).astype(np.int32, copy=False),
        "saturation": np.stack([s["saturation"] for s in snaps]).astype(np.float32, copy=False),
        "plant_stage": np.stack([s["plant_stage"] for s in snaps]).astype(np.int32, copy=False),
        "weed": np.stack([s["weed"] for s in snaps]).astype(np.int8, copy=False),
        "fertilized": np.stack([s["fertilized"] for s in snaps]).astype(np.int8, copy=False),
    }

    final_money = float(info.get("final_money", env.money))
    return out, final_money


def moving_avg(x: List[float], window: int) -> np.ndarray:
    if len(x) < window:
        return np.array([], dtype=np.float32)
    arr = np.array(x, dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(arr, kernel, mode="valid")


def rolling_mean_std(x: List[float], window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (mean, std) computed over a rolling window (valid mode)."""
    if window <= 1:
        arr = np.array(x, dtype=np.float32)
        return arr, np.zeros_like(arr)
    if len(x) < window:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    arr = np.array(x, dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    mean = np.convolve(arr, kernel, mode="valid")
    mean2 = np.convolve(arr * arr, kernel, mode="valid")
    var = np.maximum(mean2 - mean * mean, 0.0)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def save_metrics(run_dir: str, metrics: Dict[str, List[float]], *, plots: bool = True):
    os.makedirs(run_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(run_dir, "metrics.csv")
    keys = list(metrics.keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode"] + keys)
        for i in range(len(next(iter(metrics.values())))):
            w.writerow([i + 1] + [metrics[k][i] for k in keys])

    # NPZ
    np.savez(os.path.join(run_dir, "metrics.npz"), **{k: np.array(v) for k, v in metrics.items()})

    if not plots:
        print(f"[saved] metrics (no plots) -> {run_dir}")
        return

    write_plots(run_dir, metrics)
    print(f"[saved] metrics + plots -> {run_dir}")


def write_plots(run_dir: str, metrics: Dict[str, List[float]]):
    """Write PNG plots for an already-saved metrics dict (does not write CSV/NPZ)."""

    def plot_series(name: str, y: List[float], ylabel: str, window: int = 50, title: str | None = None):
        plt.figure()
        mean, std = rolling_mean_std(y, window)
        if len(mean) > 0:
            x = np.arange(window, window + len(mean))
            plt.plot(x, mean, linewidth=2.0, label="mean")
            plt.fill_between(x, mean - std, mean + std, alpha=0.20, linewidth=0, label="±1 std")
        else:
            # Too short for rolling stats; fall back to raw.
            x = np.arange(1, len(y) + 1)
            plt.plot(x, y, linewidth=2.0, label="value")
        plt.title(title or name)
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"{name}.png"), dpi=220)
        plt.close()

    plot_series("episode_reward", metrics["episode_reward"], "Episode reward", window=50, title="Training Reward")
    plot_series("final_money", metrics["final_money"], "Final money", window=50, title="Final Money")
    plot_series("loss", metrics["loss"], "TD loss (Huber)", window=20, title="Training Loss")
    plot_series("epsilon", metrics["epsilon"], "Epsilon", window=1, title="Exploration Schedule")
    plot_series("invalid_rate", metrics["invalid_rate"], "Invalid action rate", window=20, title="Invalid Action Rate")
    plot_series(
        "missed_charger_days",
        metrics["missed_charger_days"],
        "Missed charger days (episode total)",
        window=20,
        title="Missed Charger Days",
    )

    # Optional: sparse greedy-eval series (NaN for episodes without eval)
    if "eval_final_money_mean" in metrics:
        y = np.array(metrics["eval_final_money_mean"], dtype=np.float32)
        x = np.arange(1, len(y) + 1)
        ok = np.isfinite(y)
        if np.any(ok):
            plt.figure()
            plt.plot(x[ok], y[ok], marker="o", linewidth=1.5, label="eval_final_money_mean")
            if "eval_final_money_std" in metrics:
                s = np.array(metrics["eval_final_money_std"], dtype=np.float32)
                if s.shape == y.shape and np.any(np.isfinite(s)):
                    s_ok = np.isfinite(s) & ok
                    plt.fill_between(x[s_ok], (y - s)[s_ok], (y + s)[s_ok], alpha=0.2, linewidth=0)
            plt.xlabel("Episode")
            plt.ylabel("Final money (greedy eval)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "eval_final_money.png"), dpi=200)
            plt.close()


def replot_run(run_dir: str):
    """Regenerate per-experiment and comparison plots for a previously completed run."""
    if not os.path.exists(run_dir):
        candidate = os.path.join("runs", run_dir)
        if os.path.exists(candidate):
            run_dir = candidate
        else:
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

    all_metrics: Dict[str, Dict[str, List[float]]] = {}

    for name in sorted(os.listdir(run_dir)):
        exp_dir = os.path.join(run_dir, name)
        if not os.path.isdir(exp_dir):
            continue
        npz_path = os.path.join(exp_dir, "metrics.npz")
        if not os.path.exists(npz_path):
            continue

        data = np.load(npz_path, allow_pickle=False)
        metrics: Dict[str, List[float]] = {k: data[k].astype(np.float32).tolist() for k in data.files}
        all_metrics[name] = metrics

        write_plots(exp_dir, metrics)
        print(f"[replot] {name} -> {exp_dir}")

    if len(all_metrics) >= 2:
        save_comparison(run_dir, all_metrics)
        print(f"[replot] comparison -> {run_dir}")
    elif len(all_metrics) == 1:
        print("[replot] only one experiment found; skipping comparison plots")
    else:
        print("[replot] no experiments with metrics.npz found")


def save_comparison(parent_dir: str, all_runs: Dict[str, Dict[str, List[float]]]):
    os.makedirs(parent_dir, exist_ok=True)

    display_name = {
        "singledqn": "Single DQN",
        "baseline": "Double DQN",
        "nstep3": "Double DQN (3-step)",
        "nstep5": "Double DQN (5-step)",
        "dueling": "Double DQN (Dueling)",
        "both": "Double DQN (Dueling + 5-step)",
        "custom": "Custom",
        "nstep": "Double DQN (n-step)",
    }

    # Prefer a stable legend order: Single DQN first, then Double DQN.
    preferred_order = ["singledqn", "baseline", "nstep3", "nstep5", "dueling", "both"]
    run_names = [n for n in preferred_order if n in all_runs]
    for n in all_runs.keys():
        if n not in run_names:
            run_names.append(n)

    def overlay(metric: str, ylabel: str, window: int, title: str):
        plt.figure()
        for name in run_names:
            metrics = all_runs[name]
            y = metrics[metric]
            mean, std = rolling_mean_std(y, window)
            label = display_name.get(name, name)
            if len(mean) > 0:
                x = np.arange(window, window + len(mean))
                plt.plot(x, mean, linewidth=2.0, label=label)
                plt.fill_between(x, mean - std, mean + std, alpha=0.15, linewidth=0)
            else:
                x = np.arange(1, len(y) + 1)
                plt.plot(x, y, linewidth=2.0, label=label)
        plt.title(title)
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(parent_dir, f"compare_{metric}.png"), dpi=240)
        plt.close()

    overlay("episode_reward", "Episode reward", window=50, title="Training Reward (mean ± 1 std)")
    overlay("final_money", "Final money", window=50, title="Final Money (mean ± 1 std)")
    overlay("invalid_rate", "Invalid action rate", window=20, title="Invalid Action Rate (mean ± 1 std)")
    overlay("missed_charger_days", "Missed charger days (episode total)", window=20, title="Missed Charger Days (mean ± 1 std)")
    overlay("loss", "TD loss (Huber)", window=20, title="Training Loss (mean ± 1 std)")

    lines = []
    lines.append("Comparison summary (last 50-episode mean):\n")
    for name, metrics in all_runs.items():
        last = min(50, len(metrics["final_money"]))
        mean_money = float(np.mean(metrics["final_money"][-last:]))
        mean_reward = float(np.mean(metrics["episode_reward"][-last:]))
        mean_invalid = float(np.mean(metrics["invalid_rate"][-last:]))
        mean_missed = float(np.mean(metrics["missed_charger_days"][-last:]))
        best_money = float(np.max(metrics["final_money"]))
        lines.append(
            f"- {name:>10s} | meanMoney={mean_money:8.2f} | bestMoney={best_money:8.2f} | "
            f"meanR={mean_reward:8.2f} | invalid={mean_invalid*100:5.2f}% | missedChg={mean_missed:6.2f}"
        )
    with open(os.path.join(parent_dir, "compare_summary.txt"), "w") as f:
        f.write("\n".join(lines))


def run_one_experiment(
    exp_dir: str,
    exp_name: str,
    *,
    episodes: int,
    seed: int,
    device: str,
    dqn_cfg: DQNConfig,
    env_cfg_overrides: Dict[str, object],
    print_every: int = 50,
    save_every: int = 100,
    eval_every: int = 0,
    eval_episodes: int = 20,
    eval_seed: int = 12345,
    plots_at_checkpoints: bool = False,
    reward_scale: float = 1.0,
    reward_clip: float = 0.0,
    action_hist_every: int = 100,
    action_hist_window: int = 100,
    action_hist_plot: str = "lines",
    save_best_replay: bool = False,
    best_replay_dir: str = "replays",
    best_replay_eps_threshold: float = 0.3,
    # Legacy (heavier): sample many greedy episodes to pick best/worst/avg.
    save_replays: bool = False,
    replay_dir: str = "replays",
    replay_episodes: int = 25,
    replay_seed: int = 4242,
) -> Tuple[str, Dict[str, List[float]]]:
    cfg = make_config()
    for k, v in env_cfg_overrides.items():
        setattr(cfg, k, v)

    env = AgriRobotEnv(cfg, seed=seed)
    obs_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    agent = DoubleDQNAgent(obs_dim, num_actions, dqn_cfg, device=device)

    gate_after_curriculum = bool(getattr(env.cfg, "ENABLE_MATURITY_GATE", True))
    n_cur = int(getattr(env.cfg, "MATURITY_CURRICULUM_EPISODES", 0) or 0)

    steps_per_ep = int(getattr(env.cfg, "NUM_DAYS", 30)) * int(getattr(env.cfg, "BATTERY_ACTIONS_PER_DAY", 40))

    # Best-replay tracking (saves only when epsilon is low enough and we beat prior best)
    best_replay_money = -1e18
    best_replay_path = ""
    replay_env = None

    def maybe_save_best_replay(ep_seed: int, eps: float, final_money: float):
        nonlocal best_replay_money, best_replay_path, replay_env
        if not bool(save_best_replay):
            return
        if float(eps) >= float(best_replay_eps_threshold):
            return

        # Build a replay env once, using the final (post-curriculum) env behavior.
        if replay_env is None:
            cfg_replay = make_config()
            for k, v in env_cfg_overrides.items():
                setattr(cfg_replay, k, v)
            cfg_replay.ENABLE_MATURITY_GATE = gate_after_curriculum
            replay_env = AgriRobotEnv(cfg_replay, seed=int(ep_seed))

        was_training = agent.q_online.training
        agent.q_online.eval()
        try:
            data, fm = record_greedy_episode(
                replay_env,
                agent,
                seed=int(ep_seed),
                max_steps=int(steps_per_ep) + 5,
            )
        finally:
            if was_training:
                agent.q_online.train()

        # Only save if the greedy replay is actually the best so far.
        if float(fm) <= float(best_replay_money) + 1e-9:
            return

        best_replay_money = float(fm)
        os.makedirs(str(best_replay_dir), exist_ok=True)
        run_id = os.path.basename(os.path.dirname(exp_dir))
        best_replay_path = os.path.join(str(best_replay_dir), f"{run_id}_{exp_name}_best.npz")
        np.savez(best_replay_path, allow_pickle=False, **data)
        print(f"[saved] best replay -> {best_replay_path} | money={best_replay_money:.2f} | eps={eps:.3f}")

    def greedy_evaluate() -> Dict[str, float]:
        # Evaluate the current policy greedily (no exploration).
        cfg_eval = make_config()
        for k, v in env_cfg_overrides.items():
            setattr(cfg_eval, k, v)

        # During eval we want the "final" environment behavior (post curriculum)
        cfg_eval.ENABLE_MATURITY_GATE = gate_after_curriculum

        eval_env = AgriRobotEnv(cfg_eval, seed=eval_seed)
        final_money = []
        ep_reward = []
        missed = []
        invalid_rates = []

        was_training = agent.q_online.training
        agent.q_online.eval()
        try:
            for i in range(int(eval_episodes)):
                obs, info = eval_env.reset(seed=int(eval_seed + 1000 + i))
                done = False
                R = 0.0
                steps = 0
                invalid = 0
                while not done:
                    mask = eval_env.valid_action_mask()
                    a = agent.act(obs, mask=mask, greedy=True)
                    obs, r, term, trunc, info = eval_env.step(int(a))
                    done = bool(term or trunc)
                    R += float(r)
                    steps += 1
                    if info.get("invalid_action", False):
                        invalid += 1

                final_money.append(float(info.get("final_money", eval_env.money)))
                ep_reward.append(float(R))
                missed.append(float(info.get("missed_charger_days", 0)))
                invalid_rates.append(float(invalid) / max(1.0, float(steps)))
        finally:
            if was_training:
                agent.q_online.train()

        fm = np.array(final_money, dtype=np.float32)
        er = np.array(ep_reward, dtype=np.float32)
        ms = np.array(missed, dtype=np.float32)
        ir = np.array(invalid_rates, dtype=np.float32)
        return {
            "eval_final_money_mean": float(fm.mean()),
            "eval_final_money_std": float(fm.std()),
            "eval_reward_mean": float(er.mean()),
            "eval_reward_std": float(er.std()),
            "eval_missed_charger_days_mean": float(ms.mean()),
            "eval_invalid_rate_mean": float(ir.mean()),
        }

    metrics: Dict[str, List[float]] = {
        "episode_reward": [],
        "final_money": [],
        "loss": [],
        "epsilon": [],
        "invalid_rate": [],
        "missed_charger_days": [],
        "best_money_so_far": [],
        # Sparse eval metrics; NaN when not evaluated this episode
        "eval_final_money_mean": [],
        "eval_final_money_std": [],
        "eval_reward_mean": [],
        "eval_reward_std": [],
        "eval_missed_charger_days_mean": [],
        "eval_invalid_rate_mean": [],
    }

    best_money = -1e18
    action_names = [a.name for a in Action]

    # Action histogram logging (grouped by prefix: MOVE, PLANT, etc.)
    group_labels: List[str] = []
    for n in action_names:
        g = _action_group(n)
        if g not in group_labels:
            group_labels.append(g)
    action_to_group_idx = np.array([group_labels.index(_action_group(n)) for n in action_names], dtype=np.int32)
    group_counts_history: List[np.ndarray] = []

    hist_snap_eps: List[int] = []
    hist_snap_fracs: List[np.ndarray] = []

    action_hist_f = None
    action_hist_writer = None
    if int(action_hist_every) > 0:
        os.makedirs(exp_dir, exist_ok=True)
        action_hist_f = open(os.path.join(exp_dir, "action_hist.csv"), "w", newline="")
        action_hist_writer = csv.writer(action_hist_f)
        header = [
            "episode",
            "window_episodes",
            "window_total_actions",
            "epsilon",
        ]
        header += [f"count_{g}" for g in group_labels]
        header += [f"frac_{g}" for g in group_labels]
        action_hist_writer.writerow(header)
        action_hist_f.flush()

    try:
        for ep in range(1, episodes + 1):
            ep_seed = seed + ep

            # Optional curriculum: disable maturity gate early to reduce delayed credit.
            if n_cur > 0 and ep <= n_cur:
                env.cfg.ENABLE_MATURITY_GATE = False
            else:
                env.cfg.ENABLE_MATURITY_GATE = gate_after_curriculum

            obs, info = env.reset(seed=ep_seed)

            done = False
            ep_reward = 0.0
            last_loss = float("nan")
            steps = 0
            invalid_count = 0
            act_counts = np.zeros(num_actions, dtype=np.int64)

            while not done:
                mask = env.valid_action_mask()
                a = agent.act(obs, mask=mask, greedy=False)

                obs2, r, term, trunc, info = env.step(int(a))
                done = bool(term or trunc)

                mask2 = info.get("action_mask")
                if mask2 is None:
                    mask2 = env.valid_action_mask()

                r_train = float(r) * float(reward_scale)
                if float(reward_clip) > 0.0:
                    r_train = float(np.clip(r_train, -float(reward_clip), float(reward_clip)))

                agent.push(obs, int(a), float(r_train), obs2, done, mask2)

                loss = agent.train_step()
                if loss is not None:
                    last_loss = float(loss)

                # Log raw environment reward (not scaled/clipped)
                ep_reward += float(r)
                obs = obs2
                steps += 1
                act_counts[int(a)] += 1
                if info.get("invalid_action", False):
                    invalid_count += 1

            final_money = float(info.get("final_money", env.money))
            missed_days = float(info.get("missed_charger_days", 0))
            inv_rate = float(invalid_count) / max(1.0, float(steps))
            eps = float(agent.epsilon())

            # Per-episode grouped action counts
            ep_group_counts = np.zeros(len(group_labels), dtype=np.int64)
            for ai, c in enumerate(act_counts.tolist()):
                if c:
                    ep_group_counts[int(action_to_group_idx[ai])] += int(c)
            group_counts_history.append(ep_group_counts)

            # Windowed histogram snapshot every N episodes (and on episode 1)
            if action_hist_writer is not None and (ep == 1 or ep % int(action_hist_every) == 0):
                window = int(min(int(action_hist_window), len(group_counts_history)))
                win_counts = np.sum(group_counts_history[-window:], axis=0)
                total = float(np.sum(win_counts))
                if total <= 0.0:
                    total = 1.0
                fracs_arr = (win_counts.astype(np.float64) / total)
                fracs = fracs_arr.tolist()

                row = [int(ep), int(window), int(np.sum(win_counts)), float(eps)]
                row += [int(x) for x in win_counts.tolist()]
                row += [float(x) for x in fracs]
                action_hist_writer.writerow(row)
                action_hist_f.flush()

                # Keep snapshot history for a running plot
                hist_snap_eps.append(int(ep))
                hist_snap_fracs.append(fracs_arr.astype(np.float64, copy=False))

                plot_mode = str(action_hist_plot).lower().strip()
                out_dir = os.path.join(exp_dir, "action_hists")
                if plot_mode in ("bars", "both"):
                    out_path = os.path.join(out_dir, f"actions_ep{ep:04d}.png")
                    title = f"{exp_name} action distribution (episodes {ep - window + 1}-{ep})"
                    save_action_bars_png(
                        out_path,
                        labels=group_labels,
                        counts=win_counts,
                        title=title,
                        as_percent=True,
                    )
                if plot_mode in ("lines", "both"):
                    out_path = os.path.join(out_dir, "actions_over_time.png")
                    title = f"{exp_name} action distribution over time (window={window} eps)"
                    save_action_group_lines_png(
                        out_path,
                        episodes=hist_snap_eps,
                        labels=group_labels,
                        fracs=hist_snap_fracs,
                        title=title,
                    )

            maybe_save_best_replay(ep_seed=int(ep_seed), eps=float(eps), final_money=float(final_money))

            best_money = max(best_money, final_money)

            metrics["episode_reward"].append(float(ep_reward))
            metrics["final_money"].append(final_money)
            metrics["loss"].append(float(last_loss))
            metrics["epsilon"].append(eps)
            metrics["invalid_rate"].append(inv_rate)
            metrics["missed_charger_days"].append(missed_days)
            metrics["best_money_so_far"].append(float(best_money))

            # Default sparse eval values
            metrics["eval_final_money_mean"].append(float("nan"))
            metrics["eval_final_money_std"].append(float("nan"))
            metrics["eval_reward_mean"].append(float("nan"))
            metrics["eval_reward_std"].append(float("nan"))
            metrics["eval_missed_charger_days_mean"].append(float("nan"))
            metrics["eval_invalid_rate_mean"].append(float("nan"))

            if int(eval_every) > 0 and (ep % int(eval_every) == 0):
                out = greedy_evaluate()
                metrics["eval_final_money_mean"][-1] = out["eval_final_money_mean"]
                metrics["eval_final_money_std"][-1] = out["eval_final_money_std"]
                metrics["eval_reward_mean"][-1] = out["eval_reward_mean"]
                metrics["eval_reward_std"][-1] = out["eval_reward_std"]
                metrics["eval_missed_charger_days_mean"][-1] = out["eval_missed_charger_days_mean"]
                metrics["eval_invalid_rate_mean"][-1] = out["eval_invalid_rate_mean"]

            if ep == 1 or ep % print_every == 0:
                top_idx = np.argsort(-act_counts)[:6]
                top_actions = ", ".join([f"{action_names[i]}:{int(act_counts[i])}" for i in top_idx])
                window = min(10, len(metrics["episode_reward"]))
                meanR = float(np.mean(metrics["episode_reward"][-window:]))
                meanM = float(np.mean(metrics["final_money"][-window:]))
                print(
                    f"{exp_name} | Ep {ep:4d} | steps={steps:5d} | "
                    f"meanR({window})={meanR:8.2f} | meanMoney({window})={meanM:8.2f} | "
                    f"bestMoneySoFar={best_money:8.2f} | invalid={invalid_count:4d} ({inv_rate*100:5.2f}%) | "
                    f"missedChgDays={int(missed_days):4d} | eps={eps:5.3f} | lastLoss={last_loss:.4f} | "
                    f"top_actions=({top_actions})"
                )

            if ep % save_every == 0:
                # For long runs, plotting every checkpoint can dominate runtime.
                # Save CSV/NPZ each checkpoint; optionally also generate plots.
                save_metrics(exp_dir, metrics, plots=bool(plots_at_checkpoints))
                ckpt_path = os.path.join(exp_dir, f"ckpt_ep{ep}.pt")
                agent.save(ckpt_path)

    except KeyboardInterrupt:
        # If the user stops a long run, still emit plots once so they can inspect progress.
        try:
            save_metrics(exp_dir, metrics, plots=True)
            agent.save(os.path.join(exp_dir, "ckpt_interrupt.pt"))
        except Exception:
            pass
        print(f"[interrupted] saved partial metrics/plots -> {exp_dir}")
        raise
    finally:
        try:
            if action_hist_f is not None:
                action_hist_f.close()
        except Exception:
            pass

    model_path = os.path.join(exp_dir, "double_dqn_agri.pt")
    agent.save(model_path)
    try:
        shutil.copy(model_path, os.path.join(os.getcwd(), f"double_dqn_agri_{exp_name}.pt"))
    except Exception:
        pass

    save_metrics(exp_dir, metrics, plots=True)

    if bool(save_replays):
        try:
            os.makedirs(str(replay_dir), exist_ok=True)

            # Evaluate with the "final" environment behavior (post curriculum)
            cfg_replay = make_config()
            for k, v in env_cfg_overrides.items():
                setattr(cfg_replay, k, v)
            cfg_replay.ENABLE_MATURITY_GATE = gate_after_curriculum

            replay_env = AgriRobotEnv(cfg_replay, seed=int(replay_seed))

            was_training = agent.q_online.training
            agent.q_online.eval()
            try:
                eps_data: List[Dict[str, np.ndarray]] = []
                eps_money: List[float] = []
                for i in range(int(replay_episodes)):
                    data, fm = record_greedy_episode(
                        replay_env,
                        agent,
                        seed=int(replay_seed + 1000 + i),
                        max_steps=int(steps_per_ep) + 5,
                    )
                    eps_data.append(data)
                    eps_money.append(float(fm))
            finally:
                if was_training:
                    agent.q_online.train()

            arr = np.array(eps_money, dtype=np.float32)
            i_best = int(np.argmax(arr))
            i_worst = int(np.argmin(arr))
            mean_money = float(arr.mean())
            i_avg = int(np.argmin(np.abs(arr - mean_money)))

            run_id = os.path.basename(os.path.dirname(exp_dir))
            for tag, idx in (("best", i_best), ("worst", i_worst), ("avg", i_avg)):
                out_path = os.path.join(str(replay_dir), f"{run_id}_{exp_name}_{tag}.npz")
                np.savez(out_path, allow_pickle=False, **eps_data[idx])

            print(
                f"[saved] replays -> {replay_dir} | {exp_name}: best={arr[i_best]:.2f} worst={arr[i_worst]:.2f} mean={mean_money:.2f}"
            )
        except Exception as e:
            print(f"[warn] failed to save replays for {exp_name}: {e}")

    print(f"[done] {exp_name} saved model -> {model_path}")
    return model_path, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replot",
        type=str,
        default="",
        help="Regenerate plots for an existing run directory (e.g. runs/20260111_133608 or 20260111_133608)",
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--num-days", type=int, default=30)
    # Default increased to allow more within-episode exploration.
    parser.add_argument("--actions-per-day", type=int, default=60)

    # Recommended defaults for this environment (better delayed-credit handling)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--replay-size", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=128)
    # Keep the "learning begins" point roughly comparable in episode-count
    # when using 60 actions/day instead of 40.
    parser.add_argument("--learning-starts", type=int, default=15_000)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--target-every", type=int, default=5_000)
    parser.add_argument(
        "--tau",
        type=float,
        default=0.0,
        help=(
            "Soft target update coefficient (Polyak averaging). "
            "0 disables (uses hard copy every --target-every env steps). "
            "Typical values: 0.005 or 0.01."
        ),
    )
    # eps_decay_steps is in environment steps. With 60 actions/day (vs 40),
    # we scale the default so epsilon falls at roughly the same episode count.
    parser.add_argument("--eps-decay", type=int, default=2_250_000)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=10.0,
        help="Gradient clip norm (0 disables clipping)",
    )

    # Experiment suite / comparison
    parser.add_argument(
        "--suite",
        type=str,
        choices=["requested", "auto"],
        default="requested",
        help=(
            "Which experiment suite to run. "
            "requested: baseline + nstep3 + nstep5 + dueling + both(nstep5+dueling). "
            "auto: baseline + nstep(args) + dueling + both (only when dueling or n_step>1)."
        ),
    )

    parser.add_argument(
        "--experiments",
        type=str,
        default="",
        help=(
            "Comma-separated experiment names to run (filters the chosen suite). "
            "Example: --experiments baseline,nstep3"
        ),
    )

    # Innovation B toggles (used when --suite auto)
    parser.add_argument(
        "--dueling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable dueling head (Innovation B #2)",
    )
    parser.add_argument("--n-step", type=int, default=3, help="Enable n-step returns (Innovation B #1); try 3 or 5")
    parser.add_argument(
        "--double-dqn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable Double DQN target (disable => Single DQN)",
    )
    parser.add_argument("--no-compare", action="store_true", help="Disable comparison plots/summary (still runs suite)")

    # Logging / checkpoint cadence
    parser.add_argument("--print-every", type=int, default=50, help="Print training stats every N episodes")
    parser.add_argument("--save-every", type=int, default=100, help="Save checkpoint + metrics every N episodes")
    parser.add_argument(
        "--plots-at-checkpoints",
        action="store_true",
        help="Also write PNG plots at each checkpoint save (slower). Plots are always written at the end or on interrupt.",
    )

    parser.add_argument(
        "--action-hist-every",
        type=int,
        default=100,
        help=(
            "Save a grouped action histogram (MOVE/PLANT/etc.) every N episodes (0 disables). "
            "Writes action_hist.csv and action_hists/actions_epXXXX.png per experiment."
        ),
    )
    parser.add_argument(
        "--action-hist-window",
        type=int,
        default=100,
        help="Episode window used for each histogram snapshot.",
    )
    parser.add_argument(
        "--action-hist-plot",
        type=str,
        choices=["lines", "bars", "both"],
        default="lines",
        help="Visualization type for action history snapshots.",
    )

    # Periodic greedy evaluation during training
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Run greedy evaluation every N episodes (0 disables)",
    )
    parser.add_argument("--eval-episodes", type=int, default=20, help="How many greedy eval episodes to run")
    parser.add_argument("--eval-seed", type=int, default=12345, help="Base seed for greedy eval rollouts")

    # Environment-side training tweaks
    parser.add_argument(
        "--maturity-curriculum-episodes",
        type=int,
        default=500,
        help="Disable maturity gate for first N episodes (0 disables curriculum)",
    )
    parser.add_argument(
        "--maturity-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable maturity gate after curriculum",
    )
    parser.add_argument("--min-harvest-stage", type=int, default=2, help="MIN_HARVEST_STAGE when maturity gate is enabled")

    # Reward shaping knobs (keep magnitudes small)
    parser.add_argument("--plant-bonus", type=float, default=0.05)
    parser.add_argument("--trample-penalty", type=float, default=-0.20)
    parser.add_argument("--dropoff-bonus-per-item", type=float, default=0.02)
    # Step penalty is per-step; with 60 actions/day we reduce magnitude so the
    # per-episode total penalty stays comparable to the 40-actions/day setting.
    parser.add_argument("--step-penalty", type=float, default=-0.0006667)

    # Training-only reward scaling (does not change env dynamics / sim state)
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=1.0,
        help="Multiply rewards by this factor before storing to replay (logging still uses raw env reward).",
    )
    parser.add_argument(
        "--reward-clip",
        type=float,
        default=0.0,
        help="If >0, clip scaled rewards into [-reward_clip, +reward_clip] before replay.",
    )

    # Best-only replay saving (low overhead)
    parser.add_argument(
        "--save-best-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continuously overwrite a single best replay per experiment once epsilon drops below a threshold",
    )
    parser.add_argument(
        "--best-replay-eps-threshold",
        type=float,
        default=0.3,
        help="Only start saving best replay once epsilon is below this value",
    )
    parser.add_argument(
        "--best-replay-dir",
        type=str,
        default="replays",
        help="Directory to write best replay NPZ files",
    )

    # Legacy/heavier replay saving (samples many greedy episodes to pick best/worst/avg)
    parser.add_argument(
        "--save-replays",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="(Heavier) Save best/worst/closest-to-mean greedy episodes for each experiment as NPZ replays",
    )
    parser.add_argument("--replay-dir", type=str, default="replays", help="Directory to write replay NPZ files")
    parser.add_argument(
        "--replay-episodes",
        type=int,
        default=25,
        help="How many greedy episodes to sample per experiment when selecting best/worst/avg",
    )
    parser.add_argument("--replay-seed", type=int, default=4242, help="Base seed for replay rollouts")

    args = parser.parse_args()

    if str(args.replot).strip():
        replot_run(str(args.replot).strip())
        return

    # Decide experiments.
    # - suite=requested always runs the user-requested 5-way sweep.
    # - suite=auto runs a smaller sweep only when dueling or n_step>1.
    if args.suite == "requested":
        experiments = [
            ("baseline", False, 1),
            ("singledqn", False, 1),
            ("nstep3", False, 3),
            ("nstep5", False, 5),
            ("dueling", True, 1),
            ("both", True, 5),
        ]
        do_compare = not args.no_compare
    else:
        do_compare = (not args.no_compare) and (args.dueling or args.n_step > 1)
        if do_compare:
            experiments = [
                ("baseline", False, 1),
                ("nstep", False, max(2, int(args.n_step))),
                ("dueling", True, 1),
                ("both", True, max(2, int(args.n_step))),
            ]
        else:
            experiments = [
                ("custom", bool(args.dueling), int(max(1, args.n_step))),
            ]

    # Optional: filter experiments by name
    if str(args.experiments).strip():
        wanted = {s.strip() for s in str(args.experiments).split(",") if s.strip()}
        if not wanted:
            raise SystemExit("--experiments was provided but empty")
        before = list(experiments)
        experiments = [e for e in experiments if e[0] in wanted]
        missing = wanted - {e[0] for e in experiments}
        if missing:
            raise SystemExit(f"Unknown experiment(s) requested: {sorted(missing)}. Available: {[e[0] for e in before]}")
        do_compare = do_compare and (len(experiments) > 1)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    parent_dir = os.path.join("runs", timestamp)
    os.makedirs(parent_dir, exist_ok=True)

    env_overrides = {
        "NUM_DAYS": int(args.num_days),
        "BATTERY_ACTIONS_PER_DAY": int(args.actions_per_day),
        # keep these off (as you requested), weeds on:
        "ENABLE_SATURATION": False,
        "ENABLE_FERTILIZER": False,
        "ENABLE_WATERING": False,
        "ENABLE_WEEDS": True,
        "ENABLE_SPRAYING": True,

        # Maturity gate: use a short curriculum to reduce delayed-credit early
        "ENABLE_MATURITY_GATE": bool(args.maturity_gate),
        "MIN_HARVEST_STAGE": int(args.min_harvest_stage),
        "MATURITY_CURRICULUM_EPISODES": int(args.maturity_curriculum_episodes),

        # Reward shaping
        "ENABLE_STEP_PENALTY": True,
        "STEP_PENALTY": float(args.step_penalty),
        "ENABLE_PLANT_BONUS": True,
        "PLANT_BONUS": float(args.plant_bonus),
        "ENABLE_TRAMPLE_PENALTY": True,
        "TRAMPLE_PENALTY": float(args.trample_penalty),
        "ENABLE_DROPOFF_BONUS": True,
        "DROPOFF_BONUS_PER_ITEM": float(args.dropoff_bonus_per_item),
    }

    base_cfg = DQNConfig(
        gamma=float(args.gamma),
        lr=float(args.lr),
        replay_size=int(args.replay_size),
        batch_size=int(args.batch_size),
        learning_starts=int(args.learning_starts),
        train_every=int(args.train_every),
        target_update_every=int(args.target_every),
        tau=float(args.tau),
        eps_start=1.0,
        eps_end=float(args.eps_end),
        eps_decay_steps=int(args.eps_decay),
        grad_clip_norm=float(args.grad_clip),
        dueling=False,
        n_step=1,
        double_dqn=bool(args.double_dqn),
    )

    # experiments already chosen above

    steps_per_ep = int(args.num_days) * int(args.actions_per_day)
    total_steps = steps_per_ep * int(args.episodes)

    print(f"Device: {args.device}")
    print(f"Run dir: {parent_dir}")
    print(f"NUM_DAYS={args.num_days} | ACTIONS/DAY={args.actions_per_day} | steps/ep≈{steps_per_ep}")
    print(f"Episodes={args.episodes} | total env steps≈{total_steps}")
    print(
        f"DQN: replay={args.replay_size} bs={args.batch_size} starts={args.learning_starts} "
        f"train_every={args.train_every} target_every={args.target_every} tau={args.tau} "
        f"eps_decay={args.eps_decay} lr={args.lr} grad_clip={args.grad_clip} "
        f"reward_scale={args.reward_scale} reward_clip={args.reward_clip}"
    )
    if do_compare:
        print(f"Comparison sweep: {experiments}")

    try:
        torch.set_num_threads(max(1, os.cpu_count() // 2))
    except Exception:
        pass

    all_metrics: Dict[str, Dict[str, List[float]]] = {}

    for exp_name, dueling, n_step in experiments:
        exp_dir = os.path.join(parent_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        dcfg = DQNConfig(**{**base_cfg.__dict__})
        dcfg.dueling = bool(dueling)
        dcfg.n_step = int(n_step)

        # Suite-specific algorithm toggle
        if exp_name == "singledqn":
            dcfg.double_dqn = False

        print(f"\n=== EXPERIMENT: {exp_name} | dueling={dcfg.dueling} | n_step={dcfg.n_step} ===")

        _, metrics = run_one_experiment(
            exp_dir=exp_dir,
            exp_name=exp_name,
            episodes=int(args.episodes),
            seed=int(args.seed),
            device=str(args.device),
            dqn_cfg=dcfg,
            env_cfg_overrides=env_overrides,
            print_every=int(args.print_every),
            save_every=int(args.save_every),
            eval_every=int(args.eval_every),
            eval_episodes=int(args.eval_episodes),
            eval_seed=int(args.eval_seed),
            plots_at_checkpoints=bool(args.plots_at_checkpoints),
            reward_scale=float(args.reward_scale),
            reward_clip=float(args.reward_clip),
            action_hist_every=int(args.action_hist_every),
            action_hist_window=int(args.action_hist_window),
            action_hist_plot=str(args.action_hist_plot),
            save_best_replay=bool(args.save_best_replay),
            best_replay_dir=str(args.best_replay_dir),
            best_replay_eps_threshold=float(args.best_replay_eps_threshold),
            save_replays=bool(args.save_replays),
            replay_dir=str(args.replay_dir),
            replay_episodes=int(args.replay_episodes),
            replay_seed=int(args.replay_seed),
        )
        all_metrics[exp_name] = metrics

    if do_compare:
        save_comparison(parent_dir, all_metrics)
        print(f"[saved] comparison plots -> {parent_dir}")

    print("[done]")


if __name__ == "__main__":
    main()
