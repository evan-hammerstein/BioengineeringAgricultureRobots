# evaluate_policy.py
import os
import argparse
import numpy as np
import torch

from config import make_config
from agri_env import AgriRobotEnv, Action
from dqn_double import DoubleDQNAgent, DQNConfig


def _infer_dueling_from_checkpoint(path: str) -> bool:
    """
    New checkpoints are saved as dicts with a 'dueling' flag.
    Old checkpoints are raw state_dicts (no flag).
    """
    try:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "dueling" in obj:
            return bool(obj["dueling"])
    except Exception:
        pass
    return False


def evaluate(model_path: str, episodes: int, seed: int, vary_seed: bool, verbose: bool, force_dueling: bool | None):
    cfg = make_config()
    env = AgriRobotEnv(cfg, seed=seed)

    obs_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    if force_dueling is None:
        dueling = _infer_dueling_from_checkpoint(model_path)
    else:
        dueling = bool(force_dueling)

    dqn_cfg = DQNConfig(dueling=dueling)
    agent = DoubleDQNAgent(obs_dim, num_actions, dqn_cfg, device="cpu")
    agent.load(model_path)

    print(f"Loaded model: {os.path.basename(model_path)}")
    print(f"Eval env: seed={seed} vary_seed={vary_seed} | NUM_DAYS={cfg.NUM_DAYS} | ACTIONS/DAY={cfg.BATTERY_ACTIONS_PER_DAY}")
    print(f"Policy net: dueling={dueling}")

    final_money = []
    ep_reward = []
    action_names = [a.name for a in Action]

    for ep in range(1, episodes + 1):
        ep_seed = seed + ep if vary_seed else seed
        obs, info = env.reset(seed=ep_seed)

        done = False
        R = 0.0
        steps = 0
        act_counts = np.zeros(num_actions, dtype=np.int64)

        while not done:
            mask = env.valid_action_mask()
            a = agent.act(obs, mask=mask, greedy=True)

            obs, r, term, trunc, info = env.step(int(a))
            done = bool(term or trunc)

            R += float(r)
            steps += 1
            act_counts[int(a)] += 1

        m = float(info.get("final_money", env.money))
        final_money.append(m)
        ep_reward.append(float(R))

        if verbose:
            top_idx = np.argsort(-act_counts)[:6]
            top_actions = ", ".join([f"{action_names[i]}:{int(act_counts[i])}" for i in top_idx])
            print(f"[EP {ep}] steps={steps} final_money={m:.2f} ep_reward={R:.2f} top_actions=({top_actions})")

    fm = np.array(final_money, dtype=np.float32)
    er = np.array(ep_reward, dtype=np.float32)
    print(f"Final money: mean={fm.mean():.2f}, std={fm.std():.2f}")
    print(f"Episode reward: mean={er.mean():.2f}, std={er.std():.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="double_dqn_agri_nstep3.pt")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--vary-seed", action="store_true")
    p.add_argument("--verbose", action="store_true")

    p.add_argument("--dueling", action="store_true", help="Force dueling=True (override autodetect)")
    p.add_argument("--no-dueling", action="store_true", help="Force dueling=False (override autodetect)")

    args = p.parse_args()

    force_dueling = None
    if args.dueling and args.no_dueling:
        raise SystemExit("Choose only one of --dueling or --no-dueling.")
    if args.dueling:
        force_dueling = True
    if args.no_dueling:
        force_dueling = False

    evaluate(
        model_path=args.model,
        episodes=int(args.episodes),
        seed=int(args.seed),
        vary_seed=bool(args.vary_seed),
        verbose=bool(args.verbose),
        force_dueling=force_dueling,
    )


if __name__ == "__main__":
    main()
