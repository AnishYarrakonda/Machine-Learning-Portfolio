"""Evaluate trained PPO policies; export inference-density and success-rate data."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from inference_pursuit.env import InferencePursuitEnv
from inference_pursuit.geometry import torus_dist


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, nargs="+", required=True)
    p.add_argument("--model_dir", type=str, required=True,
                   help="Contains alpha_{val}/ppo_model.zip per alpha")
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--T", type=int, required=True)
    p.add_argument("--p_drift", type=float, required=True)
    p.add_argument("--R_catch", type=float, required=True)
    p.add_argument("--sigma0", type=float, required=True)
    p.add_argument("--n_eval_episodes", type=int, required=True)
    p.add_argument("--n_bins", type=int, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def eval_alpha(model, env_kwargs, n_episodes):
    alpha = env_kwargs["alpha"]
    step_rows, ep_rows = [], []
    for ep in range(n_episodes):
        env = InferencePursuitEnv(**env_kwargs)
        obs, _ = env.reset(seed=ep)
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            d = torus_dist(tuple(env.agent_pos), env.target_pos, env_kwargs["N"])
            obs, reward, term, trunc, _ = env.step(int(action))
            done = term or trunc
            ep_reward += reward
            step_rows.append({"alpha": alpha, "distance": d, "action": int(action)})
        ep_rows.append({"alpha": alpha, "episode": ep, "reward": ep_reward,
                        "success": int(ep_reward > 0)})
    return step_rows, ep_rows


def inference_density_df(step_rows, n_bins, max_dist):
    df = pd.DataFrame(step_rows)
    edges = np.linspace(0, max_dist, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    df["bin"] = pd.cut(df["distance"], bins=edges, labels=False, include_lowest=True)
    rows = []
    for alpha_val, grp in df.groupby("alpha"):
        for b in range(n_bins):
            sub = grp[grp["bin"] == b]
            n = len(sub)
            density = sub["action"].eq(0).mean() if n > 0 else float("nan")
            rows.append({"alpha": alpha_val, "bin_center": centers[b],
                         "inference_density": density, "n_steps": n})
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    max_dist = args.N * (2 ** 0.5) / 2.0

    all_steps, all_eps = [], []
    for alpha in sorted(args.alpha):
        model_path = os.path.join(args.model_dir, f"alpha_{alpha}", "ppo_model.zip")
        if not os.path.exists(model_path):
            print(f"WARNING: {model_path} not found, skipping alpha={alpha}")
            continue
        print(f"Evaluating alpha={alpha} ...")
        env_kwargs = dict(N=args.N, T=args.T, p_drift=args.p_drift, R_catch=args.R_catch,
                          sigma0=args.sigma0, alpha=alpha)
        model = PPO.load(model_path)
        steps, eps = eval_alpha(model, env_kwargs, args.n_eval_episodes)
        all_steps.extend(steps)
        all_eps.extend(eps)

    if not all_steps:
        print("No data. Exiting.")
        return

    # Inference density
    dens_df = inference_density_df(all_steps, args.n_bins, max_dist)
    dens_csv = os.path.join(args.output_dir, "inference_density.csv")
    dens_df.to_csv(dens_csv, index=False)
    print(f"Inference density → {dens_csv}")

    fig, ax = plt.subplots(figsize=(9, 5))
    for alpha_val, grp in dens_df.groupby("alpha"):
        g = grp.dropna(subset=["inference_density"])
        ax.plot(g["bin_center"], g["inference_density"], marker="o", ms=4, label=f"α={alpha_val}")
    ax.set_xlabel("Distance to Target")
    ax.set_ylabel("Inference Density P(action=0)")
    ax.set_title("Inference Density vs. Distance (by α)")
    ax.set_ylim(0, 1)
    ax.legend()
    dens_png = os.path.join(args.output_dir, "inference_density.png")
    fig.savefig(dens_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Inference density plot → {dens_png}")

    # Success rates
    ep_df = pd.DataFrame(all_eps)
    succ_df = (ep_df.groupby("alpha")
               .agg(success_rate=("success", "mean"),
                    mean_reward=("reward", "mean"),
                    n_episodes=("episode", "count"))
               .reset_index())
    succ_csv = os.path.join(args.output_dir, "success_rates.csv")
    succ_df.to_csv(succ_csv, index=False)
    print(f"Success rates → {succ_csv}")

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.bar(succ_df["alpha"].astype(str), succ_df["success_rate"], color="steelblue")
    ax2.axhline(0.95, color="red", linestyle="--", label="95% target")
    ax2.set_xlabel("α")
    ax2.set_ylabel("Success Rate")
    ax2.set_title("Success Rate by α")
    ax2.set_ylim(0, 1)
    ax2.legend()
    succ_png = os.path.join(args.output_dir, "success_rates.png")
    fig2.savefig(succ_png, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Success rates plot → {succ_png}")


if __name__ == "__main__":
    main()
