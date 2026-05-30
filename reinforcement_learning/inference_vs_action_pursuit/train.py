"""Train PPO on InferencePursuitEnv and export episode statistics."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
torch.set_num_threads(1)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from inference_pursuit.env import InferencePursuitEnv


class _EpisodeLogger(BaseCallback):
    """Collects per-episode stats from Monitor info dicts; works with any n_envs."""

    def __init__(self):
        super().__init__()
        self.episodes: list[dict] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.episodes.append({
                    "reward": float(ep["r"]),
                    "length": int(ep["l"]),
                    "success": int(ep["r"] > 0),
                })
        return True


def run_training(
    run_name: str,
    N: int,
    T: int,
    p_drift: float,
    R_catch: float,
    sigma0: float,
    alpha: float,
    total_timesteps: int,
    n_envs: int = 4,
    checkpoint_every: int = 0,  # 0 = disabled; otherwise save every N timesteps
    output_base: str = "models",
) -> str:
    """Train a PPO agent and save model + episode CSV + training curve PNG.

    Returns the output directory path.
    """
    output_dir = os.path.join(output_base, run_name)
    os.makedirs(output_dir, exist_ok=True)

    env_kwargs = dict(
        N=N, T=T, p_drift=p_drift,
        R_catch=R_catch,
        sigma0=sigma0, alpha=alpha,
    )

    def _make():
        return Monitor(InferencePursuitEnv(**env_kwargs))

    env = make_vec_env(_make, n_envs=n_envs)

    episode_logger = _EpisodeLogger()
    callbacks: list = [episode_logger]
    if checkpoint_every > 0:
        callbacks.append(CheckpointCallback(
            save_freq=max(1, checkpoint_every // n_envs),  # SB3 counts per-env steps
            save_path=os.path.join(output_dir, "checkpoints"),
            name_prefix="ckpt",
            verbose=0,
        ))
    callback = CallbackList(callbacks)

    model = PPO("MlpPolicy", env, verbose=0, device="cpu")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    model_path = os.path.join(output_dir, "ppo_model")
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    if episode_logger.episodes:
        df = pd.DataFrame(episode_logger.episodes)
        df.insert(0, "episode", range(len(df)))

        csv_path = os.path.join(output_dir, "episode_stats.csv")
        df.to_csv(csv_path, index=False)
        print(f"Episode stats → {csv_path}")

        w = min(100, len(df))
        df["roll_reward"] = df["reward"].rolling(w, min_periods=1).mean()
        df["roll_success"] = df["success"].rolling(w, min_periods=1).mean()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax1.plot(df["episode"], df["roll_reward"], linewidth=1.5)
        ax1.set_ylabel("Rolling Mean Reward")
        ax1.set_title(f"Training — {run_name}  (alpha={alpha}, sigma0={sigma0})")
        ax2.plot(df["episode"], df["roll_success"], color="green", linewidth=1.5)
        ax2.set_ylabel("Rolling Success Rate")
        ax2.set_xlabel("Episode")
        ax2.set_ylim(0, 1)
        plt.tight_layout()
        png_path = os.path.join(output_dir, "training_curve.png")
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"Training curve → {png_path}")

    return output_dir


# ── CLI wrapper (backward-compatible) ────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", type=str, required=True,
                   help="Subfolder name inside models/")
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--T", type=int, required=True)
    p.add_argument("--p_drift", type=float, required=True)
    p.add_argument("--R_catch", type=float, required=True)
    p.add_argument("--sigma0", type=float, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--total_timesteps", type=int, required=True)
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--checkpoint_every", type=int, default=0,
                   help="Save a checkpoint every N timesteps (0 = disabled)")
    p.add_argument("--output_base", type=str, default="models")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_training(
        run_name=args.run_name,
        N=args.N, T=args.T,
        p_drift=args.p_drift, R_catch=args.R_catch,
        sigma0=args.sigma0, alpha=args.alpha,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        checkpoint_every=args.checkpoint_every,
        output_base=args.output_base,
    )
