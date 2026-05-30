#!/usr/bin/env python3
"""
Single-experiment launcher.

Edit the run_training() call below, then:
    python3 main.py

The run_name becomes the folder: models/<run_name>/
  - ppo_model.zip
  - episode_stats.csv
  - training_curve.png

See experiments.txt for the full experiment plan (38 runs across 3 groups).
"""

from train import run_training

# ════════════════════════════════════════════════════════════════════
# EDIT THIS BLOCK — one experiment at a time
# ════════════════════════════════════════════════════════════════════

run_training(
    run_name="control_drift_static",

    N=20,
    T=50,
    p_drift=0.0,
    R_catch=1.0,
    sigma0=0.00001,
    alpha=0.0,

    total_timesteps=500_000,
    n_envs=4,
    checkpoint_every=50_000,
)
