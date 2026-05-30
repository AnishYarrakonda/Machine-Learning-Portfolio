# Inference vs. Move Pursuit

A discrete-time POMDP (Partially Observable Markov Decision Process) designed to study the fundamental tradeoff between **information gathering** (Inference) and **environmental execution** (Move) in reinforcement learning.

## Overview

The agent tracks a stochastically drifting target on a 2D toroidal grid under a fixed time budget. At each timestep, the agent must choose: **Inference** (observe the target's noisy position from its current location) or **Move** (blindly walk one cell in a cardinal direction, incurring belief uncertainty).

The key innovation is the **noise profile shape parameter α**, which determines *where* observations are informative:
- **α > 0**: Observations are clear nearby, noisy far away → rational to move blind early, then infer when close
- **α < 0**: Observations are clear far away, noisy nearby → rational to infer while distant, then commit to movement
- **α = 0**: Uniform noise everywhere → no spatial structure

The agent receives **zero reward during the episode** and a **terminal reward**: `1 - t/T` on catch (earlier is better), or `-1` on miss. This sparse design ensures the reward channel leaks no distance-based movement guidance.

## Project Structure

```
src/inference_pursuit/
├── geometry.py          # Toroidal distance utilities
├── belief_filter.py     # Discrete Bayesian filter with np.roll-based drift
├── env.py              # InferencePursuitEnv (Gymnasium-compliant)
└── __init__.py

train.py               # SB3 PPO training script
analyze.py             # Alpha-sweep evaluation & visualization
tests/                 # 29 unit tests (geometry, filter, env)
```

## Installation

```bash
pip install -e .
pip install pytest
```

## Quick Start (30 seconds)

Train a single model and visualize:

```bash
# Train model with α=0.0
python3 train.py \
  --N 10 --T 20 --p_drift 0.2 --R_catch 1.0 --sigma0 1.0 \
  --alpha 0.0 --total_timesteps 5000 \
  --output_dir models/alpha_0.0/

# Run analysis
python3 analyze.py \
  --alpha 0.0 \
  --model_dir models/ \
  --N 10 --T 20 --p_drift 0.2 --R_catch 1.0 --sigma0 1.0 \
  --n_eval_episodes 5 --n_bins 3 \
  --output_dir analysis_output/
```

Output: `analysis_output/inference_density.png`, `success_rates.png`, and corresponding CSV files.

## Full Experiment (Multi-α Sweep)

Train models across the α parameter space:

```bash
# 1. Train one model per alpha value
for ALPHA in -2.0 -1.0 0.0 1.0 2.0; do
  python3 train.py \
    --N 20 --T 50 --p_drift 0.2 --R_catch 1.0 --sigma0 1.0 \
    --alpha $ALPHA --total_timesteps 500000 \
    --output_dir "models/alpha_${ALPHA}/"
done

# 2. Run the alpha-sweep analysis
python3 analyze.py \
  --alpha -2.0 -1.0 0.0 1.0 2.0 \
  --model_dir models/ \
  --N 20 --T 50 --p_drift 0.2 --R_catch 1.0 --sigma0 1.0 \
  --n_eval_episodes 500 --n_bins 10 \
  --output_dir analysis_output/
```

## Output Files

### From `train.py`
- `ppo_model.zip` — Trained PPO policy
- `episode_stats.csv` — Episode rewards, lengths, success flags
- `training_curve.png` — Rolling reward & success rate curves

### From `analyze.py`
- `inference_density.csv` — Binned inference frequency vs. distance (columns: alpha, bin_center, inference_density, n_steps)
- `inference_density.png` — **Main result plot** showing how inference behavior changes with α
- `success_rates.csv` — Success rate, mean reward, episode count per α
- `success_rates.png` — Bar chart of success rates (target ≥95%)

## Expected Primary Result

In `analysis_output/inference_density.png`:
- **α > 0 curve rises**: agent infers more as distance-to-target decreases (optimal: move blind early, observe when close)
- **α < 0 curve falls**: agent infers more as distance increases (optimal: observe while far, move blind when close)
- **α = 0 curve is flat**: no distance-dependent signal in observations

All curves should achieve **success_rate ≥ 0.95** under sufficient training.

## Key Parameters

| Parameter | Role |
|-----------|------|
| `N` | Grid size (N × N torus) |
| `T` | Episode length (timesteps) |
| `p_drift` | Probability target moves each step |
| `R_catch` | Catch radius: distance threshold for success |
| `sigma0` | Base observation noise scale |
| `alpha` | Noise profile shape (controls where obs are informative) |
| `total_timesteps` | SB3 training budget (500k+ recommended) |
| `n_eval_episodes` | Episodes for α-sweep evaluation (500 typical) |
| `n_bins` | Distance bins for inference density histogram (10 typical) |

## Testing

Run all tests:

```bash
pytest tests/ -v
```

Expected: **29 tests pass** (geometry, belief filter, environment API compliance).

## References

- Gymnasium: https://gymnasium.farama.org/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- Toroidal grids: Ensure uniform geometry and prevent boundary-induced biases
- Discrete Bayesian filtering: np.roll-based drift prediction, log-space likelihood updates
