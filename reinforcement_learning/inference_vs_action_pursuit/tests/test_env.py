import numpy as np
import pytest
import gymnasium as gym
from inference_pursuit.env import InferencePursuitEnv

ENV_KW = dict(N=10, T=20, p_drift=0.2, R_catch=1.0, sigma0=1.0, alpha=1.0)


def make():
    return InferencePursuitEnv(**ENV_KW)


def test_reset_obs_shape():
    obs, info = make().reset(seed=0)
    assert obs.shape == (4,)
    assert obs.dtype == np.float32


def test_reset_info_has_target():
    _, info = make().reset(seed=0)
    assert "target_pos" in info
    assert len(info["target_pos"]) == 2


def test_obs_space_contains_reset_obs():
    env = make()
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs), f"obs {obs} out of bounds"


def test_action_space_discrete_5():
    assert make().action_space.n == 5


def test_step_returns_shapes():
    env = make()
    env.reset(seed=0)
    obs, r, term, trunc, info = env.step(0)
    assert obs.shape == (4,)
    assert isinstance(r, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)


def test_inference_does_not_move_agent():
    env = make()
    env.reset(seed=0)
    before = list(env.agent_pos)
    env.step(0)
    assert env.agent_pos == before


def test_move_north_changes_row():
    env = make()
    env.reset(seed=0)
    env.agent_pos = [5, 5]
    env.step(1)  # North: row -= 1
    assert env.agent_pos[0] == 4
    assert env.agent_pos[1] == 5


def test_episode_terminates_at_T():
    env = make()
    env.reset(seed=0)
    term = False
    for _ in range(ENV_KW["T"]):
        _, _, term, _, _ = env.step(0)
    assert term


def test_no_intermediate_reward():
    env = make()
    env.reset(seed=0)
    for _ in range(ENV_KW["T"] - 1):
        _, r, term, _, _ = env.step(0)
        assert r == 0.0
        assert not term


def test_terminal_reward_is_float():
    env = make()
    env.reset(seed=0)
    for _ in range(ENV_KW["T"] - 1):
        env.step(0)
    _, r, term, _, _ = env.step(0)
    assert term
    assert isinstance(r, float)


def test_gym_api_compliance():
    from gymnasium.utils.env_checker import check_env
    check_env(make(), warn=True)


def test_catch_terminates_early_with_shaped_reward():
    """When the agent steps onto the target, the episode ends with reward = 1 - t/T."""
    env = InferencePursuitEnv(N=10, T=20, p_drift=0.0, R_catch=1.0,
                              sigma0=1.0, alpha=1.0)
    env.reset(seed=0)
    # Place agent one cell south of target so action=2 (south, row+=1) catches it.
    env.target_pos = (5, 5)
    env.agent_pos = [4, 5]
    env.t = 0
    obs, reward, term, trunc, _ = env.step(2)  # move south → (5,5)
    assert term, "episode should terminate on catch, not run to T"
    # t == 1 after the step
    assert abs(reward - (1.0 - 1.0 / 20.0)) < 1e-6, f"got reward={reward}"


def test_terminal_miss_reward_is_negative_one():
    """If the agent never gets within R_catch, terminal reward is -1.0."""
    env = InferencePursuitEnv(N=10, T=5, p_drift=0.0, R_catch=1.0,
                              sigma0=1.0, alpha=1.0)
    env.reset(seed=0)
    env.agent_pos = [0, 0]
    env.target_pos = (5, 5)
    env.t = 0
    reward = None
    for _ in range(5):
        _, reward, term, _, _ = env.step(0)  # Inference only, no movement
    assert term
    assert reward == -1.0
