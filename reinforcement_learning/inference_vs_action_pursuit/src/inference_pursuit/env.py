import math
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from inference_pursuit.geometry import torus_dist
from inference_pursuit.belief_filter import BeliefFilter

_MOVE_DELTAS = {1: (-1, 0), 2: (1, 0), 3: (0, 1), 4: (0, -1)}


class InferencePursuitEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        N: int,
        T: int,
        p_drift: float,
        R_catch: float,
        sigma0: float,
        alpha: float,
    ):
        super().__init__()
        self.N = N
        self.T = T
        self.p_drift = p_drift
        self.R_catch = R_catch
        self.sigma0 = sigma0
        self.alpha = alpha

        self.action_space = spaces.Discrete(5)

        half = N / 2.0
        obs_low = np.array([-half, -half, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([half, half, N * N / 2.0, float(T)], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self._filter = BeliefFilter(N=N, p_drift=p_drift)
        self.agent_pos: list = [0, 0]
        self.target_pos: Tuple[int, int] = (0, 0)
        self.t: int = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        N = self.N
        self.agent_pos = [int(self.np_random.integers(0, N)), int(self.np_random.integers(0, N))]
        self.target_pos = (int(self.np_random.integers(0, N)), int(self.np_random.integers(0, N)))
        self.t = 0
        self._filter.reset()
        return self._obs(), {"target_pos": self.target_pos}

    def step(self, action: int):
        # 1. Target drifts; filter predicts
        if self.p_drift > 0.0:
            self.target_pos = self._drift_target()
            self._filter.predict()

        # 2. Execute
        if action == 0:
            z = self._sample_obs()
            self._filter.update(self.agent_pos, z, self.sigma0, self.alpha)
        else:
            dx, dy = _MOVE_DELTAS[action]
            self.agent_pos[0] = (self.agent_pos[0] + dx) % self.N
            self.agent_pos[1] = (self.agent_pos[1] + dy) % self.N

        # 3. Advance time, evaluate catch
        self.t += 1
        d = torus_dist((self.agent_pos[0], self.agent_pos[1]), self.target_pos, self.N)
        if d <= self.R_catch:
            terminated = True
            reward = max(0.0, 1.0 - self.t / self.T)
        elif self.t >= self.T:
            terminated = True
            reward = -1.0
        else:
            terminated = False
            reward = 0.0

        return self._obs(), float(reward), terminated, False, {}

    def _drift_target(self) -> Tuple[int, int]:
        if self.np_random.random() < self.p_drift:
            dx, dy = [(-1, 0), (1, 0), (0, 1), (0, -1)][int(self.np_random.integers(0, 4))]
            return ((self.target_pos[0] + dx) % self.N, (self.target_pos[1] + dy) % self.N)
        return self.target_pos

    def _sample_obs(self) -> np.ndarray:
        d = torus_dist((self.agent_pos[0], self.agent_pos[1]), self.target_pos, self.N)
        d_eff = max(d, 1e-8)
        sigma = math.sqrt(self.sigma0 ** 2 * d_eff ** self.alpha)
        return self.np_random.normal(loc=np.array(self.target_pos, dtype=float), scale=sigma)

    def _obs(self) -> np.ndarray:
        (mx, my), trace = self._filter.belief_stats()
        half = self.N / 2.0
        dx = ((mx - self.agent_pos[0]) + half) % self.N - half
        dy = ((my - self.agent_pos[1]) + half) % self.N - half
        return np.array([dx, dy, trace, float(self.T - self.t)], dtype=np.float32)
