import math
import numpy as np


class BeliefFilter:
    def __init__(self, N: int, p_drift: float):
        self.N = N
        self.p_drift = float(p_drift)
        self._stay = 1.0 - self.p_drift
        self._k = self.p_drift / 4.0

        self.pmf = np.full((N, N), 1.0 / (N * N), dtype=np.float64)

        coords = np.arange(N, dtype=np.float64)
        self._coords = coords
        self._angles = 2.0 * math.pi * coords / N
        self._sin_a = np.sin(self._angles)
        self._cos_a = np.cos(self._angles)

        d1 = np.minimum(coords, N - coords)
        self._base_d = np.sqrt(d1[:, None] ** 2 + d1[None, :] ** 2)
        self._i_idx = coords[:, None]
        self._j_idx = coords[None, :]

    def reset(self) -> None:
        self.pmf.fill(1.0 / (self.N * self.N))

    def predict(self) -> None:
        """Diffuse PMF by the drift kernel on the toroidal grid."""
        if self.p_drift == 0.0:
            return
        p = self.pmf
        self.pmf = (
            self._stay * p
            + self._k * (np.roll(p, 1, 0) + np.roll(p, -1, 0)
                         + np.roll(p, 1, 1) + np.roll(p, -1, 1))
        )

    def update(
        self,
        agent_pos,
        obs_z: np.ndarray,
        sigma0: float,
        alpha: float,
    ) -> None:
        """Bayesian update given observation z ~ N(x_t, sigma0^2 * d^alpha * I)."""
        N = self.N
        ax = int(agent_pos[0])
        ay = int(agent_pos[1])

        # Distance from agent to each cell, on the torus, via cached base grid shifted by agent pos.
        d = np.roll(self._base_d, (ax, ay), axis=(0, 1))

        # Clamp d to a tiny epsilon (matches _sample_obs); avoids 0^alpha pathologies
        # and the agent-cell Dirac-delta -inf issue when the target sits on the agent.
        d_safe = np.maximum(d, 1e-8)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            sigma2 = sigma0 * sigma0 * d_safe ** alpha
            diff_sq = (obs_z[0] - self._i_idx) ** 2 + (obs_z[1] - self._j_idx) ** 2
            log_lik = -0.5 * diff_sq / sigma2 - np.log(2.0 * math.pi * sigma2)

        # Subtract per-grid max for numerical stability before exp.
        m = log_lik.max()
        if not np.isfinite(m):
            # All entries are -inf / nan; bail to uniform.
            self.pmf.fill(1.0 / (N * N))
            return
        log_lik -= m

        unnorm = self.pmf * np.exp(log_lik)
        total = unnorm.sum()
        if not (total > 0.0 and np.isfinite(total)):
            self.pmf.fill(1.0 / (N * N))
        else:
            self.pmf = unnorm / total

    def belief_stats(self):
        """Return (mean_x, mean_y), trace_cov computed from one pass over the PMF."""
        N = self.N
        pmf_x = self.pmf.sum(axis=1)
        pmf_y = self.pmf.sum(axis=0)

        mx = (
            math.atan2(float(pmf_x @ self._sin_a), float(pmf_x @ self._cos_a))
            * N / (2.0 * math.pi)
        ) % N
        my = (
            math.atan2(float(pmf_y @ self._sin_a), float(pmf_y @ self._cos_a))
            * N / (2.0 * math.pi)
        ) % N

        d1x = np.abs(self._coords - mx)
        d1x = np.minimum(d1x, N - d1x)
        d1y = np.abs(self._coords - my)
        d1y = np.minimum(d1y, N - d1y)

        # trace = sum_{i,j} pmf[i,j] * (d1x[i]^2 + d1y[j]^2)
        trace = float(pmf_x @ (d1x * d1x) + pmf_y @ (d1y * d1y))
        return (mx, my), trace

    def belief_mean(self) -> np.ndarray:
        (mx, my), _ = self.belief_stats()
        return np.array([mx, my], dtype=np.float64)

    def trace_cov(self) -> float:
        _, trace = self.belief_stats()
        return trace
