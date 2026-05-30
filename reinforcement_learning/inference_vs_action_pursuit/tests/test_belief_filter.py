import math
import numpy as np
import pytest
from inference_pursuit.belief_filter import BeliefFilter


def test_uniform_init():
    f = BeliefFilter(N=10, p_drift=0.1)
    assert f.pmf.shape == (10, 10)
    np.testing.assert_allclose(f.pmf.sum(), 1.0, atol=1e-10)
    np.testing.assert_allclose(f.pmf, np.full((10, 10), 1 / 100))


def test_predict_normalised():
    f = BeliefFilter(N=10, p_drift=0.3)
    f.pmf[:] = 0.0
    f.pmf[5, 5] = 1.0
    f.predict()
    np.testing.assert_allclose(f.pmf.sum(), 1.0, atol=1e-10)


def test_predict_spreads_mass():
    f = BeliefFilter(N=10, p_drift=0.4)
    f.pmf[:] = 0.0
    f.pmf[5, 5] = 1.0
    f.predict()
    assert f.pmf[5, 5] < 1.0
    for nx, ny in [(4, 5), (6, 5), (5, 4), (5, 6)]:
        assert f.pmf[nx, ny] > 0.0


def test_predict_wraps_torus():
    f = BeliefFilter(N=10, p_drift=0.4)
    f.pmf[:] = 0.0
    f.pmf[0, 0] = 1.0
    f.predict()
    assert f.pmf[9, 0] > 0.0  # North of (0,0) wraps to row 9


def test_update_sharpens():
    f = BeliefFilter(N=10, p_drift=0.1)
    before = -np.sum(f.pmf * np.log(f.pmf + 1e-12))
    f.update(agent_pos=(5, 5), obs_z=np.array([3.0, 4.0]), sigma0=1.0, alpha=1.0)
    after = -np.sum(f.pmf * np.log(f.pmf + 1e-12))
    assert after < before


def test_update_normalised():
    f = BeliefFilter(N=10, p_drift=0.1)
    f.update(agent_pos=(5, 5), obs_z=np.array([3.0, 4.0]), sigma0=1.0, alpha=1.0)
    np.testing.assert_allclose(f.pmf.sum(), 1.0, atol=1e-10)


def test_trace_cov_decreases_after_update():
    f = BeliefFilter(N=10, p_drift=0.1)
    before = f.trace_cov()
    f.update(agent_pos=(5, 5), obs_z=np.array([5.0, 5.0]), sigma0=1.0, alpha=1.0)
    assert f.trace_cov() < before


def test_belief_mean_shape():
    f = BeliefFilter(N=10, p_drift=0.1)
    assert f.belief_mean().shape == (2,)


def test_reset():
    f = BeliefFilter(N=10, p_drift=0.1)
    f.pmf[:] = 0.0
    f.pmf[3, 3] = 1.0
    f.reset()
    np.testing.assert_allclose(f.pmf, np.full((10, 10), 1 / 100))
