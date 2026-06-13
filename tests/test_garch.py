import numpy as np
import pytest
from options_hedging.volatility.garch import Garch

def make_garch(**kwargs):
    defaults = dict(omega=3.175e-6, alpha=0.07, beta=0.91,
                    n_steps=63, n_simulations=1000, seed=42)
    defaults.update(kwargs)
    return Garch(**defaults)

def test_invalid_stationarity():
    with pytest.raises(ValueError):
        make_garch(alpha=0.5, beta=0.6)

def test_invalid_omega():
    with pytest.raises(ValueError):
        make_garch(omega=-1e-6)

def test_invalid_alpha():
    with pytest.raises(ValueError):
        make_garch(alpha=-0.1, beta=0.5)

def test_invalid_beta():
    with pytest.raises(ValueError):
        make_garch(alpha=0.05, beta=-0.1)

def test_invalid_n_steps():
    with pytest.raises(ValueError):
        make_garch(n_steps=-1)

def test_output_shapes():
    garch = make_garch()
    returns, variance = garch.garch_sim()
    assert returns.shape == (64, 1000)
    assert variance.shape == (64, 1000)

def test_initial_variance():
    garch = make_garch()
    returns, variance = garch.garch_sim()
    long_run_var = garch.omega / (1 - garch.alpha - garch.beta)
    assert np.allclose(variance[0], long_run_var)

def test_no_negative_variance():
    garch = make_garch()
    _, variance = garch.garch_sim()
    assert not np.any(variance < 0)

def test_constant_variance_degenerate():
    garch = make_garch(omega=1e-4, alpha=0.0, beta=0.0)
    _, variance = garch.garch_sim()
    assert np.allclose(variance, 1e-4)

def test_mean_returns_near_zero():
    garch = make_garch(n_simulations=10000)
    returns, _ = garch.garch_sim()
    assert abs(returns[1:].mean()) < 1e-3

def test_seed_reproducibility():
    r1, v1 = make_garch(seed=1).garch_sim()
    r2, v2 = make_garch(seed=1).garch_sim()
    assert np.allclose(r1, r2)
    assert np.allclose(v1, v2)