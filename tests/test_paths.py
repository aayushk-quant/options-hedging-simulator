import numpy as np
import pytest
import warnings
from options_hedging.pricing.option_contract import OptionContract
from options_hedging.volatility.garch import Garch
from options_hedging.simulation.paths import simulate_price_paths

def make_inputs(T=0.25, n_simulations=1000, seed=42):
    option = OptionContract(S=100, K=100, T=T, r=0.05, sigma=0.20, option_type='call')
    n_steps = round(T * 252)
    garch = Garch(omega=3.175e-6, alpha=0.07, beta=0.91,
                  n_steps=n_steps, n_simulations=n_simulations, seed=seed)
    returns, variance = garch.garch_sim()
    return option, returns, variance

def test_output_shape():
    option, returns, variance = make_inputs()
    paths = simulate_price_paths(option, 100, returns, variance)
    assert paths.shape == (64, 1000)

def test_initial_price():
    option, returns, variance = make_inputs()
    paths = simulate_price_paths(option, 100, returns, variance)
    assert np.all(paths[0] == 100.0)

def test_risk_neutral_martingale():
    option, returns, variance = make_inputs(n_simulations=50000, seed=0)
    paths = simulate_price_paths(option, 100, returns, variance)
    expected = 100 * np.exp(0.05 * 0.25)
    assert np.isclose(paths[-1].mean(), expected, rtol=0.01)

def test_dt_warning():
    option = OptionContract(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='call')
    garch = Garch(omega=3.175e-6, alpha=0.07, beta=0.91,
                  n_steps=252, n_simulations=100, seed=42)
    returns, variance = garch.garch_sim()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        simulate_price_paths(option, 100, returns, variance)
        assert len(w) == 1

def test_positive_prices():
    option, returns, variance = make_inputs()
    paths = simulate_price_paths(option, 100, returns, variance)
    assert np.all(paths > 0)