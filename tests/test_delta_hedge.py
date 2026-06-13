import numpy as np
import pytest
from options_hedging.pricing.option_contract import OptionContract
from options_hedging.volatility.garch import Garch
from options_hedging.simulation.paths import simulate_price_paths
from options_hedging.hedging.delta_hedge import DeltaHedger

def make_hedger(n_simulations=1000, seed=42, alpha=0.07, beta=0.91):
    option = OptionContract(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='call')
    n_steps = round(0.25 * 252)
    garch = Garch(omega=3.175e-6, alpha=alpha, beta=beta,
                  n_steps=n_steps, n_simulations=n_simulations, seed=seed)
    returns, variance = garch.garch_sim()
    price_paths = simulate_price_paths(option, 100, returns, variance)
    return DeltaHedger(option, price_paths)

def test_error_before_run():
    hedger = make_hedger()
    with pytest.raises(AttributeError):
        hedger.get_hedging_error()

def test_hedging_error_shape():
    hedger = make_hedger()
    hedger.run_sim()
    assert hedger.get_hedging_error().shape == (1000,)

def test_cash_delta_shapes():
    hedger = make_hedger()
    hedger.run_sim()
    n_steps = round(0.25 * 252)
    assert hedger.cash.shape == (n_steps + 1, 1000)
    assert hedger.delta.shape == (n_steps + 1, 1000)

def test_gbm_limiting_case():
    option = OptionContract(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='call')
    n_steps = round(0.25 * 252)
    omega = 0.20 ** 2 / 252
    garch = Garch(omega=omega, alpha=0.0, beta=0.0,
                  n_steps=n_steps, n_simulations=50000, seed=42)
    returns, variance = garch.garch_sim()
    price_paths = simulate_price_paths(option, 100, returns, variance)
    hedger = DeltaHedger(option, price_paths)
    hedger.run_sim()
    error = hedger.get_hedging_error()
    assert abs(error.mean()) < 0.05

def test_put_hedging_error_shape():
    option = OptionContract(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='put')
    n_steps = round(0.25 * 252)
    garch = Garch(omega=3.175e-6, alpha=0.07, beta=0.91,
                  n_steps=n_steps, n_simulations=1000, seed=42)
    returns, variance = garch.garch_sim()
    price_paths = simulate_price_paths(option, 100, returns, variance)
    hedger = DeltaHedger(option, price_paths)
    hedger.run_sim()
    assert hedger.get_hedging_error().shape == (1000,)