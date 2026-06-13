import numpy as np
import warnings

def simulate_price_paths(option_contract, S0, returns, variance):
    n_steps = returns.shape[0] - 1
    dt = option_contract.T / n_steps
    if not np.isclose(dt, 1/252, rtol = 5e-2):
        warnings.warn('dt is far from being per day so results may be inaccurate')
    r = option_contract.r
    log_returns = r * dt - variance[1:] / 2 + returns[1:]
    S0_row = np.full((1, returns.shape[1]), S0)
    price_paths = np.vstack([S0_row, S0 * np.exp(np.cumsum(log_returns, axis=0))])
    return price_paths