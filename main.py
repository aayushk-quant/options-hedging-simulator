from options_hedging.pricing.option_contract import OptionContract
from options_hedging.volatility.garch import Garch
from options_hedging.simulation.paths import simulate_price_paths
from options_hedging.hedging.delta_hedge import DeltaHedger
from options_hedging.analytics.pnl import pnl_stats, plot_pnl_distribution

# Parameters
S0 = 100
K = 100
T = 0.25
r = 0.05
sigma = 0.20
omega = 3.175e-6
alpha = 0.07
beta = 0.91
n_steps = round(T * 252)
n_simulations = 10000
seed = 42
# Pipeline
option = OptionContract(S=S0, K=K, T=T, r=r, sigma=sigma, option_type='call')
garch = Garch(omega=omega, alpha=alpha, beta=beta, n_steps=n_steps, n_simulations=n_simulations, seed=seed)
returns, variance = garch.garch_sim()
price_paths = simulate_price_paths(option, S0, returns, variance)
hedger = DeltaHedger(option, price_paths)
hedger.run_sim()
hedging_error = hedger.get_hedging_error()
summary = pnl_stats(hedging_error)
print(summary)
plot_pnl_distribution(hedging_error, summary)