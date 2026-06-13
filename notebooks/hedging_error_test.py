import numpy as np
import matplotlib.pyplot as plt
from options_hedging.pricing.option_contract import OptionContract
from options_hedging.volatility.garch import Garch
from options_hedging.simulation.paths import simulate_price_paths
from options_hedging.hedging.delta_hedge import DeltaHedger
from options_hedging.analytics.pnl import pnl_stats
from options_hedging.analytics.pnl import plot_pnl_distribtion

option = OptionContract(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='call')

garch = Garch(omega=3.175e-6, alpha=0.07, beta=0.91, n_steps=63, n_simulations=10000, seed=42)
returns, variance = garch.garch_sim()

price_paths = simulate_price_paths(option, 100, returns, variance)

hedger = DeltaHedger(option, price_paths)
hedger.run_sim()
hedging_error = hedger.get_hedging_error()

print("Shape:", hedging_error.shape)
print("Mean hedging error:", hedging_error.mean())
print("Std hedging error:", hedging_error.std())
print("Min/Max:", hedging_error.min(), hedging_error.max())

plt.hist(hedging_error, bins=50)
plt.axvline(x=0, color='r', linestyle='--', label='Zero error')
plt.xlabel("Hedging error")
plt.ylabel("Frequency")
plt.title("Distribution of hedging errors (10,000 simulations)")
plt.legend()
#plt.show()
summary = pnl_stats(hedging_error)
print(summary)
plot_pnl_distribtion(hedging_error, summary)