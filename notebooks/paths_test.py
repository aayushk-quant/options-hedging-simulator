import numpy as np
import matplotlib.pyplot as plt
from options_hedging.pricing.option_contract import OptionContract
from options_hedging.volatility.garch import Garch
from options_hedging.simulation.paths import simulate_price_paths

option = OptionContract(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='call')

garch = Garch(omega=3e-6, alpha=0.07, beta=0.91, n_steps=63, n_simulations=10000, seed=42)
returns, variance = garch.garch_sim()

S0 = 100
price_paths = simulate_price_paths(option, S0, returns, variance)

print("Shape:", price_paths.shape)
print("S[0] all equal S0?", np.all(price_paths[0] == S0))
print("Mean of S_T:", price_paths[-1].mean())
print("S0 * exp(r*T):", S0 * np.exp(option.r * option.T))

plt.plot(price_paths[:, :5])
plt.axhline(y=option.K, color='r', linestyle='--', label='Strike (K)')
plt.xlabel("Timestep")
plt.ylabel("Price")
plt.title("Simulated price paths (GARCH-driven volatility)")
plt.legend()
plt.show()

import warnings

bad_option = OptionContract(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='call')
# Using n_steps=252 (mismatched with T=0.25) to trigger the warning
garch_bad = Garch(omega=3.175e-6, alpha=0.07, beta=0.91, n_steps=252, n_simulations=100, seed=1)
returns_bad, variance_bad = garch_bad.garch_sim()

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    simulate_price_paths(bad_option, 100, returns_bad, variance_bad)
    print("Warning triggered?", len(w) > 0, "-", str(w[0].message) if w else "")