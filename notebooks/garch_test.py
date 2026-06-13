import numpy as np
import matplotlib.pyplot as plt
from options_hedging.volatility.garch import Garch

garch = Garch(omega=3.175e-6, alpha=0.07, beta=0.91, n_steps=252, n_simulations=10000, seed=42)
returns, variance = garch.garch_sim()

print("Shapes:", returns.shape, variance.shape)
print("variance[0] (should be ~constant, long-run variance):", variance[0, :5])
print("Implied long-run vol:", np.sqrt(variance[0, 0]))
print("Annualized:", np.sqrt(variance[0, 0]) * np.sqrt(252))
print("Any negative variance?", np.any(variance < 0))
print("Mean of returns (should be ~0):", returns[1:].mean())
print("Std of returns (should be ~ long-run vol):", returns[1:].std())
plt.plot(np.sqrt(variance[:, 0]))
plt.axhline(y=np.sqrt(0.00015), color='r', linestyle='--', label='Long-run vol')
plt.xlabel("Timestep")
plt.ylabel("Volatility")
plt.title("GARCH(1,1) simulated volatility — single path")
plt.legend()
plt.show()