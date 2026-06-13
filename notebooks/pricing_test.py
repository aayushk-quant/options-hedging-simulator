import numpy as np
from options_hedging.pricing.option_contract import OptionContract
from options_hedging.pricing.black_scholes import BlackScholes

# Simulate 3 paths with different prices and vols at some timestep
contract = OptionContract(
    S=np.array([100.0, 105.0, 95.0]),
    K=100.0,
    T=0.25,
    r=0.05,
    sigma=np.array([0.20, 0.25, 0.18]),
    option_type='call'
)

bs = BlackScholes(contract)
print("Price:", bs.price())
print("Delta:", bs.delta())
print("Gamma:", bs.gamma())