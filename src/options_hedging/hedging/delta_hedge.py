import numpy as np
from dataclasses import replace
from options_hedging.pricing.option_contract import OptionContract
from options_hedging.pricing.black_scholes import BlackScholes

class DeltaHedger:
    def __init__(self, option_contract: OptionContract, price_paths):
        self.option_contract = option_contract
        self.price_paths = price_paths
    def run_sim(self):
        r = self.option_contract.r
        n_steps = self.price_paths.shape[0] - 1
        n_simulations = self.price_paths.shape[1]
        dt = self.option_contract.T / n_steps
        delta = np.zeros((n_steps + 1, n_simulations))
        cash = np.zeros((n_steps + 1, n_simulations))
        opt_0 = replace(self.option_contract, S = self.price_paths[0], T = self.option_contract.T) 
        delta[0] = BlackScholes(opt_0).delta()
        V0 = BlackScholes(opt_0).price()
        cash[0] = V0 - delta[0] * self.price_paths[0]
        for t in range(1, n_steps):
            T_remaining = self.option_contract.T - t*dt
            opt_t = replace(self.option_contract, S = self.price_paths[t], T = T_remaining)
            delta[t] = BlackScholes(opt_t).delta()
            cash[t] = cash[(t - 1)] * np.exp(r * dt) - (delta[t]-delta[(t - 1)]) * self.price_paths[t]
        delta[n_steps] = delta[n_steps - 1]
        cash[n_steps] = cash[n_steps - 1] * np.exp(r * dt)
        self.cash = cash
        self.delta = delta
        self.n_steps = n_steps
        self.dt = dt
    def get_hedging_error(self):
        if not hasattr(self, 'cash'):
            raise AttributeError('To get hedging error run simulation first')
        n_steps = self.n_steps
        cash = self.cash
        delta = self.delta
        price_paths = self.price_paths
        if self.option_contract.option_type == 'call':
            payoff = np.maximum(price_paths[-1] - self.option_contract.K, 0)
        else:
            payoff = np.maximum(self.option_contract.K - price_paths[-1], 0)
        hedging_error = (cash[n_steps] + delta[n_steps] * price_paths[n_steps]) - payoff
        return hedging_error