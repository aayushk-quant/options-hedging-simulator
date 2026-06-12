import numpy as np

class Garch:
    def __init__(self, omega, alpha, beta, mu = 0, n_steps: int = 252, n_simulations: int = 10000, seed: int = None):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.n_steps = n_steps
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(seed)
        if alpha + beta >= 1:
            raise ValueError('Alpha and Beta must add to <1')
        if n_simulations <= 0 or n_steps <= 0:
            raise ValueError('Number of steps and simulations must be a positive number')
        if omega <= 0:
            raise ValueError('Omega must be positive')
        if alpha < 0 or beta < 0:
            raise ValueError('Alpha and Beta must be non-negative')
    
    def garch_sim(self):
        returns = np.zeros((self.n_steps + 1, self.n_simulations))
        variance = np.zeros((self.n_steps + 1, self.n_simulations))
        vol_zero= np.sqrt(self.omega / (1 - self.alpha - self.beta))
        ret_zero = self.mu
        prev_eps_sq = 0
        prev_sigma_sq = vol_zero ** 2
        returns[0] = ret_zero
        variance[0] = vol_zero ** 2
        Z = self.rng.standard_normal((self.n_steps, self.n_simulations))

        for t in range(1, self.n_steps + 1):
            variance[t] = self.omega + self.alpha * prev_eps_sq + self.beta * prev_sigma_sq
            eps_t = np.sqrt(variance[t]) * Z[t-1]
            returns[t] = self.mu + eps_t
            prev_eps_sq = eps_t ** 2
            prev_sigma_sq = variance[t]
        return returns, variance

