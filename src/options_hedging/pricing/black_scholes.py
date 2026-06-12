import numpy as np
from scipy.stats import norm
from .option_contract import OptionContract

class BlackScholes:
    def __init__(self, option: OptionContract):
        self.option = option
    def _calculate_d1_d2(self):
        S, K, T, r, sigma = self.option.S, self.option.K, self.option.T, self.option.r, self.option.sigma
        d1 = (np.log(S / K)+ (r + ((sigma ** 2) / 2)) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    def price(self):
        d1, d2 = self._calculate_d1_d2()
        S, K, T, r = self.option.S, self.option.K, self.option.T, self.option.r
        if self.option.option_type == 'call':
            price = norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r * T)
        else:
            price = norm.cdf(-d2) * K * np.exp(-r * T) - norm.cdf(-d1) * S
        return price
    def delta(self):
        d1 = self._calculate_d1_d2()[0]
        if self.option.option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        return delta
    def gamma(self):
        d1 = self._calculate_d1_d2()[0]
        S, T, sigma = self.option.S,  self.option.T, self.option.sigma
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma
    def vega(self):
        d1 = self._calculate_d1_d2()[0]
        S, T = self.option.S, self.option.T
        vega = (S * norm.pdf(d1)) * np.sqrt(T)
        return vega / 100 #per 1% change in volatility
    def theta(self):
        d1, d2 = self._calculate_d1_d2()
        S, K, T, r, sigma = self.option.S, self.option.K, self.option.T, self.option.r, self.option.sigma
        if self.option.option_type == 'call':
            theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        return theta / 365 #daily theta rate
    def rho(self):
        d2 = self._calculate_d1_d2()[1]
        K, T, r = self.option.K, self.option.T, self.option.r
        if self.option.option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        return rho / 100 #per 1% change in risk-free interest rate