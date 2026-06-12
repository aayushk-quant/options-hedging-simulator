import numpy as np
from dataclasses import dataclass

@dataclass
class OptionContract:
    S: float | np.ndarray
    K: float | np.ndarray
    T: float | np.ndarray
    r: float | np.ndarray
    sigma: float | np.ndarray
    option_type: str # 'call' or 'put'
    
    def __post_init__(self):
        self.option_type = self.option_type.lower()
        if self.option_type not in ('call', 'put'):
            raise ValueError("option_type must be 'call' or 'put'")
        if np.any((self.S <= 0) | (self.K <= 0) | (self.T <= 0) | (self.sigma <= 0)):
            raise ValueError("All S, K, T, and sigma must be positive")
        if not np.all(
            np.isfinite(self.S) & 
            np.isfinite(self.K) & 
            np.isfinite(self.T) &
            np.isfinite(self.r) &
            np.isfinite(self.sigma)
            ):
            raise ValueError("All inputs must be finite numbers")
    