import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

def pnl_stats(hedging_error, alpha: float = 0.05):
    mean_he = np.mean(hedging_error)
    std_he = np.std(hedging_error)
    min_he = np.min(hedging_error)
    max_he = np.max(hedging_error)
    range_he = max_he - min_he
    median_he = np.median(hedging_error)
    lower_q = np.percentile(hedging_error, 25)
    upper_q = np.percentile(hedging_error, 75)
    interquartile_r_he = upper_q - lower_q
    percentile_threshold = np.percentile(hedging_error, alpha * 100)
    VaR = -percentile_threshold
    tail = hedging_error[hedging_error <= percentile_threshold]
    CVaR = -tail.mean()
    skewness = skew(hedging_error)
    summary = pd.Series(
        [
            mean_he, 
            median_he, 
            std_he, 
            skewness, 
            min_he, 
            max_he, 
            range_he, 
            lower_q, 
            upper_q, 
            interquartile_r_he, 
            VaR, 
            CVaR],
        index = 
        [
            'Mean',
            'Median',
            'Standard Deviation',
            'Skewness',
            'Minimum',
            'Maximum',
            'Range',
            'Lower Quartile',
            'Upper Quartile',
            'Interquartile Range',
            'Value at Risk',
            'Conditional VaR'
        ])
    return summary

def plot_pnl_distribution(hedging_error, summary):
    plt.figure(figsize=(10, 6))
    plt.hist(hedging_error, bins=50, color='steelblue', edgecolor='black', alpha=0.7)

    plt.axvline(x=0, color='black', linestyle='-', linewidth=1, label='Zero error')
    plt.axvline(x=-summary['Value at Risk'], color='orange', linestyle='--',
                linewidth=2, label=f"VaR (95%): {summary['Value at Risk']:.2f}")
    plt.axvline(x=-summary['Conditional VaR'], color='red', linestyle='--',
                linewidth=2, label=f"CVaR (95%): {summary['Conditional VaR']:.2f}")

    plt.xlabel("Hedging error")
    plt.ylabel("Frequency")
    plt.title("Distribution of hedging errors with VaR / CVaR")
    plt.legend()
    plt.tight_layout()
    plt.show()