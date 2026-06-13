import numpy as np
import pytest
from options_hedging.analytics.pnl import pnl_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def make_errors(seed=42, n=10000):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, n)

def test_output_index():
    summary = pnl_stats(make_errors())
    expected = [
        'Mean', 'Median', 'Standard Deviation', 'Skewness',
        'Minimum', 'Maximum', 'Range', 'Lower Quartile',
        'Upper Quartile', 'Interquartile Range',
        'Value at Risk', 'Conditional VaR'
    ]
    assert list(summary.index) == expected

def test_cvar_geq_var():
    summary = pnl_stats(make_errors())
    assert summary['Conditional VaR'] >= summary['Value at Risk']

def test_range():
    summary = pnl_stats(make_errors())
    assert np.isclose(summary['Range'],
                      summary['Maximum'] - summary['Minimum'])

def test_iqr():
    summary = pnl_stats(make_errors())
    assert np.isclose(summary['Interquartile Range'],
                      summary['Upper Quartile'] - summary['Lower Quartile'])

def test_ordering():
    summary = pnl_stats(make_errors())
    assert summary['Minimum'] <= summary['Lower Quartile']
    assert summary['Lower Quartile'] <= summary['Median']
    assert summary['Median'] <= summary['Upper Quartile']
    assert summary['Upper Quartile'] <= summary['Maximum']

def test_negative_skewness_left_tail():
    rng = np.random.default_rng(0)
    errors = np.concatenate([rng.normal(0.5, 0.3, 9000),
                              rng.normal(-5, 1, 1000)])
    summary = pnl_stats(errors)
    assert summary['Skewness'] < 0

def test_alpha_parameter():
    errors = make_errors()
    s95 = pnl_stats(errors, alpha=0.05)
    s99 = pnl_stats(errors, alpha=0.01)
    assert s99['Value at Risk'] >= s95['Value at Risk']
    assert s99['Conditional VaR'] >= s95['Conditional VaR']

def test_plot_runs_without_error():
    errors = make_errors()
    summary = pnl_stats(errors)
    from options_hedging.analytics.pnl import plot_pnl_distribution
    plot_pnl_distribution(errors, summary)
    plt.close('all')