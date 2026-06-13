import numpy as np
import pytest
from options_hedging.pricing.option_contract import OptionContract
from options_hedging.pricing.black_scholes import BlackScholes

def make_contract(option_type='call', S=100.0, K=100.0, T=0.25, r=0.05, sigma=0.20):
    return OptionContract(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type)

def test_invalid_option_type():
    with pytest.raises(ValueError):
        make_contract(option_type='banana')

def test_invalid_S():
    with pytest.raises(ValueError):
        make_contract(S=-1)

def test_invalid_T():
    with pytest.raises(ValueError):
        make_contract(T=-0.1)

def test_invalid_sigma():
    with pytest.raises(ValueError):
        make_contract(sigma=-0.1)

def test_put_call_parity():
    call = BlackScholes(make_contract('call')).price()
    put = BlackScholes(make_contract('put')).price()
    S, K, T, r = 100.0, 100.0, 0.25, 0.05
    assert np.isclose(call - put, S - K * np.exp(-r * T), rtol=1e-5)

def test_deep_itm_call_delta():
    bs = BlackScholes(make_contract('call', S=200.0, K=100.0))
    assert bs.delta() > 0.99

def test_deep_otm_call_delta():
    bs = BlackScholes(make_contract('call', S=50.0, K=100.0))
    assert bs.delta() < 0.01

def test_vectorized_price_shape():
    contract = OptionContract(
        S=np.array([95.0, 100.0, 105.0]),
        K=100.0, T=0.25, r=0.05,
        sigma=np.array([0.18, 0.20, 0.22]),
        option_type='call'
    )
    price = BlackScholes(contract).price()
    assert price.shape == (3,)

def test_vectorized_delta_shape():
    contract = OptionContract(
        S=np.array([95.0, 100.0, 105.0]),
        K=100.0, T=0.25, r=0.05,
        sigma=np.array([0.18, 0.20, 0.22]),
        option_type='call'
    )
    delta = BlackScholes(contract).delta()
    assert delta.shape == (3,)

def test_gamma_positive():
    bs = BlackScholes(make_contract())
    assert bs.gamma() > 0

def test_vega_positive():
    bs = BlackScholes(make_contract())
    assert bs.vega() > 0

def test_theta_negative_call():
    bs = BlackScholes(make_contract('call'))
    assert bs.theta() < 0

def test_theta_negative_put():
    bs = BlackScholes(make_contract('put'))
    assert bs.theta() < 0

def test_rho_positive_call():
    bs = BlackScholes(make_contract('call'))
    assert bs.rho() > 0

def test_rho_negative_put():
    bs = BlackScholes(make_contract('put'))
    assert bs.rho() < 0

def test_invalid_nan_input():
    with pytest.raises(ValueError):
        OptionContract(S=float('nan'), K=100, T=0.25, r=0.05, sigma=0.20, option_type='call')

def test_invalid_inf_input():
    with pytest.raises(ValueError):
        OptionContract(S=float('inf'), K=100, T=0.25, r=0.05, sigma=0.20, option_type='call')