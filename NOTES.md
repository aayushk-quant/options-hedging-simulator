# Development Notes

## Project setup
- Package structure: `src/options_hedging/` with submodules `pricing`,
  `volatility`, `simulation`, `hedging`, `analytics`. Chosen because the
  project has distinct sequential stages (vol model, then price paths,
  then hedging logic, then P&L analysis), each independently testable.
- Editable install via `pyproject.toml` (`pip install -e .`) so modules
  import cleanly across notebooks/tests without path hacks.
- Dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`, `yfinance`.
  Deliberately NOT using `arch` for GARCH; implementing the recursion
  from first principles is the point of this project.

## Pricing module (pricing/)
- Reused/adapted `OptionContract` and `BlackScholes` from Project 1
  (options-pricing-library), dropped `iv_solver` (not needed, since
  we're not backing out implied vol from market prices here).
- Key change: `OptionContract` validation rewritten to support
  array-valued `S` and `sigma` (needed for vectorizing across simulated
  paths). Used `np.any()`/`np.all()` with elementwise `&`/`|` instead of
  Python's `or`/`and`, since the latter can't evaluate truthiness of
  arrays.
- Design pattern for the whole project: at each timestep `t`, construct
  ONE `OptionContract` where `S` and `sigma` are arrays (one value per
  simulated path), and `K`/`T`/`r` are scalars (same for all paths at a
  given `t`). Loop over time (`n_steps ~ 252`), not over paths
  (`n_simulations ~ 10,000`), to vectorize the expensive dimension.

## Volatility module (volatility/garch.py)
- Implemented GARCH(1,1) from first principles:
  `sigma_t^2 = omega + alpha*eps_{t-1}^2 + beta*sigma_{t-1}^2`,
  `eps_t = sigma_t * z_t`, `z_t ~ N(0,1)`, `r_t = mu + eps_t`
- Validation: `alpha+beta < 1` (stationarity), `omega > 0`,
  `alpha/beta >= 0` (ensures `sigma_t^2 >= 0` always),
  `n_steps/n_simulations > 0`.
- Baseline parameters chosen for ~20% annualized vol:
  `omega=3e-6, alpha=0.07, beta=0.91, mu=0`
  (`alpha+beta=0.98`, realistic persistence per literature ranges)
- Initialization: `sigma_0^2 = omega/(1-alpha-beta)` (long-run variance),
  `eps_0 = 0`. Output arrays shaped `(n_steps+1, n_simulations)`; row 0
  is this initialization, rows `1..n_steps` are the simulated path.
  Mirrors the `(n_simulations, n_steps+1)` convention from
  `MonteCarlo._simulate_paths` in Project 1 (axes swapped to match the
  "loop over time" pattern here).
- `variance` array stores `sigma_t^2` (variance), not `sigma_t` (vol).
  `np.sqrt()` applied where vol itself is needed.
- Validated: `variance[0]` matches expected long-run variance, no
  negative variances, mean/std of returns consistent with target vol,
  and visual check of `sqrt(variance)` over time shows clear volatility
  clustering (sustained periods above/below long-run level). Confirms
  the recursion behaves as expected.

  ## Simulation module (simulation/paths.py)
- `simulate_price_paths(option_contract, S0, returns, variance)` converts
  GARCH output into price paths.
- `n_steps = returns.shape[0] - 1`, and `dt = option_contract.T / n_steps`.
  `n_steps` is chosen when constructing `Garch` to match `T` (e.g.
  `T=0.25` to `n_steps=63`), so `dt ≈ 1/252` and total simulated time
  equals `T` exactly.
- `r` comes from `option_contract.r` (no separate parameter needed).
  No `seed` parameter either: GARCH already consumed the seeded RNG when
  generating `returns`/`variance`, so there's no new randomness here.
- Drift derivation: GARCH `variance[t]` is daily variance, so
  `sigma_annual^2 * dt = variance[t]` when `dt = 1/252`. Log-return
  formula: `log_return_t = r*dt - variance[t]/2 + returns[t]`, computed
  for `t=1..n_steps` via `variance[1:]`/`returns[1:]` (row 0 is GARCH's
  initialization, not a real step, so excluded).
- Price path: `price_paths[0] = S0` (exact, for every simulation), and
  `price_paths[1:] = S0 * exp(cumsum(log_returns, axis=0))`. Output shape
  `(n_steps+1, n_simulations)`, matching the GARCH output convention.
- Stacking note: used `np.vstack` (not `np.hstack` as in Project 1's
  `MonteCarlo._simulate_paths`), because the axis convention here is
  `(time, simulations)` rather than `(simulations, time)`.
- Added a runtime `warnings.warn()` if `dt` deviates from `1/252` by more
  than `rtol=5e-2`, since `n_steps` and `T` are set independently and
  nothing else enforces `dt ≈ 1/252`.
- Validated: `price_paths[0]` exactly equals `S0` for all paths; mean of
  `S_T` closely matches `S0 * exp(r*T)` (within ~0.1% for 10,000 paths),
  confirming the risk-neutral martingale property holds. Visual check of
  individual paths shows realistic price behaviour with visibly different
  volatility regimes across paths.