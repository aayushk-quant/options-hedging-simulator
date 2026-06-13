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

  ## Hedging module (hedging/delta_hedge.py)
- `DeltaHedger(option_contract, price_paths)`, with `run_sim()` and
  `get_hedging_error()`. Same input pattern as `paths.py`: `S0`, `r`, and
  `seed` all redundant (`price_paths[0]=S0`, `r=option_contract.r`,
  randomness already consumed by GARCH).
- Implements a discrete self-financing delta-hedge replication. At each
  step, hold `delta[t]` shares (computed via `BlackScholes` using
  `option_contract.sigma`, the fixed 20% assumption, not the true GARCH
  vol), with the remainder in a cash account earning `r`.
- `replace(option_contract, S=price_paths[t], T=T_remaining)` builds a
  per-timestep contract for the `BlackScholes` delta calculation, with
  `T_remaining = option_contract.T - t*dt`.
- `cash[0] = V0 - delta[0]*S0`, where `V0` is the option premium at
  `t=0`. Recursive update for `t=1..n_steps-1`:
  `cash[t] = cash[t-1]*exp(r*dt) - (delta[t]-delta[t-1])*price_paths[t]`.
- At `t=n_steps` (expiry), `delta[n_steps]=delta[n_steps-1]` (no new BS
  computation, since `T_remaining=0` breaks the formula). With this, the
  same `cash` formula collapses to pure growth (`delta` difference is
  zero), so no separate cash formula is needed for expiry.
- `get_hedging_error()`: payoff via `np.maximum(price_paths[-1]-K, 0)`
  (call) or `np.maximum(K-price_paths[-1], 0)` (put). Hedging error
  = `(cash[n_steps] + delta[n_steps]*price_paths[n_steps]) - payoff`,
  one value per simulation.
- Validated with `K=S0=100, T=0.25, r=0.05, sigma=0.20` (V0≈4.615,
  matching the earlier pricing module check). Result: mean hedging error
  ≈0.264 (~6% of V0), std≈0.935 (~20% of V0), strongly asymmetric:
  max ≈2.96 but min ≈-11.19 (over 2x V0). The long left tail corresponds
  to paths that pass through GARCH high-volatility clusters, where the
  fixed-sigma delta under-reacts to actual price moves, producing large
  losses on those specific paths. This asymmetry is the direct result of
  the sigma mismatch (fixed assumption vs realized GARCH vol) the project
  was designed to surface.

  ## Analytics module (analytics/pnl.py)
- `pnl_stats(hedging_error, alpha=0.05)` returns a labeled `pd.Series`:
  `Mean`, `Median`, `Standard Deviation`, `Skewness`, `Minimum`, `Maximum`,
  `Range`, `Lower Quartile`, `Upper Quartile`, `Interquartile Range`,
  `Value at Risk`, `Conditional VaR`.
- `alpha` is the tail probability (default 0.05 = "95% VaR/CVaR"
  convention). `percentile_threshold = np.percentile(hedging_error,
  alpha*100)` is the raw (signed, negative) tail value; `VaR =
  -percentile_threshold` flips it to a positive "loss" number. `CVaR`
  uses `percentile_threshold` (not `VaR`) for the boolean mask
  (`hedging_error <= percentile_threshold`), then takes the mean of that
  tail and negates it. Keeping the raw signed threshold separate from the
  negated "loss" value avoids a sign error in the mask.
- `skewness` via `scipy.stats.skew`, included since the hedging-error
  distribution's asymmetry is the project's central finding.
- `plot_pnl_distribution(hedging_error, summary)` is a separate function
  from `pnl_stats` (pure data computation vs visualization), taking the
  already-computed `summary` to annotate `VaR`/`CVaR` lines rather than
  recomputing them. Plots a histogram with vertical lines at zero error,
  `-summary['Value at Risk']`, and `-summary['Conditional VaR']`
  (negated back to hedging-error sign convention for the x-axis).
- Validated on the `K=S0=100, T=0.25, r=0.05, sigma=0.20` case: `Mean
  ≈0.264`, `Median ≈0.337` (median > mean, consistent with left skew),
  `Skewness ≈-1.38` (strongly negative, confirming the long left tail
  numerically), `VaR ≈1.31`, `CVaR ≈2.25`. The CVaR/VaR ratio (~1.7x)
  quantifies the tail severity visible in the histogram: beyond the 5th
  percentile, average outcomes are considerably worse than the percentile
  threshold itself, driven by extreme paths (min ≈-11.19) passing through
  GARCH high-volatility clusters.