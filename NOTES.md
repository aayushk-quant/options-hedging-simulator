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