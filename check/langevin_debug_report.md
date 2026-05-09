# Langevin Sampling Debug Report

Date: 2026-05-08

## Background

The reported symptom was that MALA and ULA chains did not converge and their
traces looked close to Brownian motion. I checked the Langevin kernels first on
a standard normal target, then on a Cox partial likelihood with a Gaussian
prior, and finally on the existing Bayesian lasso / prox-lasso paths.

## Checked Code

- `mainfuncs/samplers/mala_sampler.jl`: MALA, ULA, hybrid beta Langevin,
  prox-lasso and prox-SSLASSO paths.
- `mainfuncs/likelihood/cox_likelihood.jl`: Cox partial likelihood and score.
- `mainfuncs/posterior.jl`: log posterior and joint gradient dispatch.
- `mainfuncs/priors/prior_common.jl`: zero-sum penalties and L1 epigraph prox.
- `mainfuncs/priors/bayesian_lasso_methods.jl`: lasso prior state and gradients.
- `mainfuncs/targets/cox_targets.jl`: explicit prox targets.
- `check/*.jl`: existing sampler and gradient checks.
- `scripts/run_simulation.cmd`, `scripts/run_simulation.jl`,
  `scripts/analysis_common.jl`: single-run simulation entry.

## Changes

- Added `check/langevin_diagnostics.jl`.
  It runs reproducible parameter scans and writes CSV metrics plus trace,
  log-posterior, and ACF plots under the requested output directory.
- Added this report: `check/langevin_debug_report.md`.
- Added `--thin` support to `scripts/run_simulation.jl` and threaded it through
  `scripts/analysis_common.jl` to the actual samplers.
- Updated `scripts/run_simulation.cmd` so its single experiment can be changed
  through environment variables. Its default Langevin test now uses
  `constraint=prox` and a conservative step size instead of the previously
  unstable `soft_center + step=1e-3`.

No core MALA/ULA proposal or acceptance-ratio bug was found.

## Main Findings

1. The project MALA kernel is correct on a standard normal target. With 10
   dimensions, 4 chains, 5000 samples, and 2000 warmup iterations, MALA produced
   means near zero, variance near one, finite traces, and acceptance near the
   expected range.
2. ULA with a very small step is not exploding on the normal target; it is just
   highly autocorrelated. For `step=0.01`, lag-1 ACF was about `0.989`, so the
   trace visually resembles Brownian motion. Larger stable ULA steps fixed this.
3. Cox Gaussian-prior gradients are numerically correct. The finite difference
   / ForwardDiff check gave max absolute error `1.78e-14`.
4. The serious instability came from combining Langevin with the default
   `soft_center` zero-sum penalty. The default scale is `0.001*p`, so the
   sum-direction curvature is `p / scale^2 = 1e6 / p`. The ULA stability limit
   for that quadratic direction is approximately `2 * scale^2 / p = 2e-6*p`.
   For `p=150`, that is about `3e-4`, while the original `.cmd` used `1e-3`.
5. Replacing `soft_center` by `constraint=prox` with
   `prox_sum0_lambda=1e-2` reduces the sum-direction curvature to about `100`,
   allowing materially larger Langevin steps.
6. Existing lasso/prox gradients are correct by AD checks. Instability is from
   stiffness and step-size mismatch, not gradient sign or MALA log-q algebra.

## Parameter Search Summary

### Standard Normal

Command:

```powershell
julia +1.11.7 --project=. check\langevin_diagnostics.jl --mode normal --normal-dim 10 --normal-samples 5000 --normal-warmup 2000 --chains 4 --normal-mala-steps 0.25,0.5,0.75 --normal-ula-steps 0.01,0.05,0.1,0.2 --outdir results\langevin_debug\formal --max-lag 50
```

Key results:

| sampler | step | max abs mean | mean variance | lag-1 ACF | min ESS | max R-hat | acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|
| MALA | 0.5 | 0.025 | 1.006 | 0.543 | 5343 | 1.001 | 0.592 |
| ULA | 0.01 | 0.188 | 0.984 | 0.989 | 99 | 1.070 | NA |
| ULA | 0.2 | 0.050 | 1.103 | 0.798 | 2060 | 1.004 | NA |

Conclusion: MALA passes. ULA needs a larger step than `0.01` on this scale;
`0.1` to `0.2` is stable for this target, with the known ULA variance bias.

### Cox + Gaussian Prior

Command:

```powershell
julia +1.11.7 --project=. check\langevin_diagnostics.jl --mode cox_gaussian --cox-n 120 --cox-p 12 --cox-nonzero 4 --cox-samples 3000 --cox-warmup 1500 --chains 4 --prior-sd 2.0 --cox-mala-steps 0.001,0.002,0.005,0.01 --cox-ula-steps 0.0005,0.001,0.002,0.005 --outdir results\langevin_debug\formal --max-lag 50
```

Key results:

| sampler | step | lag-1 ACF | min ESS | max R-hat | acceptance | mean abs error to truth |
|---|---:|---:|---:|---:|---:|---:|
| MALA | 0.002 | 0.704 | 920 | 1.005 | 0.527 | 0.076 |
| MALA | 0.01 | 0.702 | 1003 | 1.007 | 0.548 | 0.075 |
| ULA | 0.002 | 0.877 | 408 | 1.011 | NA | 0.076 |
| ULA | 0.005 | 0.681 | 1053 | 1.010 | NA | 0.077 |

Conclusion: Cox likelihood, Gaussian prior, posterior scaling, and gradient
signs are sound on this controlled problem.

### Cox + Lasso / Prox-Lasso

Commands:

```powershell
julia +1.11.7 --project=. check\langevin_diagnostics.jl --mode lasso --lasso-n 120 --lasso-p 12 --lasso-nonzero 4 --lasso-samples 2000 --lasso-warmup 1000 --chains 4 --lasso-steps 0.00001,0.0001,0.0005 --lasso-lambdas 0.25,1.0 --lasso-priors gaussian-lasso,prox-lasso --lasso-samplers mala,ula --prox-l1-lambda 0.01 --prox-sum0-lambda 0.01 --outdir results\langevin_debug\formal --max-lag 50

julia +1.11.7 --project=. check\langevin_diagnostics.jl --mode lasso --lasso-n 120 --lasso-p 12 --lasso-nonzero 4 --lasso-samples 2000 --lasso-warmup 1000 --chains 4 --lasso-steps 0.001,0.002,0.005 --lasso-lambdas 1.0 --lasso-priors prox-lasso --lasso-samplers mala,ula --prox-l1-lambda 0.01 --prox-sum0-lambda 0.01 --outdir results\langevin_debug\formal_prox_wide --max-lag 50

julia +1.11.7 --project=. check\langevin_diagnostics.jl --mode lasso --lasso-n 120 --lasso-p 12 --lasso-nonzero 4 --lasso-samples 2000 --lasso-warmup 1000 --chains 4 --lasso-steps 0.01,0.02 --lasso-lambdas 1.0 --lasso-priors prox-lasso --lasso-samplers ula --prox-l1-lambda 0.01 --prox-sum0-lambda 0.01 --outdir results\langevin_debug\formal_prox_ula_high --max-lag 50
```

Key results:

| prior / constraint | sampler | step | lambda | max state | lag-1 ACF | min ESS | max R-hat | acceptance |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| lasso + soft_center | ULA | 1e-4 | 1.0 | `9.3e150` | 0.9995 | 16 | `5.34e15` | NA |
| lasso + soft_center | ULA | 1e-5 | 1.0 | 0.77 | 0.996 | 17 | 2.60 | NA |
| prox-lasso + prox | MALA | 0.001 | 1.0 | 1.39 | 0.727 | 481 | 1.011 | 0.589 |
| prox-lasso + prox | MALA | 0.005 | 1.0 | 1.41 | 0.737 | 455 | 1.014 | 0.549 |
| prox-lasso + prox | ULA | 0.005 | 1.0 | 1.44 | 0.654 | 704 | 1.007 | NA |
| prox-lasso + prox | ULA | 0.01 | 1.0 | 1.64 | 0.170 | 1828 | 1.004 | NA |
| prox-lasso + prox | ULA | 0.02 | 1.0 | 6059 | -0.976 | 513 | 1.776 | NA |

Conclusion: `soft_center` is the immediate cause of ULA blow-up for Langevin.
Prox-lasso is stable with `prox_sum0_lambda=1e-2`, `prox_l1_lambda=1e-2`.
For this n=120, p=12 case, ULA `0.005` to `0.01` and MALA initial
`0.001` to `0.005` are reproducibly stable.

## Existing Simulation Entry

The `.cmd` entry was validated after modification. Example PowerShell command:

```powershell
$env:test_sampler='mala'; $env:test_samples='5000'; $env:test_warmup='2000'; $env:test_thin='1'; $env:test_chains='4'; $env:test_n='200'; $env:test_p='50'; $env:test_nonzero='6'; $env:test_priors='lasso'; $env:test_constraint='prox'; $env:test_mala_step_size='1e-3'; $env:test_progress='false'; $env:test_diagnose='true'; cmd /c scripts\run_simulation.cmd
```

Output:

- `results/simulation/test_n200_p50_non6_weibull_20260508_225711`
- acceptance rate `0.55`
- warmup acceptance rate `0.5775`
- final beta/prox step by chain about `0.001`
- no NaN/Inf or runtime failure

For p=50 lasso, 5000 samples were still not enough for all beta R-hats to be
near 1.01. This is a high-dimensional lasso mixing limitation, not the
Brownian-motion instability. ULA at `step=0.002` improved R-hat compared with
`0.0005` but showed more approximation bias; `step=0.005` mixed faster but
hurt MSE badly in the p=50 simulation.

## Recommended Defaults

For Langevin runs in this project:

- Use `constraint=prox`, not `soft_center`, for MALA/ULA.
- Use `prox_sum0_lambda=1e-2` and `prox_l1_lambda=1e-2` as a stable starting
  point.
- For standard normal sanity checks: MALA `step=0.5`; ULA `step=0.1` to `0.2`.
- For Cox + Gaussian prior at moderate p: MALA initial `0.002` to `0.01`;
  ULA `0.002` to `0.005`.
- For prox-lasso at p around 10-20: MALA initial `1e-3` to `5e-3`; ULA
  `0.005`, increasing to `0.01` only if diagnostics remain stable.
- For p around 50 or larger: prefer MALA over ULA; start MALA at `1e-3`,
  use at least 4 chains, 2000+ warmup, and 5000+ saved samples. Treat ULA as
  approximate and check MSE/log-posterior/PSIS diagnostics before using it.
- If `soft_center` must be used with default scale `0.001*p`, keep ULA
  `step_size < 2e-6*p`; this is why `p=150, step=1e-3` is unstable.

## Risks And Limits

- ULA is not Metropolis corrected. Larger ULA steps improve mixing but introduce
  discretization bias; this was visible in the p=50 simulation.
- High-dimensional Bayesian lasso remains slow to mix even after removing the
  soft-center stiffness. NUTS explicit remains the better reference sampler
  when runtime allows.
- I did not change the core MALA proposal or acceptance implementation because
  the standard normal, Cox Gaussian-prior, and AD gradient checks did not show
  an implementation bug.

## Verification

Package tests:

```powershell
julia +1.11.7 --project=. -e "using Pkg; Pkg.test()"
```

Result: all tests passed.

