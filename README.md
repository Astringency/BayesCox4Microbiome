# BayesCox4Micro

BayesCox4Micro is a Julia implementation of Bayesian Cox proportional hazards
models for microbiome abundance data. It provides compositional preprocessing,
sparse Cox priors, optional CAR smoothing across taxa, several MCMC backends,
and scripts for simulation and real-data analysis.

## Quick Start

Run commands from the repository root with the project environment activated:

```powershell
julia --project=. -e "using Pkg; Pkg.instantiate()"
julia --project=. -e "using Pkg; Pkg.test()"
```

Fit a small simulation:

```powershell
julia --project=. scripts\run_simulation.jl --sampler mh --samples 50 --warmup 20 --reps 1 --n 200 --p 50 --nonzero 6 --priors hs,cox-rhs,lasso --p0 truth
```

Analyze the bundled T1D data:

```powershell
julia --project=. scripts\analyze_real_data.jl --input data\T1D.csv --sampler nuts --samples 500 --warmup 200 --priors hs,cox-rhs,lasso,sslasso
```

Use Julia threads for multi-chain runs:

```powershell
$env:JULIA_NUM_THREADS = "4"
julia --project=. scripts\run_simulation.jl --sampler nuts --chains 4
```

## Core API

The top-level API accepts an `n x p` nonnegative abundance matrix, survival
times, and 0/1 event indicators:

```julia
using BayesCox4Micro

prior = cox_paper_prior("cox-rhs"; X = X, time = time, event = event)

fit = sample_bayes_cox(
    X,
    time,
    event;
    prior = prior,
    sampler = :nuts,
    constraint = :soft_center,
)
```

By default `sample_bayes_cox` applies closure, adds a pseudocount, and takes a
log transform:

```julia
prepare_cox_data(X, time, event;
    transform = :log,
    pseudocount = 1e-6,
    closure = true,
    standardize = false)
```

Posterior helpers:

```julia
draws = posterior_beta_draws(fit)
summary = posterior_beta_summary(fit)
chains = result_to_chains(fit)
save_run_outputs(fit; output_dir = "results/runs/example")
diagnose_run("results/runs/example")
```

## Priors

Use the prior name to select the prior implementation. In particular,
`lasso` is the ordinary Bayesian LASSO, while `prox-lasso` selects the
proximal L1 implementation. This choice is independent of `constraint`.

| Name | Construction | Notes |
|---|---|---|
| `hs` | `cox_paper_prior("hs"; ...)` | Standard horseshoe. |
| `rhs` | `cox_paper_prior("rhs"; ...)` | Regularized horseshoe with linear `tau0`. |
| `new-rhs` | `cox_paper_prior("new-rhs"; ...)` | Regularized horseshoe with fixed `pseudo_sigma = 1`. |
| `est-rhs` | `cox_paper_prior("est-rhs"; X, time, event, ...)` | Regularized horseshoe with `pseudo_sigma` estimated from a preliminary Cox fit. |
| `cox-hs` | `cox_paper_prior("cox-hs"; ...)` | Horseshoe with Cox-specific `tau0`. |
| `cox-rhs` | `cox_paper_prior("cox-rhs"; ...)` | Regularized horseshoe with Cox-specific `tau0`. |
| `lasso` | `cox_paper_prior("lasso")` | Bayesian LASSO using the normal-exponential mixture representation. |
| `sslasso` | `cox_paper_prior("sslasso"; realdata = false/true)` | Spike-and-slab LASSO. |
| `prox-lasso` | `cox_paper_prior("prox-lasso")` | Proximal LASSO using L1 epigraph Moreau smoothing. |
| `prox-sslasso` | `cox_paper_prior("prox-sslasso"; realdata = false/true)` | Proximal spike-and-slab LASSO with Gibbs updates for inclusion indicators. |

For HS/RHS-style priors, `p0` controls the expected number of nonzero
coefficients used in the `tau0` formulas. If `p0` is omitted, the scripts and
`cox_paper_prior` estimate it by univariate Cox screening and clamp it to
`1:(p - 1)`.

Supported script prior names:

```text
hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso,prox-lasso,prox-sslasso
```

## Constraint Semantics

`constraint` only controls the zero-sum coefficient constraint. It does not
select the L1 prior implementation.

| Value | Effect |
|---|---|
| `:soft_center` | Adds a Gaussian penalty on `sum(beta)`. If `soft_center_scale` is omitted, the scale is `0.001 * p`. |
| `:prox` | Adds the zero-sum Moreau-Yosida penalty using `prox_sum0_lambda`. |
| `:none` | Disables the zero-sum penalty. |

The command-line spelling is:

```powershell
--constraint soft_center
--constraint prox
--constraint none
```

Examples:

```powershell
# Ordinary BLASSO with a proximal zero-sum constraint
julia --project=. scripts\run_simulation.jl --priors lasso --constraint prox

# Proximal LASSO with the default soft-centered zero-sum constraint
julia --project=. scripts\run_simulation.jl --priors prox-lasso --constraint soft_center

# Proximal LASSO plus proximal zero-sum constraint
julia --project=. scripts\run_simulation.jl --priors prox-lasso --constraint prox
```

L1 proximal controls are used only by `prox-lasso` and `prox-sslasso`:

```text
--prox-l1-lambda
--prox-root-tol
--prox-root-maxiter
```

Zero-sum proximal control is used only when `--constraint prox`:

```text
--prox-sum0-lambda
```

## Samplers

| Symbol | Description |
|---|---|
| `:nuts` | NUTS with `nuts_backend = :turing` or `:explicit`. |
| `:nuts_turing` | Alias for `sampler = :nuts, nuts_backend = :turing`. |
| `:nuts_explicit` | Alias for `sampler = :nuts, nuts_backend = :explicit`. The typo `:nuts_explict` is still accepted. |
| `:mh` | Adaptive random-walk Metropolis. Proximal L1 priors are not implemented for this sampler. |
| `:mala` | MALA for continuous states, including `prox-lasso` and `prox-sslasso`. |
| `:ula` | Unadjusted Langevin updates using `mala_step_size`, including proximal L1 priors. |
| `:gibbs` | Gibbs / Metropolis-within-Gibbs. SSLASSO uses spike/slab Gibbs updates; `prox-sslasso` uses the proximal block path. |

Turing NUTS supports threaded chains through `Turing.MCMCThreads()` when
`chains > 1`. Explicit NUTS runs chain-level threading by default when
`chains > 1`; pass `parallel_chains = false` to force serial chains. Built-in
MH/MALA/ULA/Gibbs use `multichains = true` for chain-level threading.

## CAR Smoothing

When `use_car = true`, CAR smoothing is applied to HS/RHS local log-scales.
The scripts only pass CAR weights to prior names containing `hs`.

Supported `car_distance` values:

```text
bray,jaccard,jsd
```

## Simulation Script

Run:

```powershell
julia --project=. scripts\run_simulation.jl [options]
```

Common options:

| Option | Default | Description |
|---|---:|---|
| `--outdir` | `results/simulation` | Output directory. |
| `--reps` | `1` | Number of simulation repetitions. |
| `--n` | `200` | Sample size. Comma-separated grids are supported. |
| `--p` | `50` | Number of taxa/covariates. Comma-separated grids are supported. |
| `--nonzero` | `6` | Number of true nonzero coefficients. Supported values are `6` and `12`. |
| `--baseline` | `weibull` | `weibull`, `exponential`, or `gompertz`. |
| `--priors` | `hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso` | Comma-separated prior names. |
| `--sampler` | `nuts` | `nuts`, `nuts_turing`, `nuts_explicit`, `mh`, `mala`, `ula`, or `gibbs`. |
| `--nuts-backend` | `turing` | `turing` or `explicit` when `--sampler nuts`. |
| `--samples` | `500` | Saved posterior samples per fit. |
| `--warmup` | `200` | Warmup/adaptation iterations. |
| `--thin` | `1` | MCMC thinning before in-memory summaries. |
| `--chains` | `4` | Number of chains. |
| `--multichains` | `false` | Parallelize non-NUTS chains. |
| `--target-accept` | `0.8` | Target acceptance for NUTS or adaptive non-NUTS samplers. |
| `--seed` | `1` | Base random seed. |
| `--use-car` | `true` | Use CAR smoothing for HS/RHS-style priors. |
| `--car-distance` | `bray` | CAR distance. |
| `--p0` | `auto` | `auto`, numeric, or `truth` for simulation. |
| `--p0-alpha` | `0.05` | Screening threshold for automatic `p0`. |
| `--constraint` | `soft_center` | Zero-sum constraint mode. |
| `--soft-center-scale` | automatic | Scale for `soft_center`. |
| `--prox-sum0-lambda` | `1e-3` | Zero-sum Moreau smoothing parameter. |
| `--prox-l1-lambda` | `1e-3` | L1 Moreau smoothing parameter for proximal L1 priors. |
| `--prox-root-tol` | `1e-10` | Projection root tolerance for proximal L1 priors. |
| `--prox-root-maxiter` | `100` | Projection root iteration limit for proximal L1 priors. |
| `--mala-step-size` | `1e-4` | MALA/ULA step size. |
| `--mala-target-accept` | `0.574` | MALA step-size adaptation target. |
| `--mala-adapt-every` | `50` | MALA adaptation interval. |
| `--diagnose` | `true` | Write diagnostic tables for every repetition; diagnostic plots are generated only for plotted repetitions. |
| `--resume` | `true` | Reuse complete existing run directories. |
| `--sample-thin` | `10` | Saved-sample thinning after the first repetition. |
| `--first-rep-sample-thin` | `1` | Saved-sample thinning for first repetitions. |
| `--save-wide-csv` | `true` | Save `samples_wide.csv`. |
| `--save-jld2` | `false` | Save `result_bundle.jld2`. |

Simulation outputs are written under:

```text
results/simulation/
  tables/
  figures/
  runs/<run_id>/
    samples_wide.csv
    metadata.json
    diagnostics.json
    posterior_summary.csv
    metrics.csv
    false_discovery.csv
    complete.txt
    diagnose/
```

The long posterior sample CSV is not written. Diagnostics and recovery use
`result_bundle.jld2` when present, otherwise `samples_wide.csv`.

## Simulation Data and Censoring Summary

Generate simulation datasets without fitting MCMC models, and report event and
censoring summaries:

```powershell
julia --project=. scripts\generate_simulation_data.jl --n 200,400 --p 50,150 --nonzero 6,12 --baseline weibull,exponential,gompertz
```

Common options:

| Option | Default | Description |
|---|---:|---|
| `--outdir` | `results/simulation_data` | Output directory. |
| `--reps` | `1` | Number of generated repetitions per setting. |
| `--n` | `200` | Sample size. Comma-separated grids are supported. |
| `--p` | `50` | Number of taxa/covariates. Comma-separated grids are supported. |
| `--nonzero` | `6` | Number of true nonzero coefficients. Supported values are `6` and `12`. |
| `--baseline` | `weibull` | `weibull`, `exponential`, or `gompertz`. |
| `--seed` | `1` | Base random seed. |
| `--tmax` | `100.0` | Administrative censoring time. |
| `--save-data` | `false` | Also save generated `X`, `time`, `event`, and true beta CSV files. |

The censoring summary is written to
`results/simulation_data/tables/censoring_summary.csv`.

## Real-Data Script

Run:

```powershell
julia --project=. scripts\analyze_real_data.jl [options]
```

Common options:

| Option | Default | Description |
|---|---:|---|
| `--input` | `data/T1D.csv` | Input CSV file. |
| `--outdir` | `results/realdata/T1D` | Output directory. |
| `--time-col` | `time` | Survival time column. |
| `--status-col` | `status` | Event indicator column. |
| `--exclude-cols` | `Sex,Antibiotics` | Non-taxa columns to exclude. |
| `--priors` | `hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso` | Comma-separated prior names. |
| `--sampler` | `nuts` | Same sampler names as the simulation script. |
| `--samples` | `500` | Saved posterior samples per fit. |
| `--warmup` | `200` | Warmup/adaptation iterations. |
| `--chains` | `4` | Number of chains. |
| `--p0` | `auto` | `auto` or numeric. `truth` is rejected for real data. |
| `--constraint` | `soft_center` | Zero-sum constraint mode. |
| `--diagnose` | `true` | Generate diagnostics after each fit. |

The script detects taxa as numeric columns not listed in `--time-col`,
`--status-col`, or `--exclude-cols`. Missing abundance values are coalesced to
zero before fitting.

Real-data outputs are written under:

```text
results/realdata/T1D/
  tables/
  figures/
  runs/<run_id>/
    samples_wide.csv
    result_bundle.jld2
    metadata.json
    diagnostics.json
    posterior_summary.csv
    diagnose/
```

## Saved Outputs and Diagnostics

`save_run_outputs` writes:

```text
samples_wide.csv
metadata.json
diagnostics.json
posterior_summary.csv
result_bundle.jld2          # when save_jld2 = true
```

`samples_wide.csv` contains `chain`, `iter`, `draw`, one column per parameter,
and `logposterior` when available. `result_bundle.jld2` stores the wide sample
table, a `MCMCChains.Chains` object when available, diagnostics, metadata,
settings, data information, and posterior summaries.

`diagnose_run` writes:

```text
diagnostics_summary.csv
diagnostics_summary.json
traceplot.<format>
density.<format>
acf.<format>
rhat_scatter.<format>       # multi-chain only
rhat_histogram.<format>     # multi-chain only
gamma_inclusion_probabilities.csv  # when gamma parameters are present
```

Command-line diagnostics:

```powershell
julia --project=. scripts\run_diagnose.jl --input results\runs\example
scripts\run_diagnose.cmd results\runs\example
powershell -ExecutionPolicy Bypass -File scripts\run_diagnose.ps1 results/runs/example
```

```bash
bash scripts/run_diagnose.sh results/runs/example
```

## Project Layout

```text
priors/                      Prior type definitions
mainfuncs/data/              Cox data preparation
mainfuncs/likelihood/        Cox partial likelihood and gradients
mainfuncs/priors/            Prior-specific log densities and helpers
mainfuncs/samplers/          MH, Gibbs, MALA, ULA, Turing NUTS, explicit NUTS
mainfuncs/targets/           Explicit proximal NUTS targets
mainfuncs/diagnostics/       Saving, Chains conversion, diagnostics
scripts/                     Simulation, real-data, and diagnostics CLIs
check/                       Benchmarks and sampler compatibility checks
test/                        Unit and smoke tests
```

Useful developer commands:

```powershell
julia --project=. -e "using Pkg; Pkg.test()"
julia --project=. check\sampler_matrix_experiment.jl
julia --project=. check\check_explicit_gradients.jl
julia --project=. check\bench_cox_gradient.jl 400 50
julia --project=. check\bench_samplers.jl 400 50 25 25 2
```
