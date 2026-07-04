# BayesCox4Micro

BayesCox4Micro is a Julia package for Bayesian Cox proportional hazards
models with compositional microbiome, immune-cell, or other nonnegative
abundance covariates. It combines compositional preprocessing, sparse shrinkage
priors, optional CAR smoothing across components, several MCMC backends, and
scripts for simulation and real-data workflows.

## Features

- Cox partial likelihood with Efron tied-event handling by default.
- Closure, pseudocount, log-transform, and optional standardization for
  nonnegative component matrices.
- Horseshoe, regularized horseshoe, Cox-specific HS/RHS, Bayesian LASSO,
  spike-and-slab LASSO, and proximal L1 prior variants.
- Zero-sum coefficient constraints through soft centering or a proximal
  Moreau-Yosida penalty.
- Turing NUTS, explicit NUTS, HMC, adaptive random-walk MH, MALA, ULA, and
  Gibbs / Metropolis-within-Gibbs samplers.
- Optional CAR smoothing on HS/RHS local scales with fixed or sampled CAR
  strength.
- Saved posterior draws, summaries, diagnostics, coefficient plots, C-index,
  LPML, WAIC, PSIS-LOO, and simulation recovery metrics.

## Requirements

- Julia `1.11` according to `Project.toml`.
- The checked-in `Manifest.toml` pins the tested dependency set.
- R is optional and only needed for the plotting helper scripts under
  `scripts/`.

Install dependencies from the repository root:

```powershell
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

Run the test suite:

```powershell
julia --project=. -e "using Pkg; Pkg.test()"
```

## Quick Start

Run a small simulation:

```powershell
julia --project=. scripts/run_simulation.jl --sampler mh --samples 50 --warmup 20 --reps 1 --n 200 --p 50 --nonzero 6 --priors hs,cox-rhs,lasso --p0 truth --diagnose false
```

Analyze the bundled T1D data:

```powershell
julia --project=. scripts/analyze_real_data.jl --input data/T1D.csv --sampler nuts --samples 500 --warmup 200 --priors hs,cox-rhs,lasso,sslasso
```

Use Julia threads for multi-chain runs:

```powershell
$env:JULIA_NUM_THREADS = "4"
julia --project=. scripts/run_simulation.jl --sampler nuts --chains 4 --multichains true
```

## Programmatic API

The top-level API accepts an `n x p` nonnegative component matrix, survival
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
    nuts_backend = :explicit,
    chains = 4,
    constraint = :soft_center,
)

summary = posterior_beta_summary(fit)
draws = posterior_beta_draws(fit)
save_run_outputs(fit; output_dir = "results/runs/example")
diagnose_run("results/runs/example")
```

Default preprocessing is:

```julia
prepare_cox_data(
    X,
    time,
    event;
    transform = :log,
    pseudocount = 1e-6,
    closure = true,
    standardize = false,
    tie_method = :efron,
)
```

Use `tie_method = :breslow` only when Breslow tied-event behavior is required.

## Priors

Construct prior objects with `cox_paper_prior(name; ...)`. HS/RHS-family priors
use `p0` as the expected number of nonzero coefficients; when `p0` is omitted,
the package estimates it by univariate Cox screening and clamps it to
`1:(p - 1)`.

| Name | Prior |
| --- | --- |
| `hs` | Horseshoe. |
| `rhs` | Regularized horseshoe with linear `tau0` and sampled pseudo scale. |
| `new-rhs` | Regularized horseshoe with fixed `pseudo_sigma = 1`. |
| `est-rhs` | Regularized horseshoe with `pseudo_sigma` estimated from a preliminary Cox fit. |
| `cox-hs` | Horseshoe with Cox-specific `tau0`. |
| `cox-rhs` | Regularized horseshoe with Cox-specific `tau0`. |
| `lasso` | Bayesian LASSO using a normal-exponential mixture. |
| `sslasso` | Spike-and-slab LASSO. |
| `prox-lasso` | Proximal LASSO using L1 epigraph Moreau smoothing. |
| `prox-sslasso` | Proximal spike-and-slab LASSO with inclusion updates. |

Common prior hyperparameters are exposed through the API and CLI:

- HS/RHS: `hs_local_scale`, `hs_tau_scale`, `rhs_local_scale`,
  `rhs_tau_scale`, `rhs_slab_df`, `rhs_slab_s2`.
- BLASSO: `lasso_lambda`, `lasso_lambda2_shape`, `lasso_lambda2_rate`.
- SSLASSO: `sslasso_spike_scale`, `sslasso_slab_scale`,
  `sslasso_theta_a`, `sslasso_theta_b`.
- Proximal L1 priors: `prox_l1_lambda`, `alpha_prior_shape`,
  `alpha_prior_scale`, `alpha_prior_kind`, `prox_root_tol`,
  `prox_root_maxiter`.

## Samplers

| Symbol / CLI value | Notes |
| --- | --- |
| `:nuts` / `nuts` | NUTS with `nuts_backend = :turing` or `:explicit`. |
| `:nuts_turing` / `nuts_turing` | Alias for Turing NUTS. |
| `:nuts_explicit` / `nuts_explicit` | Alias for explicit-gradient NUTS. |
| `:hmc` / `hmc` | Explicit-gradient HMC with fixed leapfrog length. |
| `:mh` / `mh` | Adaptive random-walk Metropolis. Proximal L1 priors are not supported. |
| `:mala` / `mala` | MALA, optionally observed-information preconditioned. |
| `:ula` / `ula` | Unadjusted Langevin updates using the MALA step-size interface. |
| `:gibbs` / `gibbs` | Gibbs or Metropolis-within-Gibbs; includes SSLASSO updates. |

Important compatibility notes:

- `prox-lasso` and `prox-sslasso` are not implemented for `sampler = :mh`.
- `sampler = :gibbs` uses the proximal L1 path only for `prox-sslasso`.
- `prox-sslasso` with Turing NUTS-Gibbs is disabled; use explicit NUTS, HMC,
  MALA, or Gibbs instead.
- HS/RHS-family priors support `noncentered = true` with NUTS and HMC.
- Turing NUTS uses `Turing.MCMCThreads()` for multi-chain runs. Explicit NUTS,
  MH, MALA, ULA, and Gibbs can run chain-level threading when requested.

MALA preconditioning is opt-in:

```julia
sample_bayes_cox(
    X,
    time,
    event;
    prior = cox_paper_prior("prox-lasso"),
    sampler = :mala,
    mala_precondition = true,
    mala_stepsize_schedule = :adaptive_sgld,
)
```

The preconditioner uses observed information,
`G(u) = -d^2 log posterior(u) + ridge * I`, with Cholesky solves rather than
forming `inv(G)`.

## Constraints and CAR Smoothing

The coefficient constraint is separate from the prior choice:

| Constraint | Effect |
| --- | --- |
| `:soft_center` / `soft_center` | Adds a Gaussian penalty on `sum(beta)`. |
| `:prox` / `prox` | Adds a zero-sum Moreau-Yosida penalty. |
| `:none` / `none` | Disables the zero-sum penalty. |

CAR smoothing is enabled with `use_car = true` or `--use-car true`. The scripts
apply CAR weights only to prior names containing `hs`. Supported component
distances are `bray`, `jaccard`, and `jsd`.

Fixed CAR strength uses `car_precision`. Adaptive CAR strength samples
`kappa ~ Beta(a, b)` and sets `rho = kappa / dbar_W`, where `dbar_W` is a
degree scale computed from the active CAR graph.

## Command-Line Workflows

All Julia commands below should be run from the repository root with
`--project=.`. Unknown CLI flags are rejected; the source of truth for shared
analysis flags is `scripts/analysis_common.jl`.

### Simulation

```powershell
julia --project=. scripts/run_simulation.jl [options]
```

Core simulation options:

| Option | Default | Description |
| --- | --- | --- |
| `--outdir` | `results/simulation` | Output root. |
| `--reps` | `1` | Number of repetitions. |
| `--n` | `200` | Sample-size grid; comma-separated values are accepted. |
| `--p` | `50` | Component-count grid; comma-separated values are accepted. |
| `--nonzero` | `6` | Number of true nonzero coefficients. |
| `--baseline` | `weibull` | `weibull`, `exponential`, or `gompertz`. |
| `--priors` | HS/RHS/LASSO/SSLASSO set | Comma-separated prior names. |
| `--p0` | `auto` | `auto`, numeric, or `truth` for simulation. |
| `--resume` | `true` | Reuse complete compatible run directories. |

Shared model, sampler, prior, CAR, diagnostic, and tuning flags include:

```text
--sampler --nuts-backend --samples --warmup --thin --chains --multichains
--target-accept --max-depth --hmc-n-leapfrog --hmc-step-size
--mala-step-size --mala-precondition --mala-stepsize-schedule
--constraint --prox-sum0-lambda --prox-l1-lambda
--pseudocount --standardize --tie-method --noncentered
--use-car --car-distance --car-weight-quantile --car-precision
--sample-car-strength --diagnose --diagnose-format
--tune_priorsparams --tune_algparams --cv-folds
```

Baseline-specific aliases are also supported, for example
`--weibull-rate`, `--weibull-scale`, `--weibull-shape`,
`--gompertz-rate`, `--gompertz-alpha`, and `--exponential-rate`.

### Real Data

```powershell
julia --project=. scripts/analyze_real_data.jl [options]
```

Input CSV files must contain survival time, status, and numeric nonnegative
component columns. Missing survival rows are dropped. Missing component values
are treated as zero.

| Option | Default | Description |
| --- | --- | --- |
| `--input` | `data/T1D.csv` | Input CSV. |
| `--outdir` | `results/realdata/T1D` | Output root. |
| `--time-col` | `time` | Survival time column. |
| `--status-col` | `status` | Event indicator column. |
| `--comp-cols` | empty | Explicit comma-separated component columns. |
| `--exclude-cols` | `Sex,Antibiotics` | Columns excluded when `--comp-cols` is empty. |
| `--prior-configs` | empty | Optional JSON file with per-prior hyperparameter overrides. |
| `--p0` | `auto` | `auto` or numeric. `truth` is invalid for real data. |

Bundled real-data inputs include `data/T1D.csv`, several HCT CSV files, TIMER2
TCGA processed CSV files, and CIBERSORT TCGA survival CSV files.

### Simulation Data Only

Generate datasets and censoring summaries without fitting MCMC models:

```powershell
julia --project=. scripts/generate_simulation_data.jl --n 200,400 --p 50,150 --nonzero 6,12 --baseline weibull,exponential,gompertz --save-data true
```

The censoring summary is written to
`results/simulation_data/tables/censoring_summary.csv`.

### Diagnostics

Run diagnostics for an existing run directory:

```powershell
julia --project=. scripts/run_diagnose.jl --input results/runs/example --format pdf
```

`run_diagnose.jl` accepts a positional input path as well:

```powershell
julia --project=. scripts/run_diagnose.jl results/runs/example
```

## Outputs

Simulation outputs are written under:

```text
results/simulation/
  tables/
    simulation_metrics_long.csv
    simulation_metrics_summary.csv
    simulation_false_discovery.csv
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

Real-data outputs are written under:

```text
results/realdata/<dataset>/
  tables/
    component_columns.csv
    model_metrics.csv
    univariate_cox_screen.csv
    coef_<prior>.csv
  figures/
  runs/realdata_<prior>/
    samples_wide.csv
    result_bundle.jld2
    metadata.json
    diagnostics.json
    posterior_summary.csv
    diagnose/
```

`save_run_outputs` writes `samples_wide.csv`, `metadata.json`,
`diagnostics.json`, `posterior_summary.csv`, and optionally
`result_bundle.jld2`. `diagnose_run` writes diagnostic summaries and plots such
as trace, density, ACF, and multi-chain R-hat figures.

## Repository Layout

```text
Project.toml / Manifest.toml    Julia environment
src/BayesCox4Micro.jl           Package entry point
main.jl                         Module exports and includes
priors/                         Prior type definitions
mainfuncs/data/                 Cox data preparation
mainfuncs/likelihood/           Cox partial likelihood and gradients
mainfuncs/priors/               Prior log densities, CAR, constraints, prox helpers
mainfuncs/samplers/             MH, Gibbs, MALA, ULA, HMC, Turing NUTS, explicit NUTS
mainfuncs/targets/              Explicit HMC/NUTS target objects
mainfuncs/diagnostics/          Saving, Chains conversion, diagnostics
scripts/                        Simulation, real-data, plotting, and summary CLIs
data/                           Bundled example and processed survival datasets
test/                           Unit and smoke tests
```

## Development Notes

Useful commands:

```powershell
julia --project=. -e "using Pkg; Pkg.test()"
julia --project=. scripts/run_simulation.jl --sampler mh --samples 5 --warmup 5 --reps 1 --n 30 --p 6 --nonzero 6 --priors lasso --p0 truth --diagnose false --outdir results/smoke
julia --project=. scripts/analyze_real_data.jl --input data/T1D.csv --sampler mh --samples 5 --warmup 5 --priors lasso --diagnose false --outdir results/smoke_real
```

The repository intentionally stores generated analysis artifacts under
`results/`; generated contents are ignored by Git. Use `metadata.json` and the
script command line to reproduce a run.
