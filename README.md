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
    standardize = false,
    tie_method = :efron)
```

Tied event times use Efron's approximation by default. Pass
`tie_method = :breslow` to reproduce the previous Breslow approximation.

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
| `:soft_center` | Adds a Gaussian penalty on `sum(beta)` with scale `0.001 * p`. |
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
| `:nuts_explicit` | Alias for `sampler = :nuts, nuts_backend = :explicit`. |
| `:hmc` | Explicit-gradient AdvancedHMC with fixed leapfrog trajectories. |
| `:mh` | Adaptive random-walk Metropolis. Proximal L1 priors are not implemented for this sampler. |
| `:mala` | MALA for continuous states, including `prox-lasso` and `prox-sslasso`. Set `mala_precondition = true` to use observed-information preconditioning. |
| `:ula` | Unadjusted Langevin updates using `mala_step_size`, including proximal L1 priors. |
| `:gibbs` | Gibbs / Metropolis-within-Gibbs. SSLASSO uses spike/slab Gibbs updates; `prox-sslasso` uses the proximal block path. |

For `prox-sslasso`, the Gibbs sampler updates the continuous block
`(beta, eta)` with a block-level HMC or NUTS transition, then updates
`gamma` and `theta`. Each continuous transition is built from the
`gamma`/`theta` values present at the start of that block; the updated
discrete state is used by the next block. Warmup is burn-in plus block
step-size adaptation (`target_accept`, `prox_block_adapt_every`), and
sampling freezes the step size and diagonal metric. `sampler = :hmc` uses
the block HMC kernel, while `sampler = :nuts, nuts_backend = :explicit`
uses the block NUTS kernel. `prox-sslasso` with Turing NUTS-Gibbs is
disabled because the old single-draw `Turing.NUTS(0)` block update did not
maintain a valid persistent continuous-block state; use
`nuts_backend = :explicit`, `sampler = :hmc`, or `sampler = :mala`.

Turing NUTS supports threaded chains through `Turing.MCMCThreads()` when
`chains > 1`. Explicit NUTS runs chain-level threading by default when
`chains > 1`; pass `parallel_chains = false` to force serial chains. Built-in
MH/MALA/ULA/Gibbs use `multichains = true` for chain-level threading.
For HS/RHS-family priors (`hs`, `rhs`, `new-rhs`, `est-rhs`, `cox-hs`,
`cox-rhs`), NUTS and HMC also support `noncentered = true` / `--noncentered true`.
The sampler then uses latent `beta_raw` variables internally but saves the
usual `beta[...]` draws.

MALA defaults to the existing identity-mass proposal and warmup-window step-size
adaptation. Hessian-preconditioned MALA is opt-in:

```julia
sample_bayes_cox(
    data;
    prior = cox_paper_prior("cox-rhs"; p0 = 1.0),
    sampler = :mala,
    mala_precondition = true,
)
```

The preconditioner uses observed information, not the raw Hessian:

```text
D^2(u) = [-d^2 log posterior(u) + ridge I]^(-1)
```

The sampler builds `G(u) = -d^2 log posterior(u) + ridge I` and uses Cholesky
solves with `G(u)`; it does not form `inv(G)` and does not use the raw Hessian
as a covariance matrix. Standard priors use beta-block preconditioning, while
`prox-lasso` and `prox-sslasso` use the full continuous `(beta, eta)` block and
keep the discrete Gibbs updates unchanged.

MALA supports two step-size schedules:

| `mala_stepsize_schedule` | Behavior |
|---|---|
| `:adaptive_mcmc` | Default. Existing BayesCox4Micro warmup-only acceptance-window adaptation of `mala_step_size`; the step is frozen after warmup. |
| `:adaptive_sgld` | Recomputes `epsilon_m = c0 * m^(-0.55) / (n * d_block)` every iteration, with warmup-only `c0` tuning by default to keep acceptance near 40%-60%. |

For the SIM-style schedule:

```julia
sample_bayes_cox(
    data;
    sampler = :mala,
    prior = cox_paper_prior("prox-lasso"),
    mala_precondition = true,
    mala_stepsize_schedule = :adaptive_sgld,
    mala_sgld_c0 = 1.0,
)
```

Here `n = size(data.z, 1)` and `d_block` is `p` for beta-block updates and
`p + 1` for proximal continuous blocks. The scalar `epsilon_m` is still clamped
by `mala_min_step` and `mala_max_step`.

## CAR Smoothing

When `use_car = true`, CAR smoothing is applied to HS/RHS local log-scales.
The scripts only pass CAR weights to prior names containing `hs`.

Supported `car_distance` values:

```text
bray,jaccard,jsd
```

CAR uses the Gaussian quadratic penalty on adjacent log-local scales:
`-0.5 * rho * sum(w_ij * (loglambda_i - loglambda_j)^2)`.

By default CAR uses the fixed `car_precision` value. To sample the CAR strength,
enable `sample_car_strength`; the model samples `kappa ~ Beta(a, b)` and sets
`rho = kappa / dbar_W`. Here `dbar_W` is computed only from the CAR weight
matrix as the mean active weighted degree, excluding isolated nodes. `Beta(1,1)`
is uniform on `kappa` in `(0, 1)`. When `rho` is sampled, the Gaussian CAR
normalizing term is included.

```powershell
--sample-car-strength true
--car-strength-prior-a 1.0
--car-strength-prior-b 1.0
--car-degree-scale mean_active
--car-kappa-init 0.5
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
| `--baseline` | `weibull` | `weibull`, `exponential`, or `gompertz`. Default censoring is baseline-specific: exponential `U(0,64)`, Weibull `U(0,35)`, Gompertz `U(0,35)`. |
| `--exponential-rate` | `1/14` | Exponential baseline hazard rate `lambda`. Aliases: `--exponential-lambda`, `--exponential-scale` for reciprocal rate. |
| `--weibull-rate` | `1/144` | Weibull baseline hazard parameter `lambda` in `lambda * nu * t^(nu-1)`. Aliases: `--weibull-lambda`, `--weibull-scale` for reciprocal rate. |
| `--weibull-shape` | `2.0` | Weibull shape `nu`. Alias: `--weibull-nu`. |
| `--gompertz-rate` | `1/25` | Gompertz baseline hazard rate `lambda`. Aliases: `--gompertz-lambda`, `--gompertz-scale` for reciprocal rate. |
| `--gompertz-alpha` | `0.1` | Gompertz shape `alpha` in `lambda * exp(alpha * t)`. Alias: `--gompertz-shape`. |
| `--priors` | `hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso` | Comma-separated prior names. |
| `--sampler` | `nuts` | `nuts`, `nuts_turing`, `nuts_explicit`, `hmc`, `mh`, `mala`, `ula`, or `gibbs`. |
| `--nuts-backend` | `turing` | `turing` or `explicit` when `--sampler nuts`. |
| `--samples` | `500` | Saved posterior samples per fit. |
| `--warmup` | `200` | Warmup/adaptation iterations. |
| `--thin` | `1` | MCMC thinning before in-memory summaries. |
| `--chains` | `4` | Number of chains. |
| `--multichains` | `false` | Parallelize non-NUTS chains. |
| `--target-accept` | `0.8` | Target acceptance for NUTS/HMC or adaptive non-NUTS samplers. |
| `--hmc-n-leapfrog` | `10` | Fixed number of leapfrog steps for `--sampler hmc`. |
| `--hmc-step-size` | auto | Optional initial HMC leapfrog step size; when omitted, AdvancedHMC finds one. |
| `--noncentered` | `false` | Use non-centered HS/RHS parameterization for NUTS/HMC. |
| `--seed` | `1` | Base random seed. |
| `--use-car` | `true` | Use CAR smoothing for HS/RHS-style priors. |
| `--car-distance` | `bray` | CAR distance. |
| `--car-weight-quantile` | `0.75` | Similarity quantile threshold used when building CAR weights. |
| `--car-precision` | `1.0` | CAR smoothing precision. |
| `--sample-car-strength` | `false` | Sample adaptive Gaussian CAR strength `kappa ~ Beta(a,b)` and use `rho = kappa / dbar_W`; when false, use fixed `--car-precision`. |
| `--car-strength-prior-a` | `1.0` | Beta prior shape `a` for adaptive CAR `kappa`. |
| `--car-strength-prior-b` | `1.0` | Beta prior shape `b` for adaptive CAR `kappa`. |
| `--car-degree-scale` | `mean_active` | Scale for adaptive CAR degree normalization. Supported values: `mean_active`, `median_active`, `q90_active`. |
| `--car-kappa-init` | `0.5` | Initial adaptive CAR `kappa` in `(0,1)`. |
| `--p0` | `auto` | `auto`, numeric, or `truth` for simulation. |
| `--p0-alpha` | `0.05` | Screening threshold for automatic `p0`. |
| `--tie-method` | `efron` | Cox tied-event method: `efron` or `breslow`. |
| `--constraint` | `soft_center` | Zero-sum constraint mode. |
| `--prox-sum0-lambda` | `1e-3` | Zero-sum Moreau smoothing parameter. |
| `--prox-l1-lambda` | `1e-3` | L1 Moreau smoothing parameter for proximal L1 priors. |
| `--prox-root-tol` | `1e-10` | Projection root tolerance for proximal L1 priors. |
| `--prox-root-maxiter` | `100` | Projection root iteration limit for proximal L1 priors. |
| `--hs-local-scale`, `--hs-tau-scale` | `1`, `1` | HS/CoxHS local shrinkage scale and fixed HS global scale. `--hs-tau-scale` is only active when the prior does not use Cox `tau0`. |
| `--rhs-local-scale`, `--rhs-tau-scale` | `1`, `1` | RHS/CoxRHS local shrinkage scale and fixed RHS global scale. `--rhs-tau-scale` is only active for fixed `tau_formula`. |
| `--rhs-slab-df`, `--rhs-slab-s2` | `4`, `2` | RHS slab `c2 ~ InverseGamma(df/2, df*s2/2)` hyperparameters. |
| `--lasso-lambda2-shape`, `--lasso-lambda2-rate` | `1`, `0.1` | Hyperprior for BLASSO global `lambda^2`. |
| `--sslasso-spike-scale`, `--sslasso-slab-scale` | simulation `0.04`, `0.5`; real data `1`, `10` | SSLASSO spike and slab Laplace scales. |
| `--sslasso-theta-a`, `--sslasso-theta-b` | `1`, `1` | Beta prior hyperparameters for SSLASSO inclusion probability. |
| `--alpha-prior-shape`, `--alpha-prior-rate` | prior-specific | Proximal L1 epigraph `alpha` prior hyperparameters for `prox-lasso` and `prox-sslasso`. |
| `--mala-step-size` | `1e-4` | MALA/ULA step size. |
| `--mala-target-accept` | `0.574` | MALA step-size adaptation target. |
| `--mala-adapt-every` | `50` | MALA adaptation interval. |
| `--mala-precondition` | `false` | Enable observed-information Hessian-preconditioned MALA. |
| `--mala-precondition-mode` | `auto` | Preconditioning block: `auto`, `beta_dense`, `beta_diag`, or `full_dense`. |
| `--mala-precondition-diag` | `false` | Use only the diagonal of the observed-information metric. |
| `--mala-metric-ridge` | `1e-6` | Ridge added to the observed-information metric before Cholesky. |
| `--mala-metric-max-cholesky-tries` | `8` | Maximum jitter attempts before diagonal fallback. |
| `--mala-stepsize-schedule` | `adaptive_mcmc` | `adaptive_mcmc` for existing warmup adaptation or `adaptive_sgld` for SIM-style iteration-dependent steps. |
| `--mala-sgld-c0` | `1.0` | Scale `c0` in `epsilon_m = c0 * m^(-0.55)/(n*d_block)`. |
| `--mala-sgld-target-accept` | `0.5` | Documented target for SIM-style tuning. |
| `--mala-sgld-accept-low`, `--mala-sgld-accept-high` | `0.40`, `0.60` | Warmup-only `c0` tuning band. |
| `--mala-sgld-adapt-c0` | `true` | Tune `c0` during warmup windows. |
| `--mala-sgld-adapt-every` | `--mala-adapt-every` | Warmup window length for `c0` tuning. |
| `--mala-sgld-c0-min`, `--mala-sgld-c0-max` | `1e-12`, `1e12` | Bounds for `c0`. |
| `--diagnose` | `true` | Write diagnostic tables for every repetition; diagnostic plots are generated only for plotted repetitions. |
| `--resume` | `true` | Reuse complete existing run directories. |
| `--sample-thin` | `10` | Saved-sample thinning after the first repetition. |
| `--first-rep-sample-thin` | `1` | Saved-sample thinning for first repetitions. |
| `--save-wide-csv` | `true` | Save `samples_wide.csv`. |
| `--save-jld2` | `false` | Save `result_bundle.jld2`. |

Programmatic `prox-sslasso` block HMC/NUTS tuning also accepts
`prox_block_step_size`, `prox_block_min_step_size`,
`prox_block_max_step_size`, `prox_block_adapt_every`,
`prox_block_divergence_threshold`, and `prox_block_metric = :unit_diag`.

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

For focused C-index tuning, use the hyperparameter grid runner:

```powershell
julia --project=. scripts\tune_prior_hyperparams.jl --priors rhs,lasso,sslasso --samples 200 --warmup 100 --chains 2 --p0 truth
```

It writes `hyperparam_tuning_metrics.csv` and a `Cindex_mean`-sorted
`hyperparam_tuning_summary.csv` under the selected `--outdir`. MSE is kept as
a simulation-only reference column when the true coefficients are available.
The tuning runner also accepts grid options such as `--hs-p0-grid`,
`--rhs-local-scale-grid`, `--prox-l1-lambda-grid`, and
`--alpha-prior-shape-grid` for CoxHS/CoxRHS and proximal priors.

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
| `--baseline` | `weibull` | `weibull`, `exponential`, or `gompertz`. Default censoring is baseline-specific: exponential `U(0,64)`, Weibull `U(0,35)`, Gompertz `U(0,35)`. |
| `--exponential-rate` | `1/14` | Exponential baseline hazard rate `lambda`. Aliases: `--exponential-lambda`, `--exponential-scale` for reciprocal rate. |
| `--weibull-rate` | `1/144` | Weibull baseline hazard parameter `lambda` in `lambda * nu * t^(nu-1)`. Aliases: `--weibull-lambda`, `--weibull-scale` for reciprocal rate. |
| `--weibull-shape` | `2.0` | Weibull shape `nu`. Alias: `--weibull-nu`. |
| `--gompertz-rate` | `1/25` | Gompertz baseline hazard rate `lambda`. Aliases: `--gompertz-lambda`, `--gompertz-scale` for reciprocal rate. |
| `--gompertz-alpha` | `0.1` | Gompertz shape `alpha` in `lambda * exp(alpha * t)`. Alias: `--gompertz-shape`. |
| `--seed` | `1` | Base random seed. |
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
| `--comp-cols` | empty | Comma-separated component covariate columns. If empty, all columns except `--time-col`, `--status-col`, and `--exclude-cols` are used. |
| `--exclude-cols` | `Sex,Antibiotics` | Columns to exclude when `--comp-cols` is empty. Use `none` for an empty exclusion list. |
| `--priors` | `hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso` | Comma-separated prior names. |
| `--sampler` | `nuts` | Same sampler names as the simulation script. |
| `--samples` | `500` | Saved posterior samples per fit. |
| `--warmup` | `200` | Warmup/adaptation iterations. |
| `--chains` | `4` | Number of chains. |
| `--p0` | `auto` | `auto` or numeric. `truth` is rejected for real data. |
| `--tie-method` | `efron` | Cox tied-event method: `efron` or `breslow`. |
| `--constraint` | `soft_center` | Zero-sum constraint mode. |
| `--diagnose` | `true` | Generate diagnostics after each fit. |

When `--comp-cols` is empty, the script uses all columns not listed in
`--time-col`, `--status-col`, or `--exclude-cols` as component covariates. When
`--comp-cols` is provided, only those columns are used. Component covariates must
be numeric and nonnegative; missing component values are coalesced to zero before
fitting.

Real-data outputs are written under:

```text
results/realdata/T1D/
  tables/
    component_columns.csv
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
mainfuncs/samplers/          MH, Gibbs, MALA, ULA, HMC, Turing NUTS, explicit NUTS
mainfuncs/targets/           Explicit proximal NUTS/HMC targets
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
