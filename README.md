# BayesCox4Micro

BayesCox4Micro is a Julia implementation of Bayesian Cox proportional hazards
models for microbiome abundance data. The package focuses on sparse regression
priors, compositional preprocessing, optional CAR smoothing across taxa, and
scripted simulation and real-data workflows.

## Version and Package Management

This repository is a Julia project. Always run scripts from the repository root
with the project environment activated.

```julia
versioninfo()

using Pkg
Pkg.activate(pwd())
Pkg.instantiate()
Pkg.status()
```

The package metadata is defined in `Project.toml`:

- Package name: `BayesCox4Micro`
- Package version: `0.1.0`
- Julia compatibility: `1.11`
- Main modeling dependencies: `AdvancedHMC`, `Distances`, `Distributions`,
  `ForwardDiff`, `LogDensityProblems`, `LogDensityProblemsAD`,
  `PosteriorStats`, `ProgressMeter`, `Survival`, and `Turing`.
- Data, persistence, and diagnostics dependencies: `CSV`, `DataFrames`,
  `JLD2`, `JSON`, `MCMCChains`, `Plots`, and `StatsPlots`.

`Manifest.toml` is committed so the exact dependency resolution can be restored
with `Pkg.instantiate()`.

Typical commands:

```powershell
julia --project=. -e "using Pkg; Pkg.instantiate()"
julia --project=. -e "using Pkg; Pkg.test()"
julia --project=. scripts\run_simulation.jl --samples 50 --warmup 20 --reps 1
julia --project=. scripts\analyze_real_data.jl --samples 50 --warmup 20
julia --project=. check\sampler_matrix_experiment.jl
julia --project=. check\bench_cox_gradient.jl 400 50
julia --project=. check\bench_samplers.jl 400 50 25 25 2
```

For threaded NUTS chains, start Julia with threads enabled:

```powershell
$env:JULIA_NUM_THREADS = "4"
julia --project=. scripts\run_simulation.jl --sampler nuts --chains 4
```

During script development, setting `USE_REVISE=true` makes the scripts try
`Revise.includet` for `scripts/analysis_common.jl` before falling back to
`include`.

## Project Structure

```text
BayesCox4Micro/
  Project.toml
  Manifest.toml
  README.md
  main.jl
  src/
    BayesCox4Micro.jl
  priors/
    abstract.jl
    horseshoe.jl
    regularized_horseshoe.jl
    bayesian_lasso.jl
    spike_slab_lasso.jl
  mainfuncs/
    types.jl
    data/
      data.jl
    likelihood/
      cox_likelihood.jl
    priors/
      prior_common.jl
      horseshoe_methods.jl
      regularized_horseshoe_methods.jl
      bayesian_lasso_methods.jl
      spike_slab_lasso_methods.jl
    targets/
      cox_targets.jl
    samplers/
      mh_sampler.jl
      gibbs_sampler.jl
      mala_sampler.jl
      nuts_turing_sampler.jl
      nuts_explicit_sampler.jl
    diagnostics/
      diagnostics.jl
    posterior.jl
    api.jl
  scripts/
    analysis_common.jl
    run_simulation.jl
    analyze_real_data.jl
    run_diagnose.jl
    run_simulation.{ps1,cmd,sh}
    analyze_real_data.{ps1,cmd,sh}
    run_diagnose.{ps1,cmd,sh}
  check/
    bench_cox_gradient.jl
    bench_samplers.jl
    check_explicit_gradients.jl
    requested_sampler_matrix.jl
    sampler_matrix_experiment.jl
    test_samplers_cox_normal_prior.jl
    test_samplers_normal.jl
  data/
    T1D.csv
  test/
    runtests.jl
  results/
```

Key roles:

- `src/BayesCox4Micro.jl` loads `main.jl`, which defines the module and exports
  the public API.
- `priors/` contains the prior type definitions.
- `mainfuncs/priors/prior_common.jl` contains shared prior utilities, CAR weight
  construction, zero-sum constraint helpers, and proximal projection helpers.
- `mainfuncs/data/data.jl` prepares Cox data, including zero handling, optional
  closure, log transform, optional standardization, and risk-set construction.
- `mainfuncs/likelihood/cox_likelihood.jl` implements the Cox partial
  likelihood, event log-likelihood contributions, and the fast explicit Cox
  score gradient.
- `mainfuncs/priors/*_methods.jl` implements prior-specific validation,
  initial states, state names, and log posterior contributions.
- `mainfuncs/targets/cox_targets.jl` implements explicit log density and
  gradient targets for proximal Cox samplers.
- `mainfuncs/samplers/nuts_turing_sampler.jl` implements the existing
  Turing/NUTS backend.
- `mainfuncs/samplers/nuts_explicit_sampler.jl` implements the AdvancedHMC
  explicit-gradient NUTS backend for proximal LASSO.
- `mainfuncs/samplers/mh_sampler.jl` implements adaptive random-walk Metropolis.
- `mainfuncs/samplers/mala_sampler.jl` implements MALA, ULA, and proximal
  variants with reusable workspaces.
- `mainfuncs/samplers/gibbs_sampler.jl` implements the SSLASSO Gibbs /
  Metropolis-within-Gibbs sampler.
- `mainfuncs/posterior.jl` and `mainfuncs/api.jl` provide posterior extraction,
  summaries, MAP extraction, diagnostics, C-index, `cox_paper_prior`, and the
  top-level `sample_bayes_cox` API.
- `mainfuncs/diagnostics/diagnostics.jl` converts sampler outputs to
  `MCMCChains.Chains`, writes raw samples and metadata, and generates
  diagnostics by calling `MCMCChains`, `StatsPlots`, and `Plots`.
- `scripts/analysis_common.jl` contains shared CLI parsing, simulation data
  generation, fitting, metrics, and `Plots.jl`-based PDF/EPS plotting.
- `scripts/run_simulation.jl` runs simulation grids.
- `scripts/analyze_real_data.jl` analyzes a CSV survival dataset such as
  `data/T1D.csv`.
- `scripts/run_diagnose.jl` regenerates diagnostics from saved run outputs
  without rerunning MCMC.
- `check/sampler_matrix_experiment.jl` runs a compact prior-by-sampler smoke
  experiment and writes a status/timing summary.

## Implementation Details

### Data Model

The top-level API is:

```julia
fit = sample_bayes_cox(
    X,
    time,
    event;
    prior = cox_paper_prior("cox-rhs"; X = X, time = time, event = event),
    sampler = :nuts,
)
```

`X` is an `n x p` nonnegative abundance matrix, `time` is a nonnegative survival
time vector, and `event` must contain `0/1` indicators. By default,
`sample_bayes_cox` calls:

```julia
prepare_cox_data(X, time, event;
    transform = :log,
    pseudocount = 1e-6,
    closure = true,
    standardize = false)
```

For `transform = :log`, the implementation adds the pseudocount, optionally
normalizes each row to sum to one (`closure = true`), and then applies
`log.(X)`. If `pseudocount = 0`, zeros are replaced by half of the minimum
positive abundance before closure and log transform.

### Priors

The repository implements the following priors through concrete subtypes of
`AbstractCoxPrior`.

| Name | API construction | Implementation |
|---|---|---|
| `hs` | `cox_paper_prior("hs"; ...)` | Standard horseshoe. `tau ~ Cauchy+(0, 1)`, local scales `lambda_j ~ Cauchy+(0, local_scale)`, and `beta_j ~ Normal(0, tau * lambda_j)`. |
| `rhs` | `cox_paper_prior("rhs"; ...)` | Regularized horseshoe with the linear tau0 formula, sampled `pseudo_sigma^2 ~ half-t_3(0, 10)`, `c2 ~ InverseGamma(slab_df / 2, slab_df * slab_s2 / 2)`, and default slab settings `slab_df = 4`, `slab_s2 = 2`. |
| `new-rhs` | `cox_paper_prior("new-rhs"; ...)` | Regularized horseshoe with the linear tau0 formula and fixed `pseudo_sigma = 1`. |
| `est-rhs` | `cox_paper_prior("est-rhs"; X = X, time = time, event = event, ...)` | Regularized horseshoe with the linear tau0 formula and `pseudo_sigma = sqrt(1 / muhat)`, where `muhat = mean(exp(X * beta_hat))` from a preliminary `Survival.coxph` fit on prepared data. |
| `cox-hs` | `cox_paper_prior("cox-hs"; ...)` | Horseshoe with Cox-specific global scale `tau0 = (p0 / (p - p0)) / sqrt(n_events)`. |
| `cox-rhs` | `cox_paper_prior("cox-rhs"; ...)` | Regularized horseshoe with Cox-specific `tau0` and the regularized slab. |
| `lasso` | `cox_paper_prior("lasso")` | Bayesian LASSO. If `lambda` is not fixed, `lambda2 ~ Gamma(lambda2_shape, 1 / lambda2_rate)` with defaults `lambda2_shape = 1`, `lambda2_rate = 0.1`; local variances use the normal-exponential mixture representation. |
| `sslasso` | `cox_paper_prior("sslasso"; realdata = false/true)` | Spike-and-slab LASSO. The marginal API uses a mixture of Laplace components with `theta ~ Beta(theta_a, theta_b)`. Simulation defaults are `spike_scale = 0.04`, `slab_scale = 0.5`; real-data script defaults are `spike_scale = 1`, `slab_scale = 10`. |
| `prox-lasso` | `cox_paper_prior("prox-lasso")` plus `constraint = :prox` | Proximal LASSO implementation using an L1 epigraph variable and Moreau smoothing. Supported by `mala`, `ula`, and the prox-specialized `nuts` path. |
| `prox-sslasso` | `cox_paper_prior("prox-sslasso")` plus `constraint = :prox` | Conditional proximal SSLASSO implementation. Discrete `gamma` indicators are updated by Gibbs steps, while continuous `beta` and the L1 epigraph variable are updated by MALA, ULA, or NUTS block updates with weighted L1 Moreau smoothing. |

For HS/RHS-style priors, `p0` controls the expected number of nonzero
coefficients in the tau0 formulas. If `p0` is omitted, `cox_paper_prior`
estimates it by fitting one univariate Cox model per covariate with
`Survival.jl`; `p0` is the count of Wald p-values below `p0_alpha`, clamped to
`1:(p - 1)`.

The script-level prior names are:

```text
hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso
```

### CAR Smoothing

When `use_car = true`, CAR smoothing is applied to HS/RHS local log-scales.
The helper builds taxa-level distances from the original abundance columns,
double-centers squared distances into a similarity matrix, thresholds absolute
similarities by `car_weight_quantile = 0.75`, and adds a pairwise smoothing
penalty on differences in `log(lambda_j)`.

Supported `car_distance` values are:

```text
bray, jaccard, jsd
```

In both analysis scripts, CAR is only passed to prior names containing `hs`.

### Samplers

Supported sampler symbols are:

- `:nuts`: NUTS with selectable backend. The default
  `nuts_backend = :turing` preserves the existing Turing.jl behavior. Use
  `nuts_backend = :explicit` for the AdvancedHMC explicit-gradient backend.
  The explicit backend supports all built-in priors with hand-coded gradients.
  Proximal Bayesian LASSO and proximal SSLASSO use explicit-gradient proximal
  NUTS blocks.
- `:nuts_explicit`: convenience alias for `sampler = :nuts,
  nuts_backend = :explicit`. The historical typo `:nuts_explict` is also
  accepted.
- `:nuts_turing`: convenience alias for `sampler = :nuts,
  nuts_backend = :turing`.
- `:mh`: adaptive random-walk Metropolis fallback.
- `:mala`: MALA for continuous states, including proximal LASSO and proximal
  SSLASSO modes when `constraint = :prox`.
- `:ula`: unadjusted Langevin updates with fixed `--mala-step-size`; for
  hybrid priors, non-beta hyperparameters still use fixed random-walk MH steps.
- `:gibbs`: Gibbs / Metropolis-within-Gibbs sampler. SSLASSO uses its
  spike/slab Gibbs updates, prox-SSLASSO uses the proximal block Gibbs path, and
  other priors use adaptive coordinate-wise Metropolis-within-Gibbs. It does not support
  CAR weights.

Turing NUTS supports multiple threaded chains through `Turing.MCMCThreads()`
when `chains > 1`. Explicit NUTS defaults to chain-level threading when
`chains > 1`; pass `parallel_chains = false` to force serial chains. Built-in
MH/MALA/ULA/Gibbs still use `multichains = true` for chain-level threading.
Each threaded chain gets an independent RNG and private sampler workspaces.
When `progress = true`, threaded built-in chains display per-chain progress bars.

All sampler outputs can be saved through `save_run_outputs`. Turing NUTS
results keep the original Turing/MCMCChains chain dimension. Built-in `:mh`,
`:mala`, `:ula`, `:gibbs`, and explicit `:nuts` results are converted to
`MCMCChains.Chains` by `result_to_chains`.

Example backend selection:

```julia
fit_turing = sample_bayes_cox(X, time, event;
    prior = BayesianLassoPrior(),
    sampler = :nuts,
    nuts_backend = :turing,
    constraint = :prox)

fit_explicit = sample_bayes_cox(X, time, event;
    prior = BayesianLassoPrior(),
    sampler = :nuts,
    nuts_backend = :explicit,
    constraint = :prox,
    chains = 4,
    multichains = true)
```

### Zero-Sum Coefficient Constraint

The package implements two active ways to enforce the compositional zero-sum
coefficient constraint, plus an opt-out mode:

1. `constraint = :soft_center`

   Adds a Gaussian penalty on `sum(beta)`, implemented as
   `logpdf(Normal(0, soft_center_scale), sum(beta))`. If
   `soft_center_scale` is not provided, the default is `0.001 * p`. The legacy
   keyword `soft_center = true` maps to this mode.

2. `constraint = :prox`

   Adds the Moreau-Yosida zero-sum penalty
   `-0.5 * sum(beta)^2 / (prox_sum0_lambda * p)`, where
   `prox_sum0_lambda` defaults to `1e-3`. The exported helpers
   `project_sum0`, `sum0_moreau`, and `sum0_moreau_gradient` implement the
   projection and penalty pieces.

3. `constraint = :none`

   Disables the zero-sum penalty. The legacy keyword `soft_center = false` maps
   to this mode.

The proximal LASSO machinery is separate from the zero-sum proximal penalty.
For `BayesianLassoPrior` with `sampler = :mala` and `constraint = :prox`, the
sampler also applies L1 epigraph Moreau smoothing through `project_l1_epigraph`
and `l1_moreau`. For `SpikeSlabLassoPrior`, it uses
`project_weighted_l1_epigraph` and `weighted_l1_moreau`.

The simulation and real-data scripts expose the same zero-sum constraint
controls through `--constraint`, `--soft-center-scale`, and
`--prox-sum0-lambda`. The L1 epigraph proximal controls
`--prox-l1-lambda`, `--prox-root-tol`, and `--prox-root-maxiter` are relevant
only for `--sampler mala` with `lasso` or `sslasso` and `--constraint prox`.

## Simulation Script

Run:

```powershell
julia --project=. scripts\run_simulation.jl [options]
```

Example smoke test:

```powershell
julia --project=. scripts\run_simulation.jl --sampler mh --samples 50 --warmup 20 --reps 1 --n 200 --p 50 --nonzero 6 --priors hs,cox-rhs,lasso --baseline weibull --p0 truth
```

Example with the proximal zero-sum constraint:

```powershell
julia --project=. scripts\run_simulation.jl --sampler mala --constraint prox --prox-sum0-lambda 1e-3 --samples 50 --warmup 20 --reps 1 --n 200 --p 50 --nonzero 6 --priors hs,lasso,sslasso --baseline weibull --p0 truth
```

`scripts/run_simulation.jl` accepts the following parameters.

| Parameter | Default | Explanation |
|---|---:|---|
| `--outdir` | `results/simulation` | Output directory. Tables are written to `outdir/tables`; PDF and EPS figures are written to `outdir/figures`. |
| `--reps` | `1` | Number of simulation repetitions. |
| `--n` | `200` | Sample size. Comma-separated grids are supported, for example `200,400`. |
| `--p` | `50` | Number of taxa/covariates. Comma-separated grids are supported. |
| `--nonzero` | `6` | Number of true nonzero coefficients. Supported values are `6` and `12`; comma-separated grids are supported. |
| `--baseline` | `weibull` | Baseline event-time model. Supported values are `weibull`, `exponential`, and `gompertz`; comma-separated grids are supported. |
| `--priors` | `hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso` | Comma-separated prior names to fit. |
| `--sampler` | `nuts` | Sampler backend. Supported values are `nuts`, `mh`, `mala`, `ula`, and `gibbs`. `gibbs` is only implemented for SSLASSO. |
| `--nuts-backend` | `turing` | Backend used when `--sampler nuts`. Supported values are `turing` and `explicit`. The aliases `--sampler nuts_turing` and `--sampler nuts_explicit` select the corresponding backend directly. |
| `--samples` | `500` | Number of saved posterior samples per fit. |
| `--warmup` | `200` | Number of warmup/adaptation iterations. |
| `--chains` | `4` | Number of chains. NUTS uses Turing's multi-chain sampler; `mh`, `mala`, `ula`, and `gibbs` run independent chains and combine them for diagnostics. |
| `--multichains` | `false` | Run non-NUTS independent chains in parallel with Julia threads. Start Julia with multiple threads for this to help. |
| `--target-accept` | `0.8` | Target acceptance probability passed to NUTS or adaptive non-NUTS samplers. |
| `--seed` | `1` | Base random seed. Repetition `rep` uses `seed + rep - 1`. |
| `--use-car` | `true` | Whether to use CAR smoothing for HS/RHS-style prior names. Boolean parsing treats `true`, `1`, `yes`, and `y` as true; any other value is false. |
| `--progress` | `true` | Whether samplers show progress bars. Same boolean parsing as `--use-car`. |
| `--car-distance` | `bray` | Distance used to construct CAR weights. Supported values are `bray`, `jaccard`, and `jsd`. |
| `--p0` | `auto` | Prior nonzero count for HS/RHS-style priors. `auto`, omitted, `none`, or `missing` triggers univariate Cox screening; a numeric value fixes `p0`; `truth` uses the simulation `--nonzero` value. |
| `--p0-alpha` | `0.05` | Wald p-value threshold used by univariate Cox screening when `--p0` is automatic. |
| `--constraint` | `soft_center` | Zero-sum coefficient constraint passed to `sample_bayes_cox`. Supported values are `soft_center`, `prox`, and `none`. |
| `--soft-center-scale` | automatic | Optional scale for `constraint = :soft_center`. If omitted, the package uses `0.001 * p`. |
| `--prox-sum0-lambda` | `1e-3` | Moreau-Yosida smoothing parameter for the zero-sum constraint when `--constraint prox`. |
| `--prox-l1-lambda` | `1e-3` | L1 epigraph Moreau smoothing parameter for proximal LASSO / proximal SSLASSO. Used by `mala`, `ula`, and prox-specialized `nuts` paths with LASSO or SSLASSO and `--constraint prox`. |
| `--prox-root-tol` | `1e-10` | Root-finding tolerance for L1 epigraph projections used by proximal samplers. |
| `--prox-root-maxiter` | `100` | Maximum root-finding iterations for L1 epigraph projections used by proximal samplers. |
| `--mala-step-size` | `1e-4` | Initial MALA step size. |
| `--mala-target-accept` | `0.574` | Target acceptance rate for MALA step-size adaptation. This is separate from `--target-accept`. |
| `--mala-adapt-every` | `50` | Number of warmup iterations between MALA step-size adaptation updates. |
| `--mala-min-step` | `1e-8` | Lower bound for the adapted MALA step size. |
| `--mala-max-step` | `1.0` | Upper bound for the adapted MALA step size. |
| `--diagnose` | `true` | Run `diagnose_run` for plotted repetitions and save diagnostics under that run directory. With the default `--plot-first-rep-only true`, diagnostics are generated only for `rep == 1`. |
| `--diagnose-format` | `png` | Comma-separated diagnostic plot formats used by automatic diagnostics. |
| `--diagnose-top-k-beta` | `10` | Number of beta parameters selected for automatic diagnostic plots. |
| `--diagnose-max-lag` | `50` | Maximum lag for automatic ACF diagnostic plots. |
| `--resume` | `true` | Reuse an existing run directory when saved samples and posterior summaries are available. Missing per-run metrics are recovered from the saved samples. |
| `--plot-first-rep-only` | `true` | Save coefficient and diagnostic plots only for the first repetition of each setting. Set to `false` to plot every repetition. |
| `--sample-thin` | `10` | Save every kth posterior draw for repetitions after the first one. Metrics and coefficient summaries are still computed from the full in-memory draw set. |
| `--first-rep-sample-thin` | `1` | Saved-sample thinning for `rep == 1`. |
| `--save-wide-csv` | `true` | Save the compact wide posterior sample table. |
| `--save-long-csv` | `false` | Save the much larger long posterior sample table. |
| `--save-jld2` | `false` | Save the JLD2 result bundle. Disabled by default in simulations to reduce disk usage. |

Simulation data generation follows the paper-style setup in
`scripts/analysis_common.jl`: true nonzero coefficients are inserted at fixed
taxon indices, abundance rows are generated through a softmax transformation of
multivariate normal latent variables, and event times are generated from the
selected baseline hazard with administrative censoring at `tmax = 100`.

Outputs include coefficient summaries, per-run metrics, long-format metrics,
summarized metrics, and false discovery summaries. By default coefficient and
diagnostic plots are written only for the first repetition of each setting.
Automatic diagnostic trace, density, and ACF plots are restricted to the true
nonzero coefficients. Metrics include log partial likelihood, LPML, AIC, BIC,
WAIC, DIC, PSIS-LOO/LOOIC, Pareto-k diagnostics, C-index, and sampler timing
diagnostics.

Each fitted simulation run is also saved under:

```text
results/simulation/runs/<run_id>/
  samples_wide.csv
  metadata.json
  diagnostics.json
  posterior_summary.csv
  metrics.csv
  false_discovery.csv
  complete.txt
  diagnose/
```

`samples_long.csv` and `result_bundle.jld2` are optional in simulation runs and
are written only when `--save-long-csv true` or `--save-jld2 true` is passed.

## Real-Data Analysis Script

Run:

```powershell
julia --project=. scripts\analyze_real_data.jl [options]
```

Example:

```powershell
julia --project=. scripts\analyze_real_data.jl --input data\T1D.csv --sampler nuts --samples 500 --warmup 200 --priors hs,cox-rhs,lasso,sslasso
```

Example with the proximal zero-sum constraint:

```powershell
julia --project=. scripts\analyze_real_data.jl --input data\T1D.csv --sampler mala --constraint prox --prox-sum0-lambda 1e-3 --samples 500 --warmup 200 --priors hs,lasso,sslasso
```

`scripts/analyze_real_data.jl` accepts the following parameters.

| Parameter | Default | Explanation |
|---|---:|---|
| `--input` | `data/T1D.csv` | Input CSV file. |
| `--outdir` | `results/realdata/T1D` | Output directory. Tables are written to `outdir/tables`; PDF and EPS figures are written to `outdir/figures`. |
| `--time-col` | `time` | Column name containing survival time. |
| `--status-col` | `status` | Column name containing the event indicator. Values must be `0/1`. |
| `--exclude-cols` | `Sex,Antibiotics` | Comma-separated non-taxa columns to exclude from the abundance matrix, in addition to the time and status columns. |
| `--priors` | `hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso` | Comma-separated prior names to fit. |
| `--sampler` | `nuts` | Sampler backend. Supported values are `nuts`, `mh`, `mala`, `ula`, and `gibbs`. `gibbs` is only implemented for SSLASSO. |
| `--nuts-backend` | `turing` | Backend used when `--sampler nuts`. Supported values are `turing` and `explicit`. The aliases `--sampler nuts_turing` and `--sampler nuts_explicit` select the corresponding backend directly. |
| `--samples` | `500` | Number of saved posterior samples per fit. |
| `--warmup` | `200` | Number of warmup/adaptation iterations. |
| `--chains` | `4` | Number of chains. NUTS uses Turing's multi-chain sampler; `mh`, `mala`, `ula`, and `gibbs` run independent chains and combine them for diagnostics. |
| `--multichains` | `false` | Run non-NUTS independent chains in parallel with Julia threads. Start Julia with multiple threads for this to help. |
| `--target-accept` | `0.8` | Target acceptance probability passed to NUTS or adaptive non-NUTS samplers. |
| `--seed` | `1` | Random seed. |
| `--use-car` | `true` | Whether to use CAR smoothing for HS/RHS-style prior names. Boolean parsing treats `true`, `1`, `yes`, and `y` as true; any other value is false. |
| `--progress` | `true` | Whether samplers show progress bars. Same boolean parsing as `--use-car`. |
| `--car-distance` | `bray` | Distance used to construct CAR weights. Supported values are `bray`, `jaccard`, and `jsd`. |
| `--p0` | `auto` | Prior nonzero count for HS/RHS-style priors. `auto`, omitted, `none`, or `missing` triggers univariate Cox screening; a numeric value fixes `p0`. `truth` is rejected for real-data analysis. |
| `--p0-alpha` | `0.05` | Wald p-value threshold used by univariate Cox screening when `--p0` is automatic. |
| `--constraint` | `soft_center` | Zero-sum coefficient constraint passed to `sample_bayes_cox`. Supported values are `soft_center`, `prox`, and `none`. |
| `--soft-center-scale` | automatic | Optional scale for `constraint = :soft_center`. If omitted, the package uses `0.001 * p`. |
| `--prox-sum0-lambda` | `1e-3` | Moreau-Yosida smoothing parameter for the zero-sum constraint when `--constraint prox`. |
| `--prox-l1-lambda` | `1e-3` | L1 epigraph Moreau smoothing parameter for proximal LASSO / proximal SSLASSO. Used by `mala`, `ula`, and prox-specialized `nuts` paths with LASSO or SSLASSO and `--constraint prox`. |
| `--prox-root-tol` | `1e-10` | Root-finding tolerance for L1 epigraph projections used by proximal samplers. |
| `--prox-root-maxiter` | `100` | Maximum root-finding iterations for L1 epigraph projections used by proximal samplers. |
| `--mala-step-size` | `1e-4` | Initial MALA step size. |
| `--mala-target-accept` | `0.574` | Target acceptance rate for MALA step-size adaptation. This is separate from `--target-accept`. |
| `--mala-adapt-every` | `50` | Number of warmup iterations between MALA step-size adaptation updates. |
| `--mala-min-step` | `1e-8` | Lower bound for the adapted MALA step size. |
| `--mala-max-step` | `1.0` | Upper bound for the adapted MALA step size. |
| `--diagnose` | `true` | Run `diagnose_run` after each fit and save diagnostics under that run directory. |
| `--diagnose-format` | `png` | Comma-separated diagnostic plot formats used by automatic diagnostics. |
| `--diagnose-top-k-beta` | `10` | Number of beta parameters selected for automatic diagnostic plots. |
| `--diagnose-max-lag` | `50` | Maximum lag for automatic ACF diagnostic plots. |

The script detects taxa columns as numeric columns not listed in
`--time-col`, `--status-col`, or `--exclude-cols`. Missing abundance values are
coalesced to zero before model fitting, and taxa columns must be nonnegative.
When `--p0` is automatic, the script also writes
`univariate_cox_screen.csv`.

Outputs include taxa column names, per-prior coefficient summaries, coefficient
plots for posterior mean, median, and MAP estimates, model metrics, and a
metrics plot.

Each fitted real-data run is also saved under:

```text
results/realdata/T1D/runs/<run_id>/
  samples_wide.csv
  samples_long.csv
  result_bundle.jld2
  metadata.json
  diagnostics.json
  posterior_summary.csv
  diagnose/
```

## Saving Samples and Diagnostics

Raw sampler results can be converted to `MCMCChains.Chains` and saved without
rerunning the model:

```julia
result = sample_bayes_cox(X, time, event; prior = BayesianLassoPrior(), sampler = :mh)

chains = result_to_chains(result)
save_run_outputs(
    result;
    output_dir = "results/runs/example",
    run_id = "example",
    metadata = Dict("dataset_name" => "example", "prior_name" => "lasso"),
    settings = Dict("n_samples" => 500, "warmup" => 200, "thin" => 1),
)

diagnose_run("results/runs/example"; output_dir = "results/runs/example/diagnose")
```

The saved CSV files use:

- `samples_wide.csv`: `chain`, `iter`, `draw`, one column per parameter, and
  `logposterior` when available.
- `samples_long.csv`: `chain`, `iter`, `draw`, `parameter`, `value`.

`result_bundle.jld2` stores the wide and long tables, a `MCMCChains.Chains`
object when available, diagnostics, metadata, settings, data information, and
posterior summaries.

`diagnose_run` writes:

```text
diagnostics_summary.csv
diagnostics_summary.json
traceplot.<format>
density.<format>
acf.<format>
rhat_scatter.<format>      # multi-chain only
rhat_histogram.<format>    # multi-chain only
gamma_inclusion_probabilities.csv  # when gamma parameters are present
```

Diagnostics use Julia ecosystem packages rather than custom implementations:

- `MCMCChains.jl` for summary statistics, ESS, R-hat, quantiles, and ACF used by
  plotting recipes.
- `StatsPlots.jl` / `Plots.jl` for trace, density, ACF, R-hat scatter, and
  R-hat histogram plots.

Run diagnostics from the command line:

```powershell
julia --project=. scripts\run_diagnose.jl --input results\runs\example
scripts\run_diagnose.cmd results\runs\example
powershell -ExecutionPolicy Bypass -File scripts\run_diagnose.ps1 results/runs/example
```

```bash
bash scripts/run_diagnose.sh results/runs/example
```

`scripts/run_diagnose.jl` parameters:

| Parameter | Default | Explanation |
|---|---:|---|
| `--input` | `results/runs/example` | Directory containing `result_bundle.jld2` or `samples_wide.csv`. A positional first argument is also accepted. |
| `--output` | `<input>/diagnose` | Directory for diagnostic plots and summaries. |
| `--params` | automatic | Comma-separated parameters to plot, for example `beta[1],beta[2],theta,logposterior`. |
| `--top-k-beta` | `10` | If `--params` is omitted, plot beta coefficients with the largest absolute posterior means. |
| `--max-lag` | `50` | Maximum lag passed to the MCMCChains autocorrelation plot recipe. |
| `--rhat-threshold` | `1.01` | Reference line for R-hat scatter and histogram plots. |
| `--format` | `png,pdf` | Comma-separated output formats for plots. |

If only one chain is available, R-hat is not computable. The diagnostic runner
emits a warning, writes summaries with missing R-hat values when necessary, and
skips R-hat plots instead of failing.

## Sampler Matrix Experiment

Run a compact compatibility/timing pass across the built-in paper priors and
sampler aliases:

```powershell
julia --project=. check\sampler_matrix_experiment.jl
```

By default this script simulates `n = 200`, `p = 50`, and `nonzero = 6`, then
fits the priors:

```text
hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso
```

against the samplers:

```text
mh,mala,ula,gibbs,nuts_explicit,nuts_turing
```

The script writes an incremental CSV summary to:

```text
results/sampler_matrix_<timestamp>/sampler_matrix_summary.csv
```

Environment variables can narrow or shorten the run:

```powershell
$env:BC4M_N = "200"
$env:BC4M_P = "50"
$env:BC4M_NONZERO = "12"
$env:BC4M_BASELINE = "weibull"
$env:BC4M_SAMPLES = "50"
$env:BC4M_WARMUP = "20"
$env:BC4M_PRIORS = "hs,lasso,sslasso"
$env:BC4M_SAMPLERS = "mh,mala,nuts_turing"
julia --project=. check\sampler_matrix_experiment.jl
```
