# BayesCox4Micro

BayesCox4Micro is a Julia package for Bayesian Cox proportional hazards models with compositional microbiome, immune-cell, or other nonnegative abundance covariates. It provides compositional preprocessing, Efron tied-event handling, sparse priors, MCMC sampling, posterior diagnostics, and complete simulation and real-data workflows.

This README focuses on two methods:

- **Cox-RHS** (CLI name `cox-rhs`): a regularized horseshoe prior with a Cox-specific global scale.
- **Prox-SSLASSO** (CLI name `prox-sslasso`): a proximal spike-and-slab LASSO prior for sparse variable selection.

## Setup

The project requires Julia 1.11. Dependency versions are recorded in `Project.toml` and `Manifest.toml`. From the repository root, run:

```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

## Simulation example

The following command generates data with a Weibull baseline hazard and fits both Cox-RHS and Prox-SSLASSO. It uses `n = 200`, `p = 50`, six nonzero coefficients, and four parallel chains:

```bash
julia -t 4 --project=. scripts/run_simulation.jl --outdir results/simulation/example --reps 1 --n 200 --p 50 --nonzero 6 --baseline weibull --priors cox-rhs,prox-sslasso --sampler nuts --nuts-backend explicit --samples 1000 --warmup 500 --chains 4 --multichains true --p0 truth --use-car false
```

Key options:

- `--reps` sets the number of repetitions. `--n`, `--p`, and `--nonzero` also accept comma-separated grids.
- `--baseline` accepts `weibull`, `exponential`, or `gompertz`.
- `--p0 truth` uses the true number of nonzero coefficients to set the Cox-RHS sparsity assumption.
- When Prox-SSLASSO is fitted with NUTS, `--nuts-backend explicit` is required; the Turing NUTS backend is not supported.
- `--use-car false` disables optional CAR smoothing so that this example compares the two priors without CAR.

Results are written to `results/simulation/example/`. Summary metrics are under `tables/`; posterior samples, metadata, and diagnostics for each method are under `runs/`.

## Real-data example

The bundled `data/T1D.csv` contains `time`, `status`, microbiome abundance columns, and the non-compositional columns `Sex` and `Antibiotics`. Run:

```bash
julia -t 4 --project=. scripts/analyze_real_data.jl --input data/T1D.csv --outdir results/realdata/T1D --time-col time --status-col status --exclude-cols Sex,Antibiotics --priors cox-rhs,prox-sslasso --sampler nuts --nuts-backend explicit --samples 1000 --warmup 500 --chains 4 --multichains true --p0 auto --use-car false
```

A custom CSV must contain a finite, nonnegative survival-time column, a 0/1 event column with at least one event, and finite, nonnegative numeric component columns. Select columns with `--time-col`, `--status-col`, `--comp-cols`, or `--exclude-cols`. Rows with missing survival outcomes are dropped; missing component values are treated as zero.

Outputs are written below the selected `--outdir`:

```text
results/realdata/T1D/
  tables/                  # Model metrics, screening, and coefficient summaries
  figures/                 # Coefficient and model-metric plots
  runs/realdata_cox-rhs/   # Cox-RHS samples and diagnostics
  runs/realdata_prox-sslasso/
```

Sampling sizes, prior hyperparameters, compositional constraints, and diagnostics can all be changed through CLI options. See `scripts/analysis_common.jl` for the authoritative option definitions.
