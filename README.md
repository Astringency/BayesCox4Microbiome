# Bayesian Variable Selection for Cox Proportional Hazards Models with Microbiome Compositional Data

## Environment

- Julia 1.11.
- `Project.toml` declares the project dependencies and Julia compatibility; `Manifest.toml` locks their exact versions.
- From the repository root, instantiate the environment with Pkg:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Repository structure

```text
.
├── Project.toml, Manifest.toml   # Julia dependencies and locked versions
├── src/BayesCox4Micro.jl         # Package entry point; loads main.jl
├── main.jl                       # Module definition and core implementation loader
├── priors/                       # Base prior types: Lasso, Horseshoe, SSLASSO, etc.
├── mainfuncs/                    # Model implementation
│   ├── data/                     # Data preprocessing
│   ├── likelihood/               # Cox partial likelihood
│   ├── priors/                   # Prior calculations and constraints
│   ├── targets/                  # Posterior target distributions
│   ├── samplers/                 # MCMC samplers
│   ├── diagnostics/              # Convergence, sampling, and visualization diagnostics
│   ├── metrics/                  # Model evaluation and simulation metrics
│   ├── evidence/                 # Bayesian evidence and bridge sampling
│   └── api.jl, posterior.jl, types.jl, progress.jl, tuning_grids.jl
│                                 # Public API, posterior, types, progress, and tuning grids
├── data/                         # T1D.csv survival data and T1D_metadata.csv annotations
├── scripts/                      # Simulation and real-data analysis entry points
│   ├── run_simulation*.sh         # Simulation launchers
│   ├── analyze_real_data.sh       # T1D analysis entry point
│   ├── lib/simulation/            # Simulation data generation, fitting, and summaries
│   ├── lib/realdata/              # T1D analysis workflow
│   ├── lib/diagnostics/           # PH and related diagnostics
│   ├── lib/plotting/              # Result plots
│   └── review/                    # Supplementary experiment launchers
└── test/                         # Julia and R tests
```

The local workspace may also contain Git-ignored directories: `results/` for run outputs, `docs/` for reference papers and R code, `audit/` for historical materials, and `check/` for temporary checks.

## Usage examples

### Simulation

Generate one dataset with a Weibull baseline hazard and fit Cox-RHS and Prox-SSLASSO:

```bash
julia -t 4 --project=. scripts/lib/simulation/run_simulation.jl \
  --outdir results/simulation/example --reps 1 --n 200 --p 50 --nonzero 6 \
  --baseline weibull --priors cox-rhs,prox-sslasso \
  --sampler nuts --nuts-backend explicit --samples 1000 --warmup 500 \
  --chains 4 --multichains true --p0 truth --use-car false
```

### Real-data analysis

Run the complete analysis workflow on the bundled T1D data, including nested cross-validation, final model fitting, and proportional-hazards diagnostics:

```bash
JULIA_BIN=julia bash scripts/analyze_real_data.sh \
  --outdir results/realdata/T1D/example
```

Schoenfeld diagnostics require R and the R package `survival`.
