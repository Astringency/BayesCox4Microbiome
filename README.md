# BayesCox4Micro

BayesCox4Micro is a Julia package for Bayesian Cox proportional hazards models with compositional microbiome, immune-cell, or other nonnegative abundance covariates. It provides compositional preprocessing, Efron tied-event handling, sparse priors, MCMC sampling, posterior diagnostics, and complete simulation and real-data workflows.

The [consolidated review results](results/review_20261012/RESULTS.md) include the
completed 2,400-fit Weibull/Gompertz main experiment, 1,800 supplementary fits,
canonical S4 graph ablation, official BCCox PH comparison, and T1D results.
Historical reevaluations and their sampler exclusions are retained in the same
report, with current and historical evidence identified separately.

This README focuses on two methods:

- **Cox-RHS** (CLI name `cox-rhs`): a regularized horseshoe prior with a Cox-specific global scale.
- **Prox-SSLASSO** (CLI name `prox-sslasso`): a proximal spike-and-slab LASSO prior for sparse variable selection.

## Exact zero-sum baseline

Use `--constraint exact` with `--sampler hmc`, or with
`--sampler nuts --nuts-backend explicit`. Cox-RHS supports both centered and
noncentered sampling; Prox-SSLASSO retains its indicator Gibbs updates. The
zero-sum smoothing parameter is inactive for this setting; the separate
Prox-SSLASSO L1 smoothing parameter remains active.

The sampler uses p-1 orthonormal contrast coordinates and evaluates the
likelihood and shrinkage kernel on the reconstructed p taxon coefficients.
If design columns are standardized, the constraint is imposed on the original
coefficients: `sum(beta_model ./ col_scale) = 0`. Sex and other ordinary
covariates must not be included in this composition constraint.

For Gaussian HS/RHS shrinkage, let D be the conditional diagonal covariance
and a = 1 ./ col_scale. The exact prior is the normalized conditional Gaussian
on a'beta=0, with covariance D - D*a*a'*D/(a'*D*a). The centered log density adds
`0.5*log(2*pi*a'*D*a) - log(norm(a))` to the unconstrained Gaussian kernel.
The noncentered implementation uses an orthogonal transform in the latent
Gaussian coordinates and preserves this same conditional distribution.
Dropping this scale-dependent normalization can change the hyperprior and can
make an exact horseshoe model improper near a zero global scale.

For Prox-SSLASSO, the existing joint Moreau prior kernel, including alpha and
indicator factors, is restricted to the zero-sum subspace. This defines a
joint constrained model; it does **not** separately renormalize the conditional
coefficient prior for every alpha/gamma combination. Consequently it must not
be described as preserving the unconstrained hyperparameter marginals. The
exact RHS baseline likewise should not be called the unmodified zero-lambda
limit of the current unnormalized prox penalty. These prior conventions must
be reported when interpreting comparisons.

Each saved run additionally writes `zero_sum_summary.csv`, containing per-chain
and pooled coefficient-sum mean, SD, quantiles, mean absolute value, RMS,
maximum absolute value and fractions within fixed tolerances. Floating-point
roundoff is expected for exact constraints. Sums of marginal posterior medians
need not vanish, even when every joint draw satisfies the constraint; the sum
of posterior means does vanish up to rounding.

`rank_diagnostics.csv` explicitly labels rank-normalized split/folded R-hat,
bulk ESS and tail ESS. `selection_mixing.csv` reports indicator switch counts,
directional transitions, dwell lengths and lag-1 correlation. Constant
indicators are flagged; they do not receive a claim of convergence.
New explicit HMC/NUTS runs also save warm-up and sampling transition records
and summaries, including acceptance probabilities and divergences. Divergence
uses numerical failures or absolute energy error above 1000; block samplers
use their configured threshold. Historical samples alone cannot reconstruct
missing transition records. Runtime is retained in `diagnostics.json`.

Reference: [Stan's sum-to-zero transform and unknown-scale correction](https://mc-stan.org/docs/reference-manual/transforms.html#sum-to-zero-transforms).

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

Each setting and repetition also generates an independent **1,000-subject test
cohort**, with the same true coefficients, covariate distribution, baseline hazard
and censoring distribution as the training data. All methods in that repetition
share the test cohort, saved under `test_data/`. Its RNG uses a separate namespace
from training and tuning. Test covariates use training-fitted preprocessing.
`Cindex` remains the historical training score; `Cindex_train` makes that meaning
explicit, and `Cindex_test` reports held-out concordance for each posterior mean,
median and MAP estimate. Both scores have mean/SD columns in the simulation
summary. `--resume true` can add test scores to compatible saved runs using their
coefficient summaries, without refitting MCMC.

Simulation metrics also report **95% credible-interval coverage**:
`coverage_95_all`, `coverage_95_nonzero`, and `coverage_95_zero`. Each repetition
counts the fraction of original-scale true coefficients inside their marginal
equal-tailed posterior intervals, including the endpoints, for all coefficients,
true signals, and true zeros respectively. These are interval metrics, so the
same values appear on the mean, median and MAP rows. They do not use the test
cohort. Both simulation summary workflows report their `_mean`, `_sd`, and `_n`
(valid repetition count). An empty coefficient group or an invalid interval in
that group gives missing coverage; fewer than two valid repetitions gives a
missing SD. `--resume true` backfills per-run metrics from saved summaries.
To refresh aggregate reports from existing `posterior_summary.csv` files that
contain `true_beta` and interval bounds, without fitting or loading MCMC samples:

```bash
julia +1.11.7 --startup-file=no --project=. scripts/summarize_simulation_results.jl --root results/simulation/example
```

This writes coverage into `model_metrics_summary.csv` and
`simulation_results_summary.csv` under the specified root. Historical intervals
without saved truth or bounds have missing coverage, rather than assumed values.

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

Prox-SSLASSO's explicit HMC/NUTS samplers honor `--init zero` and `--init rand`
for single, serial, and parallel chains. The default `zero` starts at beta=0,
alpha=1, gamma=1, and theta=0.5. With `rand`, each chain draws beta from
Uniform(-2, 2), projects it onto the configured zero-sum hyperplane when
applicable, and sets alpha to `max(1, 1.1 * sum(weights .* abs.(beta)))` so the
initial state lies inside the weighted L1 epigraph; gamma=1 and theta=0.5.
This initialization needs no preliminary SSLASSO fit. Earlier versions dropped
`init` at the matrix API and in the proximal sampler dispatch, so an old run
requested with `--init rand` may actually have used the fixed zero start.
Simulation metadata now records `init`, and resume checks include it.

## Nested C-index evaluation

Real-data analysis now defaults to **5 outer folds and 5 inner folds**, with prior
hyperparameter tuning enabled. `--outer-cv-folds` controls outer evaluation;
`--cv-folds` controls inner tuning. Inside each outer training subset, every
candidate is fitted and scored on inner folds. The best mean inner C-index selects
the configuration, which is refitted on the outer training subset and scored once
on the untouched outer test fold. Automatic `p0` screening, preprocessing and
training-dependent prior scales are recomputed inside each fit. If algorithm
tuning is enabled, algorithm candidates are also selected by inner C-index.

`nested_cv/nested_cv_summary.csv` reports the unweighted mean of outer-fold
C-indices and their SD/SE, separately for posterior mean, median and MAP.
`nested_cv/nested_cv_folds.csv` contains individual outer scores and the selected
configuration. Per-method/per-fold directories record inner scores and original
row indices for auditing. Completed outer folds resume automatically, guarded by
a fingerprint of data, settings, Julia version and source. Use a new output
directory if these change. A failed inner fold disqualifies that candidate; a
failed or undefined outer score leaves the overall evaluation missing until all
outer folds succeed.

After outer evaluation, full-data tuning and fitting still produce a final model.
Their tuning scores are not test scores. Bayesian comparison summaries now read
only completed nested evaluations; old tuning scores are kept separately as
`tuning_cv_cindex`. An old result without nested evaluation has a missing test
score. Baseline packages' existing CV scores retain their original definitions.

To backfill nested CV for saved Bayesian runs, use:

```bash
julia +1.11.7 --startup-file=no --project=. scripts/analyze_real_data.jl --workflow cv-replay --priors cox-rhs,prox-sslasso --outer-cv-folds 5 --cv-folds 5
```

This command now retunes inside each outer fold; historical fixed-configuration
CV is not reused. It uses the current candidate grids, so it need not reproduce
an older grid's selected parameters. Saved automatically screened `p0` is reset
to automatic training-only screening. Full-data `--prior-configs` files are used
only for the final full-data model, not to bypass inner tuning. Input feature
columns and fixed settings must be specified independently of held-out outcomes;
previous outcome-based feature screening outside these scripts is not repeated
automatically. `--nested-cv false` explicitly skips the new evaluation, and
`--tune_priorsparams false` disables the prior search; neither option turns a
training or old tuning score into an independent test score.

Run the focused regression tests with:

```bash
julia +1.11.7 --startup-file=no --project=. test/cindex_evaluation_tests.jl
```

## Leave-one-out evaluation

All LOO entry points default to `fixed_risk`: event-factor PSIS uses the full-data
risk sets without rebuilding them or refitting the model for each deleted subject.
The T1D runner fits the full data once per method to obtain these diagnostics.

To rerun T1D for Cox-RHS (`soft_center`) and Prox-SSLASSO (`prox`) using the saved
configurations and fixed risk sets:

```bash
julia +1.11.7 --startup-file=no -t 4 --project=. scripts/analyze_real_data.jl --workflow loo
```

The unified `bash scripts/analyze_real_data.sh loo` entry uses the same `fixed_risk` default.
Explicit `--loo-mode subject_refit` (or `both` in the T1D runner) enables subject
deletion, training risk-set reconstruction and checkpointed refits. That alternative
scores partial-likelihood ratios, including censored subjects' effects on other
event denominators, rather than full survival-time densities. Its scores and
fixed-risk PSIS scores have different evaluation targets.

## Experiment entry points

See [scripts/README.md](scripts/README.md) for the runnable entry points and
[review runners](scripts/review/README.md) for the supplementary protocols.
The baseline grid uses the same `run_simulation.jl` and prior-family shell
launchers as ordinary simulations; it has no separate Python driver.
Each simulation shell launcher keeps its own settings and loops. The terminal
shows group and overall fit counts; sampler output is saved under the run's
`logs/` directory. Shared shell helpers live in `scripts/lib/`.

All real-data workflows enter through `analyze_real_data.sh` or
`analyze_real_data.jl`: ordinary datasets, the eight T1D review combinations,
partitioned nested CV, candidate workers, final posterior retries and LOO.
The T1D implementation lives in `scripts/lib/realdata/`.

The BCCox R runners and their 50 paired S4 cohorts are maintained in the sibling
`BayesCox4micro_baseline` R project. Its `simulation_bccox_survival_analysis.R`
runs from local `datasets/S4`, `testdata/S4`, `beta/S4` and `bccox` inputs.

The local `audit/` directory is ignored by Git. Current review verification,
aggregation, reporting, and synchronization tools live in `audit/review/`;
the experiment scheduler remains in `scripts/review/run_review_rerun.py`.
Earlier tools and the pre-cleanup source archive remain under `audit/`.
The review directory now has one Markdown report, `results/review_20261012/RESULTS.md`.
Its consolidation manifest maps every original Markdown source to the merged
report and the byte-preserving archive. Numerical results remain in place.
