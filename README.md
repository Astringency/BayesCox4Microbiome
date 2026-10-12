# BayesCox4Micro

BayesCox4Micro is a Julia package for Bayesian Cox proportional hazards models with compositional microbiome, immune-cell, or other nonnegative abundance covariates. It provides compositional preprocessing, Efron tied-event handling, sparse priors, MCMC sampling, posterior diagnostics, and complete simulation and real-data workflows.

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
julia +1.11.7 --startup-file=no --project=. scripts/compute_realdata_cv_fallback.jl --priors cox-rhs,prox-sslasso --outer-cv-folds 5 --cv-folds 5
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

The existing event-factor PSIS calculation is available as `--loo-mode fixed_risk`
(default). `scripts/analyze_real_data.jl --loo-mode subject_refit` instead deletes
each subject, rebuilds training risk sets and refits the model. Its evaluation
target is a cross-validated partial-likelihood ratio, including censored subjects'
effects on other event denominators. It is not a full survival-time density;
fixed-risk WAIC is reported separately.

To rerun T1D for Cox-RHS (`soft_center`) and Prox-SSLASSO (`prox`) using the saved
configurations, with four parallel leave-one-subject-out fits:

```bash
julia +1.11.7 --startup-file=no -t 4 --project=. scripts/rerun_t1d_loo.jl --loo-mode both --parallel true
```

On Windows, `scripts/run_t1d_loo.ps1` runs the same command. Completed folds are
checkpointed for resumption. See [LOO evaluation](LOO_EVALUATION.md) for
formulas, output meanings, configuration overrides and pilot runs.

## Revision experiment runners

`REVISION_EXPERIMENTS.md` records the experiment matrix, scientific definitions,
source versions, completed results and reproducible commands.

The 2026-10-12 revision is complete: all 31 simulation cells have 50 repetitions
(1500 distinct fits), all four T1D method/adjustment arms are complete, and 3800
historical runs have saved-result metric/diagnostic replays. The
[final report archive](results/revision_20261011/reports_final_all_simulations.zip)
contains 50 hash-verified files. See the final tables and convergence/PH
limitations in [the execution record](REVISION_EXPERIMENTS.md).

`scripts/verify_revision_saved_outputs.py --revision-root REVISION_ROOT
--outdir AUDIT_DIR --require-complete` checks the requested settings, distinct
repetitions, formal sampling budgets, saved diagnostics and clinical score audits
without fitting. A partial report fails this strict completion check.

The reusable server package for that original revision is
`/large_storage/zhangxf/BayesCox/revision_20261011/code_v6`, also archived as
`code_v6.tar.gz` beside it. Its 186-file SHA256 manifest and package-load/stage-lock
checks passed. Use this package with a separate output root for new experiments.
Continue existing clinical CV checkpoints with their frozen code_v4/code_v5
projects: changing their source fingerprint would invalidate cached tuning fits.

Simulation runners default to 50 repetitions, four chains, 2000 warm-up and 5000
retained draws:

```bash
bash scripts/run_sensitivity_revision.sh lambda
bash scripts/run_sensitivity_revision.sh p0
bash scripts/run_scenarios_revision.sh
bash scripts/run_car_revision.sh
SCENARIO=phylo METHOD=prox-sslasso bash scripts/run_car_revision.sh
CONSTRAINTS=none,soft_center,exact bash scripts/run_zero_sum_revision.sh
CONSTRAINTS=prox SUM0_LAMBDA=1e-3 bash scripts/run_zero_sum_revision.sh
CONSTRAINTS=prox SUM0_LAMBDA=1e-6 bash scripts/run_zero_sum_revision.sh
```

Set `JULIA_BIN` to Julia 1.11.7 and `REVISION_OUT` to the desired isolated output
directory before running these commands. The three zero-sum commands cover all
five constraints for both methods; the runner alone defaults to exact only.
Existing compatible completed repetitions are resumed.

The p0 and CAR ablations apply only to Cox-RHS. The CAR runner now defaults to
S4 (`phylo_bm`), using Brownian shared-path covariance and three Cox-RHS graph
arms only. `SCENARIO=phylo` explicitly selects the old exponential tree-distance
DGP, where `METHOD=prox-sslasso` remains available as a legacy reference.
S4 results use a separate `S4/car/` subtree; old `phylo` results remain unchanged.
The S4 extension is complete at 50 repetitions per graph (150 fits), with all
150 fits passing the coefficient convergence screen and zero sampling divergences.
The [S1–S4 complete summary](results/simulation_S1_S4_20261012/RESULTS.md) contains
all nine cells, 450 distinct fits, mean/median/MAP metrics, coverage/FDP,
convergence summaries and S4 paired contrasts. S1–S3 posteriors were reused.
Only summary files and figures are stored locally; large draws remain on the server.
At posterior means, S4 test C-indices are 0.876288 (Independent), 0.875766
(Correct CAR) and 0.872673 (Misspecified CAR): the correct graph does not improve
prediction in this setting. Frozen S4 execution sources are at
`/large_storage/zhangxf/BayesCox/s4_20261012/code_v3`.
Use `scripts/run_s4_pool.py --root S4_ROOT --julia JULIA_1_11_7` for disjoint
formal repetition partitions, and `bash scripts/refresh_s4_reports.sh S4_ROOT`
to recompute summaries without sampling. `scripts/verify_s4_results.py --root
S4_ROOT --require-complete` checks all 150 fits and paired train/test cohorts.
S1 reduces coefficients to 0.35 times the baseline; S2 adds six correlated null
taxa; S3 generates sequencing counts with median depth 5000. Test cohorts always
contain 1000 independent subjects. `--rep-ids 33,50 --reps 50` selects only those
repetitions without changing their original seeds.

Historical diagnostics and metrics can be recomputed without posterior sampling:

```bash
julia --project=. scripts/summarize_revision_diagnostics.jl --roots OLD_RESULTS --outdir NEW_DIAGNOSTICS
julia --project=. scripts/reevaluate_revision_metrics.jl --inventory NEW_DIAGNOSTICS/run_inventory.csv --outdir NEW_METRICS
python scripts/aggregate_revision_diagnostics.py --historical-diagnostics NEW_DIAGNOSTICS --new-root NEW_RESULTS --outdir AGGREGATES
```

These scripts preserve historical inputs. Per-draw coefficient-sum statistics,
rank Rhat, bulk/tail ESS and selection mixing are recoverable from saved samples.
Missing HMC transition records cannot be reconstructed. Coefficient summaries
are enough for mean/median/MAP C-index, marginal 95% interval coverage and
interval-selection empirical FDP, FP/max(TP+FP,1). Missing MAP remains missing.

For T1D, `scripts/run_t1d_clinical_revision.sh` runs joint adjustment or the
unadjusted arm (`ADJUSTED=true|false`, `METHOD=cox-rhs|prox-sslasso`). Clinical
indicators are M versus F and Tylosin versus Control, with independent
Student-t(3,0,2.5) priors; this is a Cox adaptation of weakly informative
regression priors. They are not compositional columns and are outside the taxon
zero-sum and CAR penalties. Both arms use paired 5x5 nested CV; successful inner
fits and outer folds are resumable. Existing unadjusted Cox-RHS full-data samples
are reused as descriptive estimates, separately from CV evaluation. Real-data
defaults retain the existing 800 warm-up/800 draws per chain; `REAL_WARMUP` and
`REAL_SAMPLES` override them.

Run both methods and both clinical arms with the same output root:

```bash
for method in cox-rhs prox-sslasso; do
    for adjusted in false true; do
        METHOD="$method" ADJUSTED="$adjusted" bash scripts/run_t1d_clinical_revision.sh
    done
done
```

`scripts/check_t1d_ph.R` exports term-wise/global Schoenfeld tests and residual
plots. Its joint-score calibration refit is exploratory, not a taxon-wise
Bayesian PH test. See the execution record for the confirmed Prox-SSLASSO stale
indicator correction and the eligibility of historical/early revision outputs.

For the deployed revision, run `bash scripts/refresh_revision_reports.sh REVISION_ROOT`
to refresh diagnostic, performance and clinical tables without sampling. Its
kernel audit excludes affected SSL and unverifiable parallel RHS historical runs
from primary comparisons;
`requested_progress.csv` reports completed and missing repetitions for all 31
simulation cells. Convergence screening is shown separately from the main
all-eligible results. Generate the audit JSON from the Git checkout with
`python scripts/audit_revision_legacy_ssl.py --out REVISION_ROOT/results/legacy_ssl_kernel_audit.json`.
Then combine the RHS rules with
`python scripts/audit_revision_legacy_rhs.py --ssl-audit REVISION_ROOT/results/legacy_ssl_kernel_audit.json --out REVISION_ROOT/results/legacy_kernel_audit.json`.
Historical metrics remain available in audit rows; good Rhat does not override a
kernel exclusion. The historical serial-NUTS T1D full-data RHS fit remains reusable.
The refresh also checks the actual test CSV hashes, test-set size and recorded DGP
specifications across matched repetitions, plus training CSV hashes when saved.
Baseline training CSVs were not saved, so their audit checks the common DGP and
seeds rather than claiming training-data byte equality. These checks are available
separately through `scripts/audit_revision_data_identity.py`.

`scripts/run_rhs_kernel_followup.sh` fills only the required RHS baseline and
zero-sum cells in the isolated deployment (`REVISION_ROOT`, `JULIA_BIN`). It waits
for existing RHS sensitivity jobs to finish, retains the two corrected baseline
fits, and schedules at most five four-chain workers. The former 48-historical +
2-repair baseline manifest is superseded; `resolved_fixed_rhs_baseline.json` is
created only after all 50 corrected baseline repetitions exist.
The sensitivity runner reuses that resolved manifest when available. Otherwise,
lambda=1e-3 and p0=6 share one baseline output directory and a Linux `flock` lock,
so concurrent launches resume the same fits without duplicating the baseline.

After a successful report refresh, `scripts/snapshot_revision_reports.py
--revision-root REVISION_ROOT --out reports.zip` bundles the tables, kernel audits
and job status with SHA256 hashes. A snapshot can be partial; its progress table
records the actual number of completed repetitions.

For this deployed job set, `scripts/finalize_revision_reports.py --revision-root
REVISION_ROOT` waits for the registered tmux jobs to finish successfully, refreshes
saved-result tables, requires all 31 simulation cells to have 50 repetitions, and
writes `reports_final_all_simulations.zip`. It never launches or restarts sampling.
`--status-only` inspects its prerequisites. The server copy runs in tmux
`bcx_final_reports`; progress and failures are recorded in
`logs/final_reports_state.json`. The latest reporting helpers are in
`analysis_tools_v1`, outside the sealed model-source packages.

The deployed RHS queue also uses `advance_rhs_zero_queue.py` to advance its
last prox(1e-6) cell in freed CPU slots. It temporarily suspends only the Python
dispatcher, leaving existing Julia samplers running, and resumes it after the
advanced cell finishes. The queued invocation then reuses completed fits.
This Linux helper requires a Python build with pidfd support (the deployed
command explicitly uses `/usr/bin/python3`). Its tmux session is
`bcx_rhs_zero_advanced`, with state in `logs/advance_rhs_zero_queue.state.json`.
The original dispatcher's `T` process state is intentional while this job runs;
do not independently resume or restart it during the advanced fit.

The completed deployment also used `run_zero_tail_pool.py` (tmux
`bcx_zero_tail_pool`) for up to eight additional four-chain workers. It requires
the verified pilot record and explicit live source PIDs. `assist_zero_sum_tail.py`
copies a source worker's frozen command, changes only its output directory and
repetition subset, and selects repetitions at least six beyond the original
worker's current frontier. It writes outside `results/` until the fit is complete,
then atomically publishes into an absent run directory. Per-case publication
locks and a brief, identity-checked pause prevent races with the original cache
check. Existing posterior files are never overwritten. The original worker later
reuses the result through its normal `--resume true` path.

These operational helpers leave frozen projects unchanged. The older exact RHS
worker has no repetition-subset option and completed its original queue.
Staging directories remain experiment artifacts because the saved metadata
references their test data. `audit_zero_tail_assistance.py` verifies published
sample/metadata hashes and records publication pauses; `--require-complete`
also requires the assistance pool to finish. Final reporting waits for both the
original workers and the assistance pool.

Diagnostic rows label assisted fits with
`runtime_context=assisted_fresh_julia_process`; group summaries count these fits
in `n_assisted_fresh_julia_process`. Their runtime can include compilation in a
fresh Julia process. Original workers can include the recorded brief publication
pauses, and all timings reflect shared-server load. These are observed execution
times rather than a controlled hardware benchmark.

New nested-CV fits retain outer posterior samples for later reevaluation.
`scripts/audit_t1d_outer_fits.jl` is a recovery utility only for completed older
outer folds whose samples were not saved: it reuses the inner selection and
requires exact reproduction of the cached C-indices. It never repeats tuning
or overwrites the original CV scores.

`scripts/verify_t1d_saved_cindex.py --revision-root REVISION_ROOT --data T1D.csv
--outdir AUDIT_DIR` independently reproduces outer test scores from saved draws
without fitting. It also checks that completed outer test folds do not share
subjects and that all five folds cover every subject exactly once. Add
`--require-complete` for the final audit of all 20 method/adjustment/fold fits.

`scripts/verify_zero_sum_saved_statistics.py --revision-root REVISION_ROOT
--outdir AUDIT_DIR --replicate 1` independently checks one completed formal
repetition in each of the ten zero-sum cells. It reads the saved 4 x 5000 draws
to verify coefficient summaries, chain/pooled coefficient-sum statistics,
95% interval coverage, interval-based FDP, the three test C-indices and HMC
transition summaries. It checks shared test-data hashes and records posterior
sample hashes. This numerical reporting audit performs no sampling and does
not establish convergence or audit all 50 repetitions.

For faster completion of an existing T1D run, stop its serial drivers and run
`scripts/run_t1d_partition_pool.sh` with `BCX_FROZEN_CODE`, `REVISION_OUT` and
`JULIA_BIN` set to the same frozen deployment and output root. `CV_WORKERS=8`
uses up to 32 chain threads. The pool reuses successful inner checkpoints,
checks source/data/settings fingerprints, and completes the original final fit
and PH workflow after all outer folds finish. It refuses to overlap live
clinical workers on that output root. `partition_logs/pool_state.json` records
completion and the preservation of preexisting checkpoint hashes.

When spare cores become available, Linux deployments may advance up to two
pending stages with `scripts/run_t1d_early_stages.py --project FROZEN_CODE
--source-arm EXISTING_ARM --julia JULIA_BIN --stages full_tuning,1`. The stage
worker uses `flock` so a later queued worker waits, then reuses the completed
stage. Do this only for stages that have not already started, using the same
frozen project. The original pool still assembles summaries and final fits.
Account for four chain threads per additional worker; reduce another pending
queue's concurrency if necessary (`RHS_WORKERS=3` for the RHS follow-up wrapper).
