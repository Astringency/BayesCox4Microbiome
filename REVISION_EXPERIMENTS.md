# Revision experiments: implementation and execution record

This record tracks the requested seven additions and their actual verification.
The existing simulation, nested-CV and coverage changes remain part of the project.
An item is complete only when the code, scientific definition, tests and requested
execution artifacts have been verified. Small smoke tests are not formal results.

**Final status, 2026-10-12:** all 31 requested simulation cells are complete at
50 repetitions, comprising 1500 distinct fits (850 Cox-RHS, 650 Prox-SSLASSO).
Every fit retains 4 x 5000 draws after 2000 warm-up per chain and has an
independent n=1000 test cohort. All four T1D method/adjustment arms are complete;
20 outer folds and 60 saved-posterior C-indices were independently verified.
Historical metrics and diagnostics were recomputed for 3800 saved runs without
posterior resampling. All six experiment/reevaluation launchers are available.

The [final results archive](results/revision_20261011/reports_final_all_simulations.zip)
contains 50 SHA256-verified report and audit files. Its primary summaries retain
all eligible repetitions, including poor-mixing fits; 3971 kernel-invalid or
uncertifiable historical/early fits are excluded and retained separately.
The execution notes below preserve the intermediate snapshots and verification
history. The final zero-sum results and their limitations appear at the end.
The [final delivery audit](results/revision_20261011/delivery_audit_final_20261012.json)
maps all seven requirements and A/B/C to their verified evidence.

## Required experiment matrix

| Item | Data/settings | Required comparisons and outputs | State |
|---|---|---|---|
| 1 | Weibull, n=400, p=50, 6 true signals | Prox sum-zero lambda 1e-2, 1e-3, 1e-4, 1e-5; all other settings fixed; both priors | Complete: all eight cells have 50 repetitions and paired comparisons |
| 2 | Weibull, n=400, p=50, 6 true signals | Cox-RHS prior sparsity guesses 3, 6, 12; paired data and sampling seeds | Complete: all three cells have 50 repetitions; p0=6 shares the corrected lambda baseline |
| 3 | Weibull, n=200, p=50, 6 true signals | S1 beta=0.35 beta0; S2 six correlated nulls (latent correlation 0.6); S3 multinomial counts, median depth 5000, half-minimum zero replacement and closure | Both methods complete 50 per scenario; SSL convergence limitations reported |
| 4 | Paper zero-sum section: Gompertz, n=400, p=150, 12 true signals | none, soft_center, prox 1e-3, prox 1e-6, exact; both priors; per-draw and per-replicate sum statistics | Complete: all ten cells 50/50; full zero-sum statistics and convergence limitations reported |
| 5 | T1D | Joint Sex and Antibiotics adjustment with weakly informative priors; adjusted/unadjusted comparison; nested CV; PH diagnostics | Complete at the specified budget for both methods and arms; sampling and PH limitations reported |
| 6 | Both priors, new and historical runs | Rank-normalized split/folded R-hat, bulk/tail ESS, HMC acceptance/divergences, selection mixing, runtime | Complete: 3800 historical replays and all 1500 eligible new-fit summaries; unavailable historical transitions remain missing |
| 7 | Weibull, n=200, p=50, 6 true signals | Independent shrinkage, planted phylogenetic graph, misspecified graph; same generated data | Complete: three RHS cells and the SSL reference each 50/50; SSL convergence limitations reported |

Requested execution: finish items 1, 2, 5, 7; deploy large runs for 3, 4 and
historical reevaluation for 6, then inspect completion and artifacts. All suites
must have reusable scripts under `scripts/`.

## Design decisions and checks

- The user requires reusing existing posterior results whenever metrics can be
  recomputed. Inventory and match data/prior/sampler settings before scheduling
  any formal fit. Write reevaluation outputs separately from historical runs.
  Missing historical HMC transition diagnostics remain unavailable; do not
  rerun otherwise reusable fits just to manufacture these diagnostics.

- Paper reference inspected: `bayescox4microbiome_main.tex`, section "Effect of
  the zero-sum constraint", lines approximately 417-435. The original paper uses
  50 repetitions, 4 chains, 2000 warm-up and 5000 retained draws per chain.
- Existing simulation summaries use posterior means/medians/MAP, independent
  n=1000 test cohorts, coverage and interval-based false discovery proportions.
- The user confirmed the formal scale: 50 repetitions, 4 chains, 2000 warm-up
  and 5000 retained draws per chain. Smoke tests are for validation only.
- The simulation true signal count (`nonzero`) is distinct from prior sparsity
  guess (`p0`). The user confirmed that the p0 ablation applies only to methods
  whose prior uses p0 (Cox-RHS here). Prox-LASSO and Prox-SSLASSO retain their
  existing priors; do not create a p0-to-Beta-prior mapping for these methods.
- Exact constraints must be imposed during inference, not by postprocessing
  sampled coefficients. Their prior normalization and original coefficient
  scale require explicit treatment.
- Historical HMC divergences cannot be recovered from coefficient samples if
  transition diagnostics were not saved; report unavailable instead of zero.
- S3 event times must use the latent composition; count sampling is a measurement
  stage. Train-derived zero replacement is reused on held-out observations.
- CAR comparisons require a planted graph used in the data-generating process,
  not an abundance-distance graph relabeled as known phylogeny.

## Remote execution

- Read-only SSH access verified on 2026-10-11 to the user-designated server.
- Historical root: `/large_storage/zhangxf/BayesCox/results/` (contains
  `simulation/` and `constraint/`). 112 logical CPUs; Julia, R and tmux available.
- Credentials are supplied interactively and must not be written in code,
  command scripts, reports, version control, or experiment metadata.
- New deployment and results must use isolated versioned paths. No deletion of
  existing server content is authorized or needed.
- Initial deployment: `/large_storage/zhangxf/BayesCox/revision_20261011/code_v1`,
  with `source_manifest.json` (SHA256 per source file). Existing code is at
  `/research_data/users/zhangxifeng/C03Julia/BayesCox4Microbiome` and is untouched.
- Use Julia 1.11.7 at
  `/research_data/users/zhangxifeng/.julia/juliaup/julia-1.11.7+0.x64.linux.gnu/bin/julia`;
  the server's default `julia` is 1.10.10 and does not match this project.
- `bayescox_revision_diag` tmux runs `scripts/run_revision_diagnostics.sh`.
  Outputs: `revision_20261011/results/historical_diagnostics`; log and exit
  status: `revision_20261011/logs/historical_diagnostics.{log,exit}`.
- Historical zero-sum Cox-RHS cells each have 50 saved metadata files under
  soft_center, prox, none, prox1e-6. Prox-SSLASSO has 50 in simulation/prox.
  Old metadata omits `prox_sum0_lambda` and `noncentered`. The HS shell script
  uses the default sum-zero lambda 1e-3. The current proximal-prior shell script
  specifies 1e-6, but all three recorded historical source commits specify 1e-3.
  The latter matches a stored-density replay; do not label historical SSL as
  1e-6 from the current shell script. Historical SSL HMC is kernel-invalid anyway.
- `code_v2` adds the new DGPs and ablation scripts. Its two sampler source files
  were then updated to make the threaded target binding local; the updated
  source hashes are in its manifest. The original `code_v2.tar.gz` predates
  that two-file correction; use the deployed source manifest when reproducing.
- Initially launched Prox-SSLASSO jobs (subsequently stopped): `bayescox_exact_ssl` (code_v1),
  `bcx_ssl_sensitivity_1e-2/1e-3/1e-4/1e-5`, and
  `bcx_ssl_scenarios_S1/S2/S3` (code_v2). Job logs and exact commands for the
  latter groups are in `revision_20261011/results/job_logs/`.
- Cox-RHS exact initial HMC run was stopped after poor chain mixing was found
  (first replicate maximum beta Rhat 3.114; two chain step sizes collapsed to
  ~1e-212 and ~1e-234). Its completed outputs are preserved and flagged by
  `validation_status.json` as ineligible for the primary summary.
- Both generic explicit HMC and NUTS had an outer `target` binding captured in
  a shared `Core.Box` in their threaded branches. A target owns mutable gradient
  workspaces. Explicit `local target` removes this shared binding. It is a
  plausible explanation for the pathological parallel run; the corrected
  full-size Cox-RHS replicate subsequently passed. Historical fits are not silently
  certified as valid merely because their settings match.
- `bayescox_rhs_fixed_preflight` passed remote parallel and DGP tests and one
  full-size exact Cox-RHS repetition. `bayescox_exact_rhs_fixed` resumes to 50
  in `results/fixed_parallel/zero_sum/exact/cox-rhs`, reusing that first fit.

## Method references

- Gelman et al. (2008), *A weakly informative default prior distribution for
  logistic and other regression models*, doi:10.1214/08-AOAS191. Any adaptation
  of its prior scale to Cox log hazard ratios must be labeled as an adaptation.
- Vehtari et al. (2021), *Rank-normalization, folding, and localization: An
  improved R-hat for assessing convergence of MCMC*, arXiv:1903.08008.
- Grambsch and Therneau (1994), proportional hazards diagnostics using weighted
  residuals; R `survival::cox.zph` provides termwise and global score tests.
- Stan reference manual, sum-to-zero transforms: orthonormal subspace
  parameterization and normalization corrections for unknown scale parameters.

## Verification log

- Before this revision: coverage tests and related regressions passed, and
  small simulations for both priors were verified. These do not establish any
  of the new seven requirements above.
- Exact target: 72 checks passed (geometry, finite-difference gradients,
  conditional normalization/covariance, centered/noncentered and both samplers).
- New diagnostics: 70 checks passed, including retaining transition information
  through CSV and JLD2 saving. New diagnostics use all retained draws even when
  exported sample CSVs are thinned; historical diagnostics use available draws.
- Complete current local test suite: 43 testsets, 910 checks passed (exit 0),
  log `results/check/revision_tests_current.log`.
- End-to-end exact smoke: both methods, Gompertz n=24 p=8 nz=2, two chains,
  8 warm-up and 16 retained per chain. Largest coefficient-sum residuals:
  Cox-RHS 4.72e-16, Prox-SSLASSO 5.55e-17. Independent test C-index and coverage
  outputs also generated. This is not evidence of formal statistical performance.
- Historical reevaluation smoke completed on four existing runs without any
  resampling (`results/check/revision_historical_smoke`).
- DGP tests: 43 passed, including exact preservation of baseline draws for a
  fixed seed, S2 positive definiteness and latent correlation, S3 count totals,
  zero replacement, independent test data and prevention of incompatible reuse.
- Counts end-to-end smoke completed for both priors. For its small n=24 cohort,
  zero prevalence was 18.5% and the empirical depth median was 5490; these are
  validation results, not formal inferential results.
- C-index regression after DGP changes: 254 checks passed.
- Parallel sampler regression: two lowered-code checks and eight serial/parallel
  draw/log-density equality checks passed with four threads, locally and remotely.
- Historical reevaluation finished 3800 saved runs, exit 0. No model was resampled.

## Audit update: 2026-10-12

- User confirmed that only Cox-RHS receives CAR ablations. Prox-SSLASSO is one
  no-CAR reference on the same planted-phylogeny DGP.
- The planted balanced tree groups the six signal leaves. Latent Gaussian
  covariance is exp(-patristic_distance/4). Correct CAR edges use exp(-distance/2)
  for distance <=4, diagonal zero. A fixed seed-9017 permutation of leaf labels
  gives the wrong graph while preserving its weights and spectrum. Both groups
  use exactly the same observations. Nineteen CAR tests and pipeline smoke passed.
- Fixed parallel Cox-RHS exact repetition 1 passed: beta max Rhat 1.00268, minimum
  bulk ESS 14379.5, zero retained-sampling divergences. It was reused when resuming
  the same directory to 50 repetitions. Formal runs use frozen code_v2/code_v3.
- Baseline Cox-RHS repetitions 33 and 50 have beta Rhat approximately 2.29 and
  bulk ESS approximately 5. Other 48 baseline repetitions have beta Rhat <=1.01.
  `run_baseline_repairs.sh` selects only those two IDs via `--rep-ids`, writes to
  a separate repair root, and retains all original historical files.
- Joint clinical target keeps clinical coefficients outside the compositional
  prior, zero-sum penalty and CAR. Binary coding: M vs F, Tylosin vs Control.
  Independent Student-t(3,0,2.5) priors adapt the weakly informative regression
  prior principle to Cox log hazards. This is explicitly an adaptation, not a
  claim that Gelman et al. derived this exact Cox prior.
- T1D scale remains the existing 4 chains / 800 warm-up / 800 retained draws,
  separately from the user-approved 50 x 4 x (2000+5000) simulation scale.
  Paired 5x5 nested CV uses the existing candidate grids, with training-only p0
  and preprocessing. Individual successful inner fits now checkpoint too.
  Existing unadjusted Cox-RHS full-data posterior is reused descriptively; it
  cannot replace the required training-fold fits.
- Clinical-only frequentist PH check on all 173 T1D subjects: Sex p=0.000963,
  Antibiotics p=0.437591, global p=0.003615. These are not joint microbiome-model
  PH results. Joint score-calibration diagnostics will be labeled exploratory,
  since the posterior microbiome score was estimated on these same subjects.
- Before the SSL indicator correction, the full four-thread suite passed 1048
  checks across 52 testsets. Clinical gradient/parallel checks passed 24 tests,
  and clinical nested-CV slicing/prediction passed 42 tests. Those results do not
  substitute for the post-correction regression suite, which passed 1138 checks.

### Confirmed Prox-SSLASSO indicator defect

The explicit HMC/NUTS Gibbs loops construct gamma with `trues(p)` (a BitVector).
`CoxProxSSLassoTarget` previously copied this to a Vector{Bool}. Gibbs updates
therefore did not reach the continuous target, which retained all-true indicators.
The minimal reproduction changes external gamma[1] to false while target.gamma[1]
remains true. This affects the joint posterior, even when coefficient Rhat is good.

The target now preserves its supplied indicator vector using a parametric field.
Fourteen reference/gradient tests and 68 tests comparing saved log densities to
the saved gamma/theta states passed (HMC/NUTS, proximal/exact constraints).
All 14 affected processes launched in this revision were stopped; 170 completed
new outputs were preserved and marked ineligible in validation_status.json.
Stop records are in logs/ssl_gamma_copy_stopped.json. Do not use these as formal
Prox-SSLASSO results. Corrected runs must use a new output root and be validated
before continuation. Existing historical HMC/NUTS SSL results need a kernel audit;
good Rhat alone is not sufficient evidence of a correct posterior.

### Missing-setting provenance

The legacy save_run_outputs writer accepted a complete settings dictionary but
exported only selected entries to metadata.json; with JLD2 disabled, lambda and
noncentered were omitted. This also affected early new deployments. The writer
is now fixed to retain all settings. Frozen running deployments are not mutated.
`backfill_revision_execution_settings.py` writes separate sidecars for known new
launches, while `recover_saved_sum0_lambda.jl` tests whether a historical lambda
can be identified from stored log-posterior values. A current shell script alone
does not prove which lambda generated an older run. In particular the historical
SSL sum distribution contradicts the earlier script-only 1e-6 label; its lambda
was subsequently verified as 1e-3 for repetition 1 by density replay (maximum
residual 5.9e-13 across 30 draws) and the saved source commit's launcher.

The stored commit `6f3543986e85f1a5606bc745da50a607ea322ff8` confirms the historical
HMC stale-indicator defect too. That version recomputed saved logposterior with
the current gamma, despite the continuous kernel using its copied all-true gamma.
Consequently, matching saved log densities by itself does not validate the Markov
kernel. Historical SSL HMC outputs must remain audit/descriptive results, not
the primary corrected-method baseline.

After the gamma correction, all 1138 checks in 54 testsets passed. The corrected
SSL n400/p50 lambda=1e-3 first formal replicate passed (max beta Rhat 1.00116,
min bulk ESS 6211.4) and is reused when continuing to 50. Corrected SSL jobs use
frozen code_v5 and `results/corrected_ssl`; previous invalid outputs remain in
their original directories. The corrected exact p150 first repetition has max
beta Rhat 1.0600 and minimum bulk ESS 109.2. It is retained and resumed to 50 under
the same paper sampling budget; it must not be called a convergence success.
The two Cox-RHS baseline repairs passed their convergence checks. An earlier
manifest combined 48 historical fits and 2 repairs; the subsequent parallel-target
audit below supersedes that reuse decision. The two corrected fits remain reusable.
Historical metric replay finished all 3800 runs with zero failures and no posterior
resampling; metric availability does not establish sampler validity.

T1D Cox-RHS nested CV finished both arms at the existing 800/800 scale. Initial
mean outer C-index: unadjusted 0.569901, adjusted 0.617898. These are provisional
pending sampling-quality review: some inner fits have beta Rhat above 1.1.
The adjusted full-data fit and PH output are complete; its maximum coefficient
Rhat is 1.0332 and minimum bulk ESS 134.0. Corrected SSL arms are still running.
Do not present these
numbers as fully validated final clinical results yet.

`aggregate_revision_diagnostics.py` creates per-run and per-setting diagnostics,
including all-eligible and convergence-screened summaries. Unavailable old HMC
transition information remains missing. `reevaluate_revision_metrics.jl` rebuilds
only simulated train/test observations and computes C-index, coverage, MSE and
empirical FDP from existing coefficient summaries into a separate output tree.

### Audited report assembly and outer-posterior retention

`audit_revision_legacy_ssl.py` verifies the copied BitVector and continuous-target
lifetime at all three recorded commits (fcaa4b95, 6f354398, af40118d). Its JSON
rules exclude 1200 historical SSL HMC runs from primary comparisons, while
retaining their diagnostic and metric rows for audit. This initially excluded
1373 runs including stopped new outputs and two replaced RHS repetitions.
The later RHS audit below expands the exclusions to 3971 distinct runs.

`refresh_revision_reports.sh REVISION_ROOT` assembles audited per-run diagnostics,
performance mean/SD/Monte Carlo SE, and the 31 requested simulation cells. Its
progress table lists missing replicate IDs and the actual denominator for every
metric. All eligible repetitions, including poor-mixing ones, stay in the main
summary. A separately labeled convergence-screened sensitivity summary uses beta
Rhat <=1.01 and both bulk/tail ESS >=400. Historical Cox-RHS prox1e-6 has 0/50
passes on that screen; it is also excluded by the subsequent kernel audit.
Unknown old transition diagnostics are missing, never replaced by zero.

`summarize_t1d_clinical_revision.py` exports paired taxon coefficients, clinical
hazard-ratio transformations, selected full-data parameters, PH tests, inner-fit
quality and a split audit. The ten completed RHS folds have disjoint training
and test sets, inner validation folds partition only outer training subjects,
and adjusted/unadjusted analyses use identical splits. Joint score PH is
exploratory: adjusted RHS Sex p=0.000934 and global p=0.00844.

Earlier frozen nested-CV deployments saved outer scores but not their posterior
draws. The local implementation now retains outer posteriors and diagnostics;
266 C-index regression checks passed. `audit_t1d_outer_fits.jl` recovers only
missing outer posterior diagnostics with the existing selected configuration,
split and random seed; no inner tuning is repeated. It writes a separate output
and requires all three C-indices to reproduce cached values within 1e-12. Two
small clinical folds reproduced exactly. The ten full-size RHS fold replays have
completed in `results/T1D_outer_audit`; original scores and checkpoints are unchanged.

Report utilities are isolated under `analysis_tools_v1`, outside frozen model
deployments, so they do not invalidate ongoing CV code fingerprints. The new
outer-posterior persistence change is local only; active code_v4/code_v5 remains
unchanged.

### Partitioned clinical execution

The corrected SSL clinical grid has 57 candidates per outer fold. To use the
CPU slots freed by completed Cox-RHS jobs, the two serial SSL clinical drivers
were stopped after retaining 70 adjusted and 160 unadjusted successful inner
fits (230 total). No completed checkpoint was removed. The stop record is in
`logs/t1d_partition_serial_stop.json`.

`run_t1d_cv_partition.jl` runs one disjoint outer fold or full-data tuning stage
against the SAME frozen `code_v5`. Its calculated data/settings/code fingerprint
matches both existing manifests before fitting anything. Eight workers are
scheduled by `run_t1d_partition_pool.py`, launched through tmux session
`bcx_t1d_partition_pool`; each fit still uses four chains and 800/800 iterations.
Each worker owns a separate stage directory. The coordinator verifies hashes
of all 230 preexisting successful inner checkpoints, then resumes the original
driver to assemble CV summaries, fit the final full-data model and run PH tests.

The worker passed 84 checks covering both priors and both clinical arms: complete
posterior samples, selected configurations and all three outer C-indices agree
exactly with serial evaluation; completed folds are skipped; changed sampler
settings are rejected. The checks pass without Windows mmap cleanup warnings.
This changes execution order only, not folds, tuning grids, seeds or estimands.
Full-data tuning remains separate and is never passed into outer-fold selection.

After integrating the partition worker and outer-posterior retention, the full
four-thread local suite passed **1234 checks across 55 testsets** (exit 0), log
`results/check/revision_tests_partition.log`. On-server monitoring confirmed
eight live stage-worker processes and unchanged hashes for all 230 preexisting
successful inner checkpoints.

All ten Cox-RHS outer diagnostic replays completed and reproduced the cached
scores exactly. Their beta Rhat/ESS limitations are retained in the clinical
quality table. The historical full-data Cox-RHS posterior remains reused.

The clinical summarizer also exports `paired_cv_folds.csv` and
`paired_cv_comparison.csv`. Cox-RHS posterior-mean C-index has adjusted-minus-
unadjusted mean difference 0.047997 over the five matched outer folds, with
descriptive fold SD 0.022239. Overlapping training sets mean this SD is not an
independent-fold confidence interval. The reused full-data RHS fit has maximum
beta Rhat 1.25745 (all parameters: 1.44917) and minimum beta bulk ESS 101.50;
its absence of the parallel-kernel defect does not establish convergence.

`summarize_revision_contrasts.py` computes paired setting-minus-reference
differences for lambda, p0 and CAR using matching repetition IDs and test seeds.
It exports paired SD and Monte Carlo SE, with actual pair counts for unfinished
cells. References are lambda=1e-3, p0=6, and independent shrinkage respectively.

The weak clinical prior follows the independent Student-t family and scale
principle in [Gelman et al. (2008)](https://doi.org/10.1214/08-AOAS191). Their
logistic-regression default is Cauchy(0,2.5), not Student-t(3,0,2.5). Choosing
three degrees of freedom (finite variance) and applying it to Cox log hazards
are explicit project adaptations. PH testing follows the score-test interface
documented by [`survival::cox.zph`](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/cox.zph.html),
which references Grambsch and Therneau (1994). No claim of a direct Bayesian
taxon-wise PH test is made.

### Completed CAR comparison and SSL reference

All three Cox-RHS cells completed 50 repetitions and exited successfully. All
150 repetitions pass the beta Rhat <=1.01 / bulk and tail ESS >=400 screen.
SHA256 comparisons confirm identical training and test CSVs across all three
graphs and all 50 corrected SSL reference fits. The current
`results/data_identity_audit` tables verify 50 matched groups with four runs and
four saved training CSVs per group. The older CAR-only audit is retained as an
earlier partial checkpoint.

Posterior-mean estimates, with 1000 independent test subjects per repetition:

| Method / CAR graph | Repetitions | Test C-index | MSE | 95% coverage, nonzero coefficients | Empirical FDP |
|---|---:|---:|---:|---:|---:|
| Independent | 50 | 0.911579 | 0.005372 | 0.966667 | 0.000000 |
| Correct | 50 | 0.911130 | 0.006017 | 0.963333 | 0.002857 |
| Misspecified | 50 | 0.909029 | 0.008067 | 0.963333 | 0.011429 |
| Prox-SSLASSO / independent reference | 50 | 0.911013 | 0.006109 | 0.966667 | 0.000000 |

Relative to independent shrinkage, correct-graph mean paired test-C-index
difference is -0.000449 (Monte Carlo SE 0.000057); wrong-graph difference is
-0.002550 (MCSE 0.000177). Paired MSE differences are +0.000646 (MCSE 0.000115)
and +0.002695 (MCSE 0.000292), respectively. Under this prespecified DGP,
correct CAR does not improve on independent shrinkage; wrong CAR degrades
performance more. Do not reinterpret this experiment as evidence that correct
CAR necessarily improves prediction. The corrected SSL reference completed all
50 fits and exited with status 0; 33/50 pass the beta convergence screen. Its main
row retains all 50 fits, while the separately labeled screened summary retains
33. Do not ignore the remaining 17 sampling-quality failures in cross-method
interpretations. The worst SSL beta Rhat is 1.60354, with minimum beta bulk/tail
ESS 58.49/51.74. No retained-sampling HMC divergences were recorded; this does
not establish adequate mixing. Among the 33 screened SSL fits, mean test C-index
is 0.911373; this subset is a labeled sensitivity analysis, not a replacement
for the main 50-repetition estimate.

### Historical Cox-RHS parallel-target audit and minimal follow-up

All 13 source commits recorded by the 2600 historical explicit-HMC Cox-RHS fits
contain the shared threaded `target` binding. Old run metadata does not record
whether individual chains executed in parallel; the recorded simulation launcher
requests four parallel chains. Normal Rhat cannot certify target ownership.

`probe_legacy_chain_ownership.jl` calls the historical dispatcher with a sampler
stub and records the mutable objects supplied to its four chains. Across 1000
calls per baseline commit, shared objects occurred 1, 2, 0 and 1 times for commits
a5f9ff57, 59bb5ded, 1402e3a9 and fcaa4b95 respectively. This confirms the binding
defect can reach sampler calls. These are dispatch-only stress tests, not posterior
fits or an estimate of the proportion of historical fits affected. Thirty-two
small actual-HMC probes did not observe object sharing; this does not rule out a
scheduling-dependent race. The source audit conservatively excludes unverifiable
historical HMC runs without claiming that every historical posterior is corrupted.

`audit_revision_legacy_rhs.py` combines 13 RHS and 3 SSL source rules into
`results/legacy_kernel_audit.json`. All 3800 historical simulation runs remain in
audit tables. Together with 170 invalid early SSL outputs and one initial RHS
exact output, 3971 distinct runs are excluded from primary comparisons. The old
`resolved_baseline_reuse.json` is retained but superseded by
`baseline_reuse_status.json`; report refresh no longer consumes it.

Only the requested cells are refitted. `run_rhs_kernel_followup.py` waits for the
five existing RHS lambda/p0 workers to finish successfully, then uses at most five
four-chain workers. It fits baseline IDs 1:50 excluding the already-corrected 33
and 50, in two disjoint 24-repetition shards, plus 50 repetitions for each of none,
soft_center, prox1e-3 and prox1e-6. Thus only 248 additional fits are required, not
all 2600 historical RHS fits. Existing corrected exact and other new fits continue.
The coordinator is deployed in tmux `bcx_rhs_kernel_followup`; commands, progress
and exit codes are saved in `results/rhs_kernel_followup_logs`.

`resolve_revision_fixed_baseline.py` validates settings, seeds and all 50 IDs,
then writes `resolved_fixed_rhs_baseline.json` with 2 reused corrected fits and 48
new fits. It records convergence diagnostics but does not discard or repeat fits
to pass a convergence screen. The baseline remains incomplete until those fits
finish. The historical T1D full-data RHS fit records serial NUTS (`multichains=false`)
at commit 451ad1a6 and is outside this simulation HMC exclusion; it remains reused.

The combined-audit report refresh completed without errors (4724 runs at the
snapshot time, 3971 excluded, 6 of 31 requested cells complete). The local archive
`results/revision_20261011/reports_snapshot_20261012_audited.zip` contains 25 hashed
files. Its verifier confirms all 3800 historical runs are audit-only, the fixed
baseline is 2/50, and reported progress denominators and bounded metrics are
consistent. Later snapshots must be regenerated rather than treating these counts
as final. The reusable snapshot utility is `scripts/snapshot_revision_reports.py`.

The baseline reuse checker now refuses known affected historical source commits.
It prefers the completed fixed-kernel manifest when present. The sensitivity
runner shares the lambda=1e-3/p0=6 output directory and uses a Linux `flock` lock
to prevent concurrent duplicate fitting. These local runner improvements are not
patched into frozen active deployments. Follow-up workers use their recorded
fixed-code commands and separate shard directories. SHA256 verification found
no changed files in either active clinical deployment (code_v4: 158 files;
code_v5: 162 files); all 230 preexisting successful inner checkpoints also retain
their original hashes.

The Linux integration check `test/revision_runner_smoke.py` passed using a
non-sampling Julia stub: concurrent lambda and p0 launches serialize the common
baseline, the second call reuses its completed output, and the formal 50-repeat /
4-chain / 2000-warm-up / 5000-draw flags remain unchanged. No posterior was sampled
by this check. Its retained server scratch directory is
`/tmp/bayescox_runner_smoke_xkxrjt4l`.

`audit_revision_data_identity.py` adds a paired-data audit to every report refresh.
On the first audited snapshot, all 753 primary-eligible runs passed: 256 distinct
DGP/repetition groups, of which 191 had multiple fits, and 423 runs with saved
training CSVs. It checks identical test-file hashes and DGP specifications,
recorded training/test seeds, exactly 1000 test observations, and identical saved
training CSVs. Baseline training CSVs are unavailable; their audit is limited to
common DGP specifications/seeds rather than asserting unobserved byte equality.

### Execution scheduling update at 02:12 on 2026-10-12

All five nonbaseline Cox-RHS lambda/p0 jobs finished 50 repetitions and exited
with status 0. Before it started any posterior workers, the waiting RHS follow-up
coordinator was stopped and restarted with `--workers 3`; the command/PID/source
check and reason are recorded in `logs/rhs_queue_reschedule.json`. Both 24-run
baseline shards and the none-constraint job are now running. Other zero-sum jobs
remain queued. No active posterior fit was stopped during this change.

The eight freed chain threads advance two pending unadjusted T1D stages (outer
fold 1 and full-data tuning) through `run_t1d_early_stages.py`, while the original
eight-worker pool continues. The advanced outer fold reuses its 160 successful
inner checkpoints. The Linux stage worker now holds a per-stage `flock`; a later
queued worker waits and then reuses the finished stage. Ten concurrent lock checks
passed on the server without sampling. The lock update is outside frozen code_v5
and changes neither its fingerprint nor the model. The original pool will still
assemble all summaries, full-data fits and PH tests. Extra-stage commands, logs,
cache hashes and status are under the unadjusted arm's `early_stage_logs`.

The first two newly completed baseline repetitions (IDs 1 and 25) each retain
20,000 draws; maximum beta Rhat is 1.00117/1.00069 and minimum bulk ESS is
19465/17339. Together with the two reused corrected fits, the baseline is 4/50
in `reports_snapshot_20261012_0217.zip`. That snapshot verifies 31 file hashes,
12 complete simulation cells, no aggregation errors, and matched data for all
938 eligible runs. It includes the paired clinical-CV tables and data-identity
audit. Aggregation now requires each new run's `complete.txt`, so a partially
written posterior output cannot prematurely count as a completed repetition.

### Reusable source deployment: code_v6

The latest local source, all experiment entrypoints, metrics/diagnostic tools and
tests are packaged in `code_v6.tar.gz` and deployed to
`/large_storage/zhangxf/BayesCox/revision_20261011/code_v6`. All 186 manifest entries
match their deployed SHA256 hashes. Archive SHA256 is
`c6ea462ef4a4776cf4db495c8181d3bfebe603cf9332b07b1e11e018663f33b9`.
The package loaded successfully on Julia 1.11.7, and its ten Linux stage-lock
checks passed (exit 0). No posterior was sampled by this deployment check.
`code_v6_ready.json` records the verification; logs are under
`logs/code_v6_smoke.*`.

This package is for new invocations with separate output roots. Existing clinical
checkpoints remain on their recorded frozen code_v4/code_v5, and ongoing revision
workers are unchanged. The package's `deployment_info.json` explicitly documents
this distinction. Analysis-only report refreshes continue through
`analysis_tools_v1/refresh_revision_reports.sh` and reuse the saved posteriors.

### Independent saved-posterior C-index verification

`verify_t1d_saved_cindex.py` reconstructs the recorded T1D preprocessing and
clinical coding, obtains mean/median/MAP coefficients from the saved draws, and
independently calculates Harrell C-index on the recorded outer test subjects.
It checks disjoint outer training/test sets, inner validation partitions contained
in outer training, paired splits across methods/adjustments, and all 3200 retained
draws per outer fit. This check never launches posterior sampling.

The first check verified all 39 scores from the 13 available outer folds, with
absolute differences <=1e-12. Seven SSL folds were still pending; passing the
available checks is not evidence that all clinical fits have completed or mixed
well. Use `--require-complete` for the final check of all 20 folds / 60 scores.
The report refresh now runs this verifier, and snapshots include its CSV and
JSON audit. `T1D_DATA` can override the frozen code_v5 data path for a different
deployment. These analysis-only additions live outside the sealed code_v6 package.

### Report snapshot at 02:45 on 2026-10-12

`reports_snapshot_20261012_0245.zip` contains 33 verified file hashes, 5016 run
records (3971 excluded; 1045 eligible), zero aggregation errors, and 15/31 complete
simulation cells. The corrected shared RHS baseline is 30/50. The independent
clinical score audit now covers 14/20 outer folds and all 42 available scores.

All six S1/S2/S3 cells have 50 formal repetitions. The beta convergence screen
(maximum rank Rhat <=1.01; minimum bulk/tail ESS >=400) passes RHS 49/50, 50/50,
50/50 and SSL 23/50, 1/50, 2/50 for S1, S2, S3 respectively. Main summaries retain
all 50 repetitions and report these limitations; screened subsets do not replace
the main results or establish convergence of the remaining parameter blocks.
For SSL S1, nonzero-coefficient coverage is 0 and mean FDP is 0: a small FDP alone
must not be interpreted as successful signal recovery. No fits were repeated to
change these diagnostic or performance results.

### Advance of the last queued zero-sum cell

At 02:50, spare CPU capacity was used to advance repetition 1 of Cox-RHS
Gompertz n=400/p=150/prox lambda=1e-6, before its last-in-queue job started.
`run_zero_sum_revision.sh` used the frozen code_v5, `REPS=1`, and the unchanged
four-chain/2000-warm-up/5000-retained-draw scale and output directory. It exited
0 before the queued job started. All 20,000 coefficient draws are finite and
the coefficient, metric, coverage, FDP, zero-sum and transition outputs exist.
The original queue uses resume and will reuse this fit, then complete IDs 2:50.
Command and verification records are `logs/rhs_zero_prox1e-6_first.*`.

For this single repetition, the pooled coefficient-sum mean is 0.0004705,
SD 0.0123179, and central 95% interval [-0.0237309, 0.0245511]; mean absolute
sum is 0.00982259. Maximum beta Rhat is 1.03291, minimum bulk ESS 168.16 and
minimum tail ESS 496.43, with no retained-sampling divergences. This fails the
prespecified beta convergence screen and is retained with that limitation.
These are first-repetition output checks, not a completed 50-repetition result.

The saved-score verifier now also audits cross-fold test-set coverage. Both
completed RHS arms cover all 173 subjects exactly once. Available SSL folds have
no repeated test subjects; full coverage is checked after all five folds finish.
The original 230 successful inner-cache hashes and the early-stage subset of 160
remain unchanged.

### Completed lambda and p0 comparisons at 03:06 on 2026-10-12

All eight lambda cells and all three p0 cells contain 50 formal repetitions.
Their shared Cox-RHS baseline has 2 reused corrected fits and 48 new fits;
`resolved_fixed_rhs_baseline.json` verifies all IDs, settings and seeds. Every
lambda/p0 contrast now has 50 matched test cohorts. All 34 files in
`reports_snapshot_20261012_0306.zip` pass their SHA256 checks. The snapshot has
21/31 complete simulation cells, 1118 eligible runs and no aggregation errors;
clinical adjustment and the large zero-sum runs remain incomplete.

The following are posterior-mean estimator results, averaged over all 50 runs.
Coverage is marginal 95% coverage among the six truly nonzero coefficients.
The sum statistic is the across-run mean of posterior mean absolute coefficient
sum. Bulk ESS is the across-run mean of each run's minimum beta bulk ESS.

| Method | Lambda | Test C-index | MSE | Nonzero coverage | Mean absolute sum | Min-beta bulk ESS, averaged |
|---|---:|---:|---:|---:|---:|---:|
| Cox-RHS | 1e-2 | 0.873211 | 0.003164 | 0.9567 | 0.388794 | 14113.8 |
| Cox-RHS | 1e-3 | 0.873296 | 0.003020 | 0.9467 | 0.171552 | 15532.4 |
| Cox-RHS | 1e-4 | 0.873314 | 0.003013 | 0.9500 | 0.056270 | 9271.1 |
| Cox-RHS | 1e-5 | 0.873317 | 0.003009 | 0.9500 | 0.017840 | 1305.6 |
| Prox-SSLASSO | 1e-2 | 0.872154 | 0.003764 | 0.9567 | 0.413976 | 4972.0 |
| Prox-SSLASSO | 1e-3 | 0.872228 | 0.003609 | 0.9500 | 0.173500 | 4984.9 |
| Prox-SSLASSO | 1e-4 | 0.872217 | 0.003605 | 0.9533 | 0.056315 | 4904.7 |
| Prox-SSLASSO | 1e-5 | 0.872214 | 0.003601 | 0.9533 | 0.017824 | 5511.5 |

In this DGP, decreasing lambda below 1e-3 substantially tightens the coefficient
sum while changing test C-index very little: paired differences versus 1e-3
have magnitudes <=2.12e-5 and paired Monte Carlo SEs 1.10e-5 to 1.39e-5.
Cox-RHS mixing becomes less efficient at 1e-5; 49/50 runs pass the beta screen,
compared with 50/50 at the other three values. All four SSL cells pass 50/50.
Cox-RHS at 1e-2 has one retained-sampling divergence across its 50 runs; the
other seven lambda cells have none. These observations do not select a new
parameter using the test cohorts.

| Cox-RHS p0 | Test C-index | MSE | Nonzero coverage | Empirical FDP |
|---:|---:|---:|---:|---:|
| 3 | 0.873339 | 0.002991 | 0.9500 | 0.000000 |
| 6 | 0.873296 | 0.003020 | 0.9467 | 0.002857 |
| 12 | 0.873205 | 0.003074 | 0.9467 | 0.002857 |

Prediction is stable over these sparsity guesses in this DGP. Relative to p0=6,
paired C-index differences are +4.25e-5 (Monte Carlo SE 6.50e-6) for p0=3 and
-9.15e-5 (8.64e-6) for p0=12; their small numerical magnitude is distinct from
Monte Carlo precision. Beta-screen counts are 50/50, 50/50 and 49/50. All fits
remain in the primary comparison. Prox-SSLASSO has no p0 ablation.

Full train/test C-index, MSE, coverage (all/nonzero/zero), FDP and contingency
counts, with across-run SD and Monte Carlo SE, are in `requested_metrics.csv`.
Mean/median/MAP paired differences are in `paired_contrasts.csv`; convergence
and zero-sum details are in `diagnostics_by_setting.csv`. The independent saved
clinical score audit now verifies 48 scores from 16 completed outer folds.

At 03:12, all ten zero-sum cells have at least one completed formal repetition.
The first Cox-RHS soft-center and prox(1e-3) fits retain 20,000 draws each, have
no retained-sampling divergences, maximum beta Rhat 1.00104/1.00185, and minimum
bulk ESS 12257/10920. They are output/convergence checks, not complete 50-run
comparisons. The independent clinical audit subsequently passed
`--require-complete`: all 60 saved-posterior scores match, and each of the four
method/adjustment groups covers all 173 test subjects exactly once.

### Automatic reporting after the large simulations finish

`finalize_revision_reports.py` is deployed outside the frozen model packages and
runs in tmux `bcx_final_reports`. It polls the seven remaining simulation/queue
sessions and the clinical coordinator, checks their successful exit records,
then refreshes saved results and requires all 31 requested simulation cells to
contain 50 repetitions and all 60 clinical scores to have passed their audit.
Only then does it write `reports_final_all_simulations.zip`. It never samples
or restarts a failed job; failures are saved in `logs/final_reports_state.json`.
The actual eight-session status check passed, and the live watcher was verified
waiting on its prerequisites. Its eventual complete archive is not yet available.
Report refreshes now share a Linux file lock to serialize simultaneous requests.

### T1D completion and verified A-results archive

The corrected SSL coordinator completed all 12 stages and both final analyses
with exit code 0. All 230 preexisting successful inner checkpoints were preserved.
Both methods now have adjusted/unadjusted full-data coefficients, paired 5x5
nested CV, and clinical-only plus exploratory joint-score PH checks. No full-data
tuning score is used as test performance.

| Method | Unadjusted outer C-index | Adjusted outer C-index | Paired mean difference | SD of fold differences |
|---|---:|---:|---:|---:|
| Cox-RHS | 0.569901 | 0.617898 | 0.047997 | 0.022239 |
| Prox-SSLASSO | 0.553489 | 0.625252 | 0.071763 | 0.034174 |

These use posterior-mean risk coefficients. Fold SDs describe variability and
are not independent-fold confidence intervals. The legacy summary SE columns
remain the arithmetic SD/sqrt(5); they must not be used to claim independence
between overlapping training folds.

Full-data parameters (separate from the independently selected outer-fold
parameters) are:

| Method/arm | Full-data settings |
|---|---|
| Cox-RHS unadjusted | Reused historical fit: local scale 0.5, slab df 4, slab s2 2, p0 16, sum-zero lambda 1e-6 |
| Cox-RHS adjusted | Local scale 1, slab df 4, slab s2 2, automatic training-data p0 16, sum-zero lambda 1e-6 |
| SSL unadjusted | Spike scale 0.1, slab scale 5, L1 smoothing 1e-3, alpha InvGamma(shape 1, scale 22), sum-zero lambda 1e-6 |
| SSL adjusted | Spike scale 0.04, slab scale 2, L1 smoothing 1e-5, alpha InvGamma(shape 1, scale 22), sum-zero lambda 1e-6 |

Adjusted posterior mean log hazard ratios are -0.3701 (Sex M vs F) and 0.5572
(Tylosin vs Control) for RHS, and -0.3605 and 0.5466 for SSL. Exponentiated
posterior mean log coefficients are 0.6907/1.7458 and 0.6973/1.7273 respectively;
these are not posterior mean hazard ratios. Credible intervals are in
`T1D_summary/clinical_coefficients.csv`.

Sampling and model adequacy remain limitations. Adjusted full-data maximum beta
Rhat is 1.0332 for RHS and 1.0585 for SSL, with minimum bulk ESS 134.0/74.7 and
tail ESS 31.0/21.2. The reused unadjusted RHS fit is poorer (beta Rhat 1.2575),
and some inner/outer fits also fail the diagnostic screen. The adjusted joint-score
PH checks give Sex p=0.000934 (RHS) and 0.000907 (SSL), with global p=0.008444
and 0.009083. These frequentist score-calibration checks are exploratory and do
not constitute taxon-wise Bayesian PH tests; the detected Sex PH departure and
sampling limitations preclude treating the reported improvements as a fully
validated clinical model. Adjusted SSL also has one full-data PSIS Pareto-k warning
above 0.7; the outer C-indices do not use PSIS.

`reports_A_complete_20261012.zip` contains 38 hash-verified artifacts: all A
simulation cells, clinical tables, saved-score audits, kernel/data/source/cache
audits and current background-job state. The delivery audit verifies 15 A cells
with 50 runs each (700 distinct simulation fits, accounting for the shared RHS
baseline), 20 clinical outer folds and 60 independently reproduced scores.
S1/S2/S3 have all 300 fits; all ten zero-sum cells have formal outputs but their
full 50-run series are still running or queued. Historical replay covers 3800
saved runs with no posterior resampling. The local suite passed 1234 checks in
55 testsets; source manifests for code_v2 through code_v6 and reporting tools
match the deployed files. Runtime values come from the shared server, not a
controlled hardware benchmark.

### Advancing the remaining queued zero-sum cell

After A completed, the server load was approximately 33 cores out of 112.
`advance_rhs_zero_queue.py` now advances the queued RHS prox(1e-6) cell to 50
repetitions, reusing the existing repetition 1. Its command retains the same
frozen code_v5, seeds, settings and output root. No frozen source was edited.
The Python dispatcher PID 3455242 is temporarily suspended; its existing sampler
children (none, soft-center, prox1e-3) continue. The helper automatically resumes
the dispatcher after its own worker terminates, and the queued prox1e-6 invocation
will reuse the completed fits. It prevents concurrent writers without suspending
any existing sampler or altering its wall-clock runtime.

Ten process-control checks passed without MCMC: incorrect process identities
were refused, the verified parent could be suspended/resumed, and its child
continued to consume CPU while the parent was suspended. An initial login-shell
launch encountered a Python build without pidfd support and exited before any
process-control or sampling action. The actual launch pins the tested system
interpreter `/usr/bin/python3` and runs in tmux `bcx_rhs_zero_advanced`.
Post-launch verification confirmed continued CPU progress in all three existing
samplers, one new sampler for prox1e-6, and unchanged hashes for the existing fit.
The original dispatcher's `T` state is expected until this advanced job ends.
Do not independently resume/restart it while the advanced worker is active.

The read-only final-report watcher was restarted with the additional job as a
prerequisite. Its final checks now require preservation of existing fit hashes
and successful dispatcher resumption as well as 50 repetitions in every cell.
It is live with nine registered job prerequisites. Command, state, initial-launch
failure and process-control test records are in `logs/advance_rhs_zero_queue.*`.

### Independent saved zero-sum reporting audit

`scripts/verify_zero_sum_saved_statistics.py` independently verified repetition 1
in each of the ten corrected zero-sum cells (4 chains, 5000 retained draws per
chain, 2000 warm-up). All 8940 scalar comparisons passed. The checks cover every
coefficient's posterior mean, median, saved-draw MAP and 95% interval; chain and
pooled coefficient-sum statistics; coverage for all/nonzero/zero coefficients;
interval-based FDP; mean/median/MAP test C-indices; and HMC acceptance/divergence
summaries reconstructed from recorded transitions. Exact draws also satisfy a
coefficient-scale floating-point tolerance. All ten cells use identical test-data
hashes for the matched repetition. This was a read-only audit with no MCMC.

The numerical audit is deliberately scoped to one specified repetition in each
cell. It does not establish posterior convergence or certify all 50 repetitions.
Its output is `results/zero_sum_saved_audit/{saved_statistics_checks.csv,
saved_statistics_audit.json}` and is included in subsequent report snapshots.
The verifier initially encountered an assertion on the saved uppercase `MAP`
label; the verifier was corrected to the existing format, without altering any
scientific outputs. The initial trace and successful rerun are retained in
`logs/zero_saved_audit.log`; the terminal exit record is 0.

### Reusing future repetitions prepared with spare cores

The remaining zero-sum cells are being accelerated with isolated future
repetitions, without restarting the original workers or changing frozen model
code. The pilot prepared RHS prox(1e-6) repetition 50 while its original worker
was still at repetition 8. Its 20000 saved draws passed the publication checks;
all seven previously completed fits kept identical sample/metadata/marker hashes.
An invocation of the same frozen command on the completed staging result printed
`[SKIP]`, exited 0, and preserved both staged and published sample hashes. This
directly verifies reuse rather than relying only on compatible metadata.

Ten publication tests verified no-overwrite behavior, process identity guards,
and automatic resumption. Three concurrent publication tests verified the
per-case lock; candidate-selection tests exclude existing results, staged fits,
attempted repetitions, and repetitions too close to the current original run.
The pilot's actual publication pause was 0.000821 seconds. Copying and posterior
sampling occur before that pause. No existing sampler is paused while waiting
for an assisted fit to finish.

`run_zero_tail_pool.py` was deployed in `bcx_zero_tail_pool` with at most eight
additional four-chain workers. It assists nine existing code_v5 workers;
the exact RHS code_v2 worker does not support `--rep-ids` and is left on its
original queue. Helpers use the same project, arguments, data seeds, 2000 warm-up
and 5000 retained draws per chain. Each successful fit is atomically published
to its original cell's absent run directory. The original queue then reuses it.
Staging files are retained, and no historical content is removed.

`zero_tail_pilot.verified.json`, publication/selection test records and
`logs/zero_tail_assist/pool_state.json` document execution. The read-only
`audit_zero_tail_assistance.py` checks published sample/metadata hashes and
publication pause durations. Diagnostic rows distinguish assisted fresh Julia
processes from original workers, and aggregate tables count assisted fits.
Fresh-process compilation, shared-server load and brief publication pauses mean
the recorded runtime is not a controlled hardware benchmark.

The final-report watcher was restarted with eleven prerequisites, including the
pilot and assistance pool. Before final packaging it requires every assisted fit
to be present exactly once in diagnostic aggregation and to pass its saved-file
audit, in addition to all 31 simulation cells having 50 repetitions.

The first production wave completed all eight assisted fits with exit 0. Together
with the pilot, nine published fits passed the saved-file audit and appeared
exactly once in the primary diagnostic table. The largest observed publication
pause was 0.001510 seconds. A read-only refresh finished with no aggregation
errors: 5248 total runs, 3971 excluded and 1277 eligible. The subsequent snapshot
`reports_snapshot_20261012_0431.zip` contains 49 hash-verified artifacts, including
the assistance audit and runtime provenance. It is a partial snapshot (21/31
simulation cells complete); all 50-repetition zero-sum series remain in progress.

The assistance pool subsequently finished all 80 production tasks with exit 0;
its tmux session ended. Together with the pilot, all 81 published fits passed
`audit_zero_tail_assistance.py --require-complete`, with no pending, cancelled or
failed helpers. Sample, metadata and completion-marker hashes remain unchanged,
and all original-worker resume signals were sent. The largest publication pause
was 0.001510 seconds; the greatest accumulated pause in one original worker was
0.015136 seconds. Remaining repetitions continue in the original workers.

The RHS exact worker also finished all 50 repetitions with exit 0. Its saved
summaries cover 1,000,000 retained draws, all within the coefficient-scale
roundoff tolerance. Across these 50 fits, mean absolute coefficient-sum residual
is 1.1503e-15 and the maximum absolute residual is 1.5155e-14. The beta diagnostic
screen passes 48/50 fits; retained-sampling divergences total zero. This check
uses saved summaries and is recorded in `logs/exact_rhs_completion_audit.json`.

### Reproducible saved-output completion checks

`scripts/verify_revision_saved_outputs.py` checks the 31 requested cells against
their actual primary diagnostic rows and repetition IDs, verifies the formal
4 x (2000 warm-up + 5000 retained) metadata and saved draw counts, checks all
three coefficient estimates' train/test C-indices and coverage/FDP bounds, and
requires the recorded transition and selection diagnostics. It also cross-checks
data-identity sources and all 20 clinical outer folds / 60 reproduced scores.
Exact-constraint residual checks use saved summaries; this does not replace the
independent posterior-draw audit or establish convergence.

The check passed on all 1277 fits in the latest completed partial refresh.
`--require-complete` correctly rejected that partial report. Completion requires
31 cells with 50 distinct repetition IDs and 1500 distinct primary simulation
fits (850 RHS, 650 SSL; the RHS lambda=0.001 / p0=6 baseline is shared). Both
available-result and rejection checks are recorded in
`logs/saved_outputs_audit_checks.json`. This utility is included in the normal
read-only refresh, and its JSON report is included in subsequent snapshots.
To run a strict final check on the deployed reports:

```bash
/usr/bin/python3 analysis_tools_v1/verify_revision_saved_outputs.py \
  --revision-root /large_storage/zhangxf/BayesCox/revision_20261011 \
  --outdir /large_storage/zhangxf/BayesCox/revision_20261011/results/saved_outputs_audit \
  --require-complete
```

## Final zero-sum comparison: all 50 repetitions

These results use Gompertz baseline risk, n=400, p=150 and 12 nonzero
coefficients, matching the referenced paper section. C-index and MSE use
posterior-mean coefficients. Coverage is the average 95% equal-tail interval
coverage of the 12 true nonzero coefficients. Let S be the sum of coefficients
in one retained joint draw; the mean absolute sum column averages E(|S|) over
the 50 fits. All rows retain all 50 eligible fits.

| Method | Constraint | Test C-index | MSE | Signal coverage | Mean E(abs(S)) | Beta diagnostic screen |
|---|---|---:|---:|---:|---:|---:|
| Cox-RHS | none | 0.903187 | 0.003000 | 0.9550 | 0.804960 | 50/50 |
| Cox-RHS | soft_center | 0.903465 | 0.002712 | 0.9500 | 0.119174 | 50/50 |
| Cox-RHS | prox 1e-3 | 0.903447 | 0.002729 | 0.9517 | 0.295491 | 50/50 |
| Cox-RHS | prox 1e-6 | 0.903459 | 0.002689 | 0.9533 | 0.009758 | 0/50 |
| Cox-RHS | exact | 0.903446 | 0.002790 | 0.9400 | 1.1503e-15 | 48/50 |
| Prox-SSLASSO | none | 0.710378 | 0.220962 | 0.0000 | 0.528219 | 0/50 |
| Prox-SSLASSO | soft_center | 0.718410 | 0.221888 | 0.0000 | 0.153375 | 0/50 |
| Prox-SSLASSO | prox 1e-3 | 0.710527 | 0.221469 | 0.0000 | 0.423627 | 0/50 |
| Prox-SSLASSO | prox 1e-6 | 0.723112 | 0.221731 | 0.0000 | 0.009784 | 0/50 |
| Prox-SSLASSO | exact | 0.722501 | 0.221712 | 0.0000 | 1.3667e-16 | 0/50 |

The beta diagnostic screen requires every coefficient's rank-normalized Rhat
<=1.01 and minimum bulk/tail ESS >=400. No retained-sampling divergences were
recorded in these 500 fits. Zero divergences alone do not establish convergence:
RHS prox(1e-6) and every SSL zero-sum fit fail the screen. In particular, SSL's
zero signal coverage and poor mixing limit interpretation of its predictive and
selection results at this fixed budget. Empirical interval-based FDP is zero
in all ten rows; it does not by itself establish successful signal recovery.
The screened summaries are sensitivity results, separate from the primary table.

Both exact methods satisfy the coefficient-scale roundoff tolerance in every
saved zero-sum summary (1,000,000 retained draws per method). Maximum absolute
coefficient-sum residuals are 1.5155e-14 for RHS and 9.7318e-16 for SSL. Exact
constraints apply to joint draws. Separately computed marginal posterior medians
need not sum to zero, even when every joint draw does.

The full [diagnostic summary](results/revision_20261011/reports_final_all_simulations/diagnostic_aggregates_v2/diagnostics_by_setting.csv)
also reports posterior sum means, SDs, quantiles, RMS, maximum absolute residuals,
threshold probabilities, roundoff fractions and sums of coefficient means/medians;
each run retains chain-specific and pooled statistics. Columns ending in
`_mean` average the named within-run statistic across repetitions, and `_sd`
describes its variation between repetitions. They are not quantiles or SDs of
one pooled mixture of all repetitions. The
[performance summary](results/revision_20261011/reports_final_all_simulations/performance_aggregates/requested_metrics.csv)
includes train/test C-index, mean/median/MAP estimates, all/signal/null coverage,
FDP, confusion counts, between-repetition SDs and Monte Carlo standard errors.

Final reporting completed with all eleven prerequisite jobs at exit 0 and no
aggregation errors. The original dispatcher resumed, and its queued prox(1e-6)
invocation skipped all 50 completed fits. All 81 assisted fits also have explicit
`[SKIP]` records in their original workers, with unchanged sample, metadata and
completion-marker hashes. All 230 preexisting successful T1D inner checkpoints
remain unchanged. Frozen source manifests and the 34 reporting/operational
utility hashes match their deployed files. The formal model regression log
contains 1234 passing checks in 55 testsets. Runtime summaries retain the
shared-server and fresh-process compilation qualifications described above.

## S4 Brownian phylogeny extension (2026-10-12)

The user requested a new S4 setting and a new Cox-RHS-only CAR comparison.
This extension is separate from the completed 31-cell revision above.
Legacy `phylo` remains exp(-patristic_distance/4); it already has tree-related
dependence and must not be relabeled as Brownian motion.

S4 (`phylo_bm`) retains the planted balanced topology, positive unit edge lengths,
six neighboring signal tips, existing coefficient truth/means, n=200, p=50 and
Weibull survival/censoring. Its latent Gaussian covariance is shared root-to-MRCA
path length divided by tree height H=6, with fixed root state zero and unit
marginal variance. Softmax generates strictly positive compositions. CAR weights
remain exp(-d/2) for d<=4; the wrong graph uses the fixed seed-9017 leaf permutation.
CAR smooths log local shrinkage, not the abundance covariance directly.

The three arms are Independent, Correct CAR and Misspecified CAR, Cox-RHS only,
with p0=6 and proximal sum-zero lambda=0.001. Each arm uses 50 repetitions,
4 chains, 2000 warm-up and 5000 retained draws per chain. All arms share the same
training and independent n=1000 test cohorts within a repetition. The new DGP
requires 150 new fits; S1/S2/S3 results are reused without resampling.

The tree Newick, edges, leaf order, distance/covariance/graph matrices and SHA256
scientific fingerprints are exported. Missing/changed S4 fingerprints prevent
posterior reuse. S4 has its own aggregation profile and completion audit, leaving
the old experiment's completion matrix unchanged.

Validation: 63 phylogenetic/CAR checks passed locally and on the server; 309
scenario/C-index regression checks passed locally. Independent R ape::vcv.phylo
validation of the 50-tip Newick gave maximum absolute difference 5.551115e-17.
Three-arm end-to-end smoke and formal execution are tracked under
`/large_storage/zhangxf/BayesCox/s4_20261012/logs/`. Source v1 was a packaging
preflight missing priors/ (no fitted results); v2 includes the complete sources.
Formal source v3 has identical Julia/sampling/DGP files to tested v2, with only
reporting additions/correction (136 source files, SHA256 manifest).

The requested local deliverable is **summary results only**: all S1-S4 settings,
all 50 repetitions, mean/median/MAP metrics, interval coverage, empirical FDP,
diagnostic summaries, paired S4 contrasts, figures and a Chinese Markdown report.
Large posterior draws and intermediate datasets remain on the server.

### S4 completion and local S1–S4 delivery

All 18 disjoint formal partitions exited 0. Each S4 graph has 50 completed fits,
all 150 pass the beta convergence screen, and all have zero retained-sampling
divergences. Saved draw budgets, source hashes, tree/covariance fingerprints and
all 50 shared train/test cohort triples were verified. There are no excluded S4
repetitions and no aggregation errors.

| S4 graph | Test C-index, posterior mean | MSE | Signal coverage | Empirical FDP |
|---|---:|---:|---:|---:|
| Independent | 0.8762879693 | 0.0071186396 | 0.980000 | 0.000000 |
| Correct CAR | 0.8757656628 | 0.0079630754 | 0.976667 | 0.000000 |
| Misspecified CAR | 0.8726729905 | 0.0118115409 | 0.966667 | 0.005714 |

Correct-minus-independent test C-index is -0.0005223065, paired MCSE
0.0000691041, approximate 95% Monte Carlo interval [-0.000657751, -0.000386862].
MSE increases by 0.0008444358 (paired MCSE 0.0001489428). Thus the specified
Brownian DGP does not produce a predictive benefit from this CAR prior; the
misspecified graph degrades prediction more. This is a conclusion about the
fixed tree/support/prior configuration, not all possible phylogenetic models.

Independent Python replay verified all 450 S1–S4 fits: 2700 train/test C-indices,
MSE, all/signal/null coverage and empirical FDP match saved summaries, maximum
absolute difference 3.330669e-16. No posterior was resampled for this audit.
R ape::vcv.phylo independently matched the S4 covariance as recorded above.

The [complete Chinese results document](results/simulation_S1_S4_20261012/RESULTS.md)
and [local summary folder](results/simulation_S1_S4_20261012) contain all nine
50-repetition cells, 1350 mean/median/MAP performance rows, 54 aggregate rows,
450 diagnostic rows, paired effects and two PDF/PNG figures. S1–S3 inference
limitations remain visible (notably Prox-SSLASSO mixing); no poor-mixing fits are
silently removed from the primary results. The two figures were rendered and
visually inspected for labels, scales and uncertainty annotations.

The server summary bundle has 89 individually SHA256-verified files, archive hash
`08f2468187c2398ed37667a63dd7e630ff3d9dabe86d8eaf2648c6a91900d7d8`;
local copy: `results/s4_20261012/summary_bundle.zip`. The formal results and large
posteriors remain at `/large_storage/zhangxf/BayesCox/s4_20261012/results/S4`.
Current local reporting scripts add the complete S1–S4 narrative and independent
audit tools; the local runner also fails closed on S4 resume/recovery errors
instead of silently resampling. These postflight changes do not alter the
frozen code_v3 sampler or the completed scientific results.
