# Cox LOO evaluation modes

The public analysis helper is `evaluate_cox_loo`, loaded by
`include("scripts/analysis_common.jl")`. Select `mode=:fixed_risk` or
`mode=:subject_refit`. The real-data CLI also accepts
`--loo-mode fixed_risk|subject_refit` (default `fixed_risk`) and
`--loo-parallel true|false`.

## Predictive targets

`fixed_risk` preserves the historical calculation: PSIS removes one observed
event's partial-likelihood factor while holding every risk set fixed. Censored
subjects contribute to denominators but are not evaluation units. Its WAIC and
harmonic-mean NLPML use the same event factors. The historical column `LPML`
has the sign of **NLPML**; the T1D runner also supplies an explicit `NLPML` alias.
The harmonic mean is now evaluated with log-sum-exp to avoid overflow.
The historical Efron event-factor allocation is unchanged: fixed-risk pointwise
criteria can depend on the row ordering of tied events.

`subject_refit` removes each subject, including censored subjects, from the
training arrays. The usual fitting pipeline rebuilds training risk sets,
preprocessing and training-dependent prior scales, then runs independent MCMC.
The held-out subject is added back **only for scoring**, transformed using
training zero replacement and column scaling. For training posterior draws
`beta[-i,s]`, define

```math
a_{is}=\log L_{PL}(D;\beta_{-i,s})-\log L_{PL}(D_{-i};\beta_{-i,s}),
\qquad c_i=\log\left(\frac{1}{S}\sum_{s=1}^S \exp(a_{is})\right).
```

Both likelihoods use the same transformation fitted on `D_-i`, and each has its
own risk sets. The subtraction accounts for every denominator affected by a
deletion, even if the subject is censored. For Efron ties, total group likelihoods
are used, so the score does not depend on assigning sequential tie terms to
individual subjects. The implementation averages likelihood ratios over the
posterior before taking logs; it does not average log ratios or substitute a
posterior mean coefficient.

This is a **subject-deletion cross-validated partial-likelihood ratio score**,
following the subtraction idea used for Cox cross-validation (see the
[glmnet Cox vignette](https://glmnet.stanford.edu/articles/Coxnet.html)). Its
posterior integration is explicitly defined above. It is not a normalized
predictive density of an independent subject's survival time. A full survival
likelihood, including a baseline hazard model, would be needed for that target.

The refit summary reports:

| Column | Meaning |
|---|---|
| `LOO_LOG_SCORE` | Sum of `c_i`; higher is better |
| `LOOIC` | `-2 * sum(c_i)`, on the declared partial-likelihood ratio scale |
| `NLPML` | `-sum(c_i)`, the negative integrated partial score |
| `ELPD_LOO` | Missing for refits, to avoid calling the ratio a predictive density |
| `WAIC_pw1`, `WAIC_pw2` | Missing for refits; event-factor WAIC is in the separate reference file |
| `ParetoK_max`, `ParetoK_bad` | Missing for refits, which do not use importance sampling |
| `complete` | True only when all subjects have successful refits |

Do not compare the magnitudes of `fixed_risk` and `subject_refit` scores: their
evaluation units and targets differ. A successful fit is not a convergence
guarantee. Inspect per-fold `beta_rhat_max` and `beta_ess_bulk_min`; unavailable
diagnostics are missing. No automatic convergence retries or nested tuning are
performed. For incomplete or failed LOO runs, aggregate refit scores are missing.

## T1D rerun

The supplied script reads existing metadata and uses the same data columns,
prior settings, sampler, chain counts and warmup/sample counts. Defaults are
**Cox-RHS with `soft_center`** and **Prox-SSLASSO with `prox`**. All 173 T1D
subjects (118 events and 55 censored) are held out in turn for each method.
Historical outputs are not overwritten.

PowerShell, from the repository root:

```powershell
.\scripts\run_t1d_loo.ps1
```

This pins Julia 1.11.7, runs four folds in parallel and outputs both evaluation
modes. Equivalent Julia command (also usable on Linux):

```bash
julia +1.11.7 --startup-file=no -t 4 --project=. scripts/rerun_t1d_loo.jl --loo-mode both --parallel true --outdir results/loo/T1D
```

Use only the original fixed-risk metrics:

```bash
julia +1.11.7 --startup-file=no --project=. scripts/rerun_t1d_loo.jl --loo-mode fixed_risk --outdir results/loo/T1D_fixed
```

Use `prox` for both models:

```powershell
.\scripts\run_t1d_loo.ps1 -CoxRhsConstraint prox -OutDir results/loo/T1D_both_prox
```

Run a small, non-production pilot (aggregate refit scores remain missing):

```bash
julia +1.11.7 --startup-file=no -t 2 --project=. scripts/rerun_t1d_loo.jl --loo-mode both --parallel true --samples 32 --warmup 32 --chains 2 --folds 1,2 --outdir results/check/T1D_loo_pilot
```

The default `--p0-policy training` repeats automatic `p0` screening using only
training outcomes when source metadata says `p0_requested=auto`. Explicit `p0`
values stay fixed. `--p0-policy fixed` instead reuses the saved full-data `p0`;
this evaluates a fixed, previously selected configuration, not the entire
selection procedure. Previously tuned hyperparameters and selected input columns
are reused in both cases: these runs do not provide nested cross-validation of
the original tuning/feature selection. Cox-specific prior scales that depend on
the training sample/events are recomputed during each fit.

`--samples`, `--warmup`, `--chains` and `--seed` override saved settings.
`--methods cox-rhs` or `--methods prox-sslasso` runs one method. `--source-root`
selects a different historical real-data root. Use `--help` for all options.

Outputs under each `constraint_method/` directory:

- `settings.json`: source metadata and effective fitting settings.
- `fixed_risk_metrics.csv`: full-fit historical metrics, including WAIC/NLPML/PSIS-LOO.
- `full_fit_posterior_summary.csv`: full-fit coefficient summary.
- `fixed_risk_summary.csv`: selected fixed-risk LOO summary, if requested.
- `subject_refit_summary.csv`: aggregate refit score, if complete.
- `subject_refit_pointwise.csv`: subject status, score, training size, seed,
  `p0`, elapsed time and coefficient MCMC diagnostics.
- `folds/manifest.json` and `folds/fold_XXXXX.json`: resumable refits.

The top-level `loo_summary.csv` contains the methods/modes selected in the
current invocation. Each model's own outputs and fold files remain independent.
Restart the same command to skip successful folds and retry failed ones.
Changing data, fit settings, Julia version or Julia source invalidates saved
folds: choose a new output directory. Dependency manifest changes also invalidate
the cache. A pilot with shorter chains cannot be
resumed as a production run. A subset with production settings can be resumed
with `--folds all`. Do not run two processes against the same model checkpoint
directory at once. Parallelism is across folds; chain-level parallelism is
disabled to avoid nested oversubscription, and the T1D runner uses one BLAS thread.

## Julia interface

```julia
include("scripts/analysis_common.jl")
evaluation = evaluate_cox_loo(X, time, event;
    mode = :subject_refit,
    fit_kwargs = (prior_name="cox-rhs", nuts_backend=:explicit,
                  constraint=:soft_center, realdata=true, p0=nothing,
                  n_samples=800, warmup=800, chains=4, seed=20260618),
    checkpoint_dir = "results/loo/custom_cox_rhs",
    parallel = false)
```

At least two observed events are required so every training fold contains an
event. Subject refits reject `approx=true`. Existing `fit_and_evaluate` defaults
are unchanged; the new `compute_metrics=false` option skips expensive reference
metrics and summaries for the inner training fits.
The general real-data command keeps `model_metrics.csv` explicitly labeled
`fixed_risk` and writes the selected LOO summary to `loo_<prior>_<mode>.csv`;
it does not silently replace historical ELPD/WAIC columns with a different target.

Tests:

```bash
julia +1.11.7 --startup-file=no --project=. test/loo_tests.jl
```
