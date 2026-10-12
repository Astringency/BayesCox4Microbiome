#!/usr/bin/env bash
# Usage: run_sensitivity_revision.sh lambda|p0
# p0 is a Cox-RHS hyperparameter; it is not imposed on proximal LASSO priors.
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 GKSwstype=100
suite="${1:?choose lambda or p0}"
JULIA_BIN="${JULIA_BIN:-julia}"
REVISION_OUT="${REVISION_OUT:-./results/revision_20261011}"
HISTORICAL_ROOT="${HISTORICAL_ROOT:-/large_storage/zhangxf/BayesCox/results}"
REPS="${REPS:-50}"; CHAINS="${CHAINS:-4}"; SAMPLES="${SAMPLES:-5000}"; WARMUP="${WARMUP:-2000}"
case "$suite" in
    lambda) methods="${PRIORS:-cox-rhs,prox-sslasso}"; values="${VALUES:-1e-2,1e-3,1e-4,1e-5}" ;;
    p0) methods=cox-rhs; values="${VALUES:-3,6,12}" ;;
    *) echo 'suite must be lambda or p0' >&2; exit 2 ;;
esac
IFS=',' read -ra revision_priors <<< "$methods"
IFS=',' read -ra revision_values <<< "$values"
for method in "${revision_priors[@]}"; do
    case "$method" in
        cox-rhs) use_car=true; l1_lambda=0.001 ;;
        prox-sslasso) use_car=false; l1_lambda=0.0001 ;;
        *) echo "Unsupported method $method" >&2; exit 2 ;;
    esac
    for value in "${revision_values[@]}"; do
        p0=6; sum_lambda=1e-3
        if [[ "$suite" == lambda ]]; then sum_lambda="$value"; else p0="$value"; fi
        outdir="$REVISION_OUT/$suite/$method/$value"
        baseline_lock_fd=''
        if [[ "$method" == cox-rhs && "$p0" == 6 && "$sum_lambda" == 1e-3 ]]; then
            # Both ablations share one baseline. Serialize independent launches
            # so a second launcher resumes completed fits instead of duplicating them.
            outdir="$REVISION_OUT/lambda/cox-rhs/1e-3"
            mkdir -p "$REVISION_OUT"
            exec {baseline_lock_fd}> "$REVISION_OUT/.rhs_baseline.lock"
            flock "$baseline_lock_fd"
        fi
        # The same verified p0=6/lambda=1e-3 baseline serves both ablations.
        if [[ "${FORCE_NEW_FITS:-false}" != true && "$method" == cox-rhs && "$p0" == 6 && "$sum_lambda" == 1e-3 ]] && \
            python3 scripts/check_revision_reuse.py --historical-root "$HISTORICAL_ROOT" \
                --fixed-baseline-manifest "$REVISION_OUT/resolved_fixed_rhs_baseline.json" \
                --manifest "$outdir/reuse_manifest.json" --reps "$REPS" \
                --samples "$SAMPLES" --warmup "$WARMUP" --chains "$CHAINS"; then
            echo "Reusing audited Cox-RHS baseline: $outdir/reuse_manifest.json"
            exec {baseline_lock_fd}>&-
            continue
        fi
        "$JULIA_BIN" -t "$CHAINS" --startup-file=no --project=. scripts/run_simulation.jl \
            --outdir "$outdir" --reps "$REPS" --rep-ids "${REP_IDS:-}" --n 400 --p 50 --nonzero 6 --baseline weibull \
            --priors "$method" --sampler hmc --nuts-backend explicit --hmc-n-leapfrog 100 \
            --samples "$SAMPLES" --warmup "$WARMUP" --thin 1 --chains "$CHAINS" --multichains true \
            --noncentered true --seed 1 --use-car "$use_car" --sample-car-strength "$use_car" \
            --car-weight-quantile 0.99 --car-precision 1 --car-strength-prior-a 1 --car-strength-prior-b 1 \
            --car-distance bray --p0 "$p0" --constraint prox --prox-sum0-lambda "$sum_lambda" \
            --prox-l1-lambda "$l1_lambda" --alpha-prior-kind invgamma --alpha-prior-shape 1 --alpha-prior-scale 50 \
            --sslasso-spike-scale 0.04 --sslasso-slab-scale 0.5 --sslasso-theta-a 1 --sslasso-theta-b 1 \
            --rhs-slab-df 4 --rhs-slab-s2 2 --rhs-tau-scale 1 --rhs-local-scale 1 \
            --pseudocount 0 --standardize false --diagnose false --progress false --resume true \
            --sample-thin 1 --first-rep-sample-thin 1 --save-wide-csv true --save-jld2 false
        if [[ -n "$baseline_lock_fd" ]]; then exec {baseline_lock_fd}>&-; fi
    done
done
