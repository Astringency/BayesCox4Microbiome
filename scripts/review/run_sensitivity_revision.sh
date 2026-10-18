#!/usr/bin/env bash
# Usage: run_sensitivity_revision.sh lambda|l1|p0
# p0 is a Cox-RHS hyperparameter; it is not imposed on proximal LASSO priors.
# p0 uses soft_center; lambda and l1 sensitivity use prox.
set -euo pipefail

cd "$(dirname "$0")/../.."
suite="${1:?choose lambda, l1 or p0}"
shift

source scripts/lib/simulation_progress.sh
simulation_setup
SIM_EXTRA=("$@")
# /large_storage/zhangxf/BayesCox
REVISION_OUT="${REVISION_OUT:-./results/review_runs}"
OUT_ROOT="${OUT_ROOT:-$REVISION_OUT/$suite}"
REPS="${REPS:-50}"; CHAINS="${CHAINS:-4}"; SAMPLES="${SAMPLES:-5000}"; WARMUP="${WARMUP:-2000}"

constraint=prox
case "$suite" in
    lambda) methods="${PRIORS:-cox-rhs,prox-sslasso}"; values="${VALUES:-1e-2,1e-3,1e-4,1e-5}" ;;
    l1) methods=prox-sslasso; values="${VALUES:-1e-2,1e-3,1e-4,1e-5,1e-6}" ;;
    p0) methods=cox-rhs; values="${VALUES:-3,6,12}"; constraint=soft_center ;;
    *) echo 'suite must be lambda, l1 or p0' >&2; exit 2 ;;
esac

IFS=',' read -ra revision_priors <<< "$methods"
IFS=',' read -ra revision_values <<< "$values"

for method in "${revision_priors[@]}"; do
    case "$method" in
        cox-rhs) use_car=true; l1_lambda=1e-6 ;; # Unused by Cox-RHS.
        prox-sslasso) use_car=false; l1_lambda=1e-6 ;;
        *) echo "Unsupported method $method" >&2; exit 2 ;;
    esac
    for value in "${revision_values[@]}"; do
        p0=6; sum_lambda=1e-3
        if [[ "$suite" == lambda ]]; then sum_lambda="$value"
        elif [[ "$suite" == l1 ]]; then l1_lambda="$value"
        else p0="$value"; fi
        outdir="$REVISION_OUT/$suite/$method/$value"
        if [[ "$suite" == p0 ]]; then
            # Keep new soft_center fits separate from historical prox results.
            outdir="$REVISION_OUT/p0/soft_center/$method/$value"
        fi
        SIM_JOB_LOCK=''
        if [[ "$constraint" == prox && "$method" == cox-rhs && "$p0" == 6 && "$sum_lambda" == 1e-3 ]]; then
            # Serialize launches of the prox baseline; p0 now uses soft_center.
            outdir="$REVISION_OUT/lambda/cox-rhs/1e-3"
            SIM_JOB_LOCK="$REVISION_OUT/.rhs_baseline.lock"
        fi
        simulation_job "${suite}_${method}_${value}" "$outdir" \
            --reps "$REPS" --rep-ids "${REP_IDS:-}" --n 400 --p 50 --nonzero 6 --baseline weibull \
            --priors "$method" --sampler "${SAMPLER:-hmc}" --nuts-backend explicit --hmc-n-leapfrog "${HMC_N_LEAPFROG:-100}" \
            --samples "$SAMPLES" --warmup "$WARMUP" --thin 1 --chains "$CHAINS" --multichains true \
            --noncentered true --init "${INIT:-rand}" --seed 1 --use-car "$use_car" --sample-car-strength "$use_car" \
            --car-weight-quantile 0.99 --car-precision 1 --car-strength-prior-a 1 --car-strength-prior-b 1 \
            --car-distance bray --p0 "$p0" --constraint "$constraint" --prox-sum0-lambda "$sum_lambda" \
            --prox-l1-lambda "$l1_lambda" --alpha-prior-kind invgamma --alpha-prior-shape 1 --alpha-prior-scale 50 \
            --sslasso-spike-scale 0.04 --sslasso-slab-scale 0.5 --sslasso-theta-a 1 --sslasso-theta-b 1 \
            --rhs-slab-df 4 --rhs-slab-s2 2 --rhs-tau-scale 1 --rhs-local-scale 1 \
            --standardize false --diagnose false --progress true --resume true \
            --sample-thin 1 --first-rep-sample-thin 1 --save-wide-csv true --save-jld2 false
    done
done
simulation_finish
