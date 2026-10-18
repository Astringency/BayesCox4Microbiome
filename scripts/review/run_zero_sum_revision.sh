#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/../.."
source scripts/lib/simulation_progress.sh
simulation_setup

SIM_EXTRA=("$@")
REVISION_OUT="${REVISION_OUT:-./results/review_runs}"
OUT_ROOT="${OUT_ROOT:-$REVISION_OUT/zero_sum}"
REPS="${REPS:-50}"
CHAINS="${CHAINS:-4}"
SAMPLES="${SAMPLES:-5000}"
WARMUP="${WARMUP:-2000}"
N="${N:-400}"
P="${P:-150}"
NONZERO="${NONZERO:-12}"

# cox-rhs,prox-sslasso
IFS=',' read -ra revision_priors <<< "${PRIORS:-cox-rhs,prox-sslasso}"
# exact,none,soft_center,prox
IFS=',' read -ra revision_constraints <<< "${CONSTRAINTS:-exact,none,soft_center,prox}"

for method in "${revision_priors[@]}"; do
    case "$method" in
        cox-rhs) use_car=true; l1_lambda=1e-6 ;; # Unused by Cox-RHS.
        prox-sslasso) use_car=false; l1_lambda=1e-6 ;;
        *) echo "Unsupported method: $method" >&2; exit 2 ;;
    esac
    for constraint in "${revision_constraints[@]}"; do
        case "$constraint" in exact|none|soft_center|prox) ;; *) exit 2 ;; esac
        values=1e-3
        [[ "$constraint" != prox ]] || values="${SUM0_LAMBDAS:-1e-3,1e-6}"
        IFS=',' read -ra sum_lambdas <<< "$values"
        for sum_lambda in "${sum_lambdas[@]}"; do
        label="$constraint"
        [[ "$constraint" != prox ]] || label="prox${sum_lambda}"
        simulation_job "${label}_${method}" "$REVISION_OUT/zero_sum/$label/$method" \
            --reps "$REPS" --rep-ids "${REP_IDS:-}" --n "$N" --p "$P" --nonzero "$NONZERO" --baseline gompertz \
            --priors "$method" --sampler "${SAMPLER:-hmc}" --nuts-backend explicit --hmc-n-leapfrog "${HMC_N_LEAPFROG:-100}" \
            --samples "$SAMPLES" --warmup "$WARMUP" --thin 1 --chains "$CHAINS" --multichains true \
            --noncentered true --seed 1 --use-car "$use_car" --sample-car-strength "$use_car" \
            --car-weight-quantile 0.99 --car-precision 1 --car-strength-prior-a 1 --car-strength-prior-b 1 \
            --car-distance bray --p0 truth --constraint "$constraint" --prox-sum0-lambda "${sum_lambda}" \
            --prox-l1-lambda "$l1_lambda" --alpha-prior-kind invgamma --alpha-prior-shape 1 --alpha-prior-scale "$P" \
            --sslasso-spike-scale 0.04 --sslasso-slab-scale 0.5 --sslasso-theta-a 1 --sslasso-theta-b 1 \
            --rhs-slab-df 4 --rhs-slab-s2 2 --rhs-tau-scale 1 --rhs-local-scale 1 \
            --standardize false --diagnose false --progress true --resume true \
            --sample-thin 1 --first-rep-sample-thin 1 --save-wide-csv true --save-jld2 false
        done
    done
done
simulation_finish
