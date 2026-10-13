#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/lib/simulation_progress.sh
simulation_setup
SIM_EXTRA=("$@")
REVISION_OUT="${REVISION_OUT:-./results/review_runs}"
# Separate the revised protocol from existing prox / long-chain results.
SCENARIO_OUT="$REVISION_OUT/scenarios/revised_20261013"
OUT_ROOT="${OUT_ROOT:-$SCENARIO_OUT}"
REPS="${REPS:-50}"; CHAINS="${CHAINS:-4}"; SAMPLES="${SAMPLES:-5000}"; WARMUP="${WARMUP:-5000}"
IFS=',' read -ra revision_priors <<< "${PRIORS:-cox-rhs,prox-sslasso}"
IFS=',' read -ra revision_scenarios <<< "${SCENARIOS:-S1,S2,S3}"
for method in "${revision_priors[@]}"; do
    case "$method" in
        cox-rhs) use_car=true; constraint=soft_center; l1_lambda=1e-6; sum_lambda=1e-3 ;;
        prox-sslasso) use_car=false; constraint=prox; l1_lambda=1e-6; sum_lambda=1e-3 ;;
        *) exit 2 ;;
    esac
    for scenario in "${revision_scenarios[@]}"; do
        simulation_job "${scenario}_${method}" "$SCENARIO_OUT/$scenario/$method" \
            --reps "$REPS" --n 200 --p 50 --nonzero 6 --baseline weibull --scenario "$scenario" \
            --weak-multiplier 0.35 --null-correlation 0.6 --depth-median 5000 --depth-log-sd 0.5 \
            --priors "$method" --sampler "${SAMPLER:-hmc}" --nuts-backend explicit --hmc-n-leapfrog "${HMC_N_LEAPFROG:-300}" --target-accept "${TARGET_ACCEPT:-0.8}" \
            --samples "$SAMPLES" --warmup "$WARMUP" --thin 1 --chains "$CHAINS" --multichains true \
            --noncentered true --seed 1 --use-car "$use_car" --sample-car-strength "$use_car" \
            --car-weight-quantile 0.99 --car-precision 1 --car-strength-prior-a 1 --car-strength-prior-b 1 \
            --car-distance bray --p0 truth --constraint "$constraint" --prox-sum0-lambda "$sum_lambda" \
            --prox-l1-lambda "$l1_lambda" --alpha-prior-kind invgamma --alpha-prior-shape 1 --alpha-prior-scale 50 \
            --sslasso-spike-scale 0.04 --sslasso-slab-scale 0.5 --sslasso-theta-a 1 --sslasso-theta-b 1 \
            --rhs-slab-df 4 --rhs-slab-s2 2 --rhs-tau-scale 1 --rhs-local-scale 1 \
            --pseudocount 0 --standardize false --diagnose false --progress true --resume true \
            --sample-thin 1 --first-rep-sample-thin 1 --save-wide-csv true --save-jld2 false
    done
done
simulation_finish
