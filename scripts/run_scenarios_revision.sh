#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 GKSwstype=100
JULIA_BIN="${JULIA_BIN:-julia}"
REVISION_OUT="${REVISION_OUT:-./results/revision_20261011}"
REPS="${REPS:-50}"; CHAINS="${CHAINS:-4}"; SAMPLES="${SAMPLES:-5000}"; WARMUP="${WARMUP:-2000}"
IFS=',' read -ra revision_priors <<< "${PRIORS:-cox-rhs,prox-sslasso}"
IFS=',' read -ra revision_scenarios <<< "${SCENARIOS:-S1,S2,S3}"
for method in "${revision_priors[@]}"; do
    case "$method" in
        cox-rhs) use_car=true; l1_lambda=0.001; sum_lambda=1e-3 ;;
        prox-sslasso) use_car=false; l1_lambda=0.0001; sum_lambda=1e-6 ;;
        *) exit 2 ;;
    esac
    for scenario in "${revision_scenarios[@]}"; do
        "$JULIA_BIN" -t "$CHAINS" --startup-file=no --project=. scripts/run_simulation.jl \
            --outdir "$REVISION_OUT/scenarios/$scenario/$method" \
            --reps "$REPS" --n 200 --p 50 --nonzero 6 --baseline weibull --scenario "$scenario" \
            --weak-multiplier 0.35 --null-correlation 0.6 --depth-median 5000 --depth-log-sd 0.5 \
            --priors "$method" --sampler hmc --nuts-backend explicit --hmc-n-leapfrog 100 \
            --samples "$SAMPLES" --warmup "$WARMUP" --thin 1 --chains "$CHAINS" --multichains true \
            --noncentered true --seed 1 --use-car "$use_car" --sample-car-strength "$use_car" \
            --car-weight-quantile 0.99 --car-precision 1 --car-strength-prior-a 1 --car-strength-prior-b 1 \
            --car-distance bray --p0 truth --constraint prox --prox-sum0-lambda "$sum_lambda" \
            --prox-l1-lambda "$l1_lambda" --alpha-prior-kind invgamma --alpha-prior-shape 1 --alpha-prior-scale 50 \
            --sslasso-spike-scale 0.04 --sslasso-slab-scale 0.5 --sslasso-theta-a 1 --sslasso-theta-b 1 \
            --rhs-slab-df 4 --rhs-slab-s2 2 --rhs-tau-scale 1 --rhs-local-scale 1 \
            --pseudocount 0 --standardize false --diagnose false --progress false --resume true \
            --sample-thin 1 --first-rep-sample-thin 1 --save-wide-csv true --save-jld2 false
    done
done
