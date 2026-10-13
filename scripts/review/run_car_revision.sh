#!/usr/bin/env bash
# Unified S4 Brownian phylogeny. phylo and phylo_bm are aliases for S4.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/lib/simulation_progress.sh
simulation_setup
SIM_EXTRA=("$@")
REVISION_OUT="${REVISION_OUT:-./results/review_runs}"
REPS="${REPS:-50}"; CHAINS="${CHAINS:-4}"; SAMPLES="${SAMPLES:-5000}"; WARMUP="${WARMUP:-2000}"
METHOD="${METHOD:-cox-rhs}"
SCENARIO="${SCENARIO:-S4}"
case "$SCENARIO" in
    S4|s4|phylo|phylo_bm)
        [[ "$METHOD" == cox-rhs ]] || { echo 'S4 CAR comparison is Cox-RHS only' >&2; exit 2; }
        SCENARIO=S4
        car_out="${CAR_OUT:-$REVISION_OUT/S4/car_v2}" ;;
    *) echo "unsupported CAR scenario: $SCENARIO" >&2; exit 2 ;;
esac
OUT_ROOT="${OUT_ROOT:-$car_out}"
rep_args=(); [[ -z "${REP_IDS:-}" ]] || rep_args=(--rep-ids "$REP_IDS")
graph_values="${GRAPHS:-independent,correct,misspecified}"
IFS=',' read -ra revision_graphs <<< "$graph_values"
for graph in "${revision_graphs[@]}"; do
    use_car=true; [[ "$graph" != independent ]] || use_car=false
    sum_lambda=1e-3; l1_lambda=0.001
    simulation_job "${METHOD}_${graph}" "$car_out/$METHOD/$graph" \
        --reps "$REPS" "${rep_args[@]}" --n 200 --p 50 --nonzero 6 --baseline weibull --scenario "$SCENARIO" --car-graph "$graph" \
        --priors "$METHOD" --sampler "${SAMPLER:-hmc}" --nuts-backend explicit --hmc-n-leapfrog "${HMC_N_LEAPFROG:-100}" \
        --samples "$SAMPLES" --warmup "$WARMUP" --thin 1 --chains "$CHAINS" --multichains true \
        --noncentered true --seed 1 --use-car "$use_car" --sample-car-strength "$use_car" \
        --car-precision 1 --car-strength-prior-a 1 --car-strength-prior-b 1 \
        --p0 truth --constraint prox --prox-sum0-lambda "$sum_lambda" --prox-l1-lambda "$l1_lambda" \
        --alpha-prior-kind invgamma --alpha-prior-shape 1 --alpha-prior-scale 50 \
        --sslasso-spike-scale 0.04 --sslasso-slab-scale 0.5 --sslasso-theta-a 1 --sslasso-theta-b 1 \
        --rhs-slab-df 4 --rhs-slab-s2 2 --rhs-tau-scale 1 --rhs-local-scale 1 \
        --pseudocount 0 --standardize false --diagnose false --progress true --resume true \
        --sample-thin 1 --first-rep-sample-thin 1 --save-wide-csv true --save-jld2 false
done
simulation_finish
