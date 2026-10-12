#!/usr/bin/env bash
# New exact fits only by default. Existing soft_center/prox/none fits are
# reevaluated with run_revision_diagnostics.sh and audited before primary reuse.
# To fill a missing or kernel-invalid cell, set PRIORS and CONSTRAINTS and use
# a separate output root. Preserve historical files.
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 GKSwstype=100
JULIA_BIN="${JULIA_BIN:-julia}"
REVISION_OUT="${REVISION_OUT:-./results/revision_20261011}"
REPS="${REPS:-50}"
CHAINS="${CHAINS:-4}"
SAMPLES="${SAMPLES:-5000}"
WARMUP="${WARMUP:-2000}"
N="${N:-400}"
P="${P:-150}"
NONZERO="${NONZERO:-12}"
IFS=',' read -ra revision_priors <<< "${PRIORS:-cox-rhs,prox-sslasso}"
IFS=',' read -ra revision_constraints <<< "${CONSTRAINTS:-exact}"
for method in "${revision_priors[@]}"; do
    case "$method" in
        cox-rhs) use_car=true; l1_lambda=0.001 ;;
        prox-sslasso) use_car=false; l1_lambda=0.0001 ;;
        *) echo "Unsupported method: $method" >&2; exit 2 ;;
    esac
    for constraint in "${revision_constraints[@]}"; do
        case "$constraint" in exact|none|soft_center|prox) ;; *) exit 2 ;; esac
        label="$constraint"
        [[ "$constraint" != prox ]] || label="prox${SUM0_LAMBDA:-1e-3}"
        "$JULIA_BIN" -t "$CHAINS" --startup-file=no --project=. scripts/run_simulation.jl \
            --outdir "$REVISION_OUT/zero_sum/$label/$method" \
            --reps "$REPS" --n "$N" --p "$P" --nonzero "$NONZERO" --baseline gompertz \
            --priors "$method" --sampler hmc --nuts-backend explicit --hmc-n-leapfrog 100 \
            --samples "$SAMPLES" --warmup "$WARMUP" --thin 1 --chains "$CHAINS" --multichains true \
            --noncentered true --seed 1 --use-car "$use_car" --sample-car-strength "$use_car" \
            --car-weight-quantile 0.99 --car-precision 1 --car-strength-prior-a 1 --car-strength-prior-b 1 \
            --car-distance bray --p0 truth --constraint "$constraint" --prox-sum0-lambda "${SUM0_LAMBDA:-1e-3}" \
            --prox-l1-lambda "$l1_lambda" --alpha-prior-kind invgamma --alpha-prior-shape 1 --alpha-prior-scale "$P" \
            --sslasso-spike-scale 0.04 --sslasso-slab-scale 0.5 --sslasso-theta-a 1 --sslasso-theta-b 1 \
            --rhs-slab-df 4 --rhs-slab-s2 2 --rhs-tau-scale 1 --rhs-local-scale 1 \
            --pseudocount 0 --standardize false --diagnose false --progress false --resume true \
            --sample-thin 1 --first-rep-sample-thin 1 --save-wide-csv true --save-jld2 false
    done
done
