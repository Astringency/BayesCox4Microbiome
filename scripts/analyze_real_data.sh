#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/common.sh
source scripts/lib/realdata/datasets.sh

# T1D is the default workflow. Examples:
#   scripts/analyze_real_data.sh t1d
#   scripts/analyze_real_data.sh t1d --outdir results/review/T1D_prox_sum0_1e3_lambdaell_1e6 \
#       --constraint prox --prior prox-sslasso --prox-sum0-lambda 1e-3
# t1d_batch.sh is an internal driver; `standard` remains for other real datasets.

# Sampling settings aligned with the frozen T1D review results.
SAMPLES="${SAMPLES:-5000}"
WARMUP="${WARMUP:-4000}"
CHAINS="${CHAINS:-4}"
SEED="${SEED:-1}"
TARGET_ACCEPT="${TARGET_ACCEPT:-0.95}"
HMC_N_LEAPFROG="${HMC_N_LEAPFROG:-300}"

# The T1D workflow hardcodes sampler=hmc, multichains=true and noncentered=true;
# only these sampler controls are accepted on its command line.
T1D_SAMPLER=(
    --samples "$SAMPLES" --warmup "$WARMUP" --chains "$CHAINS" --seed "$SEED"
    --target-accept "$TARGET_ACCEPT" --leapfrog "$HMC_N_LEAPFROG"
)
SAMPLER_COMMON=(
    --sampler "${SAMPLER:-hmc}" --samples "$SAMPLES"
    --warmup "$WARMUP" --thin 1 --chains "$CHAINS"
    --multichains true --target-accept "$TARGET_ACCEPT" --noncentered true
    --seed "$SEED" --progress true --nuts-backend explicit
    --max-depth 10 --hmc-n-leapfrog "$HMC_N_LEAPFROG"
    --prox-block-min-step-size 1e-8 --prox-block-max-step-size 1.0
    --prox-block-adapt-every "${PROX_BLOCK_ADAPT_EVERY:-50}"
    --prox-block-divergence-threshold 1000.0 --prox-block-metric unit_diag
)
case "${1:-}" in
    --help) echo 'Usage: analyze_real_data.sh [t1d|standard|loo|cv-replay|convergence] [options]; default: complete T1D analysis and PH diagnostics'; exit 0;;
    t1d) shift; exec bash scripts/lib/realdata/t1d_batch.sh "${T1D_SAMPLER[@]}" "$@";;
    convergence) shift; exec bash scripts/lib/realdata/convergence_grid.sh "$@";;
    loo|cv-replay) workflow="$1"; shift; run_julia scripts/lib/realdata/analyze.jl --workflow "$workflow" "$@"; exit;;
    standard) shift;;
    *) exec bash scripts/lib/realdata/t1d_batch.sh "${T1D_SAMPLER[@]}" "$@";;
esac
constraint="${CONSTRAINT:-prox}"
OUT_ROOT="${OUT_ROOT:-results/realdata_${constraint}}"
SOFT_CONFIG_ROOT="${SOFT_CONFIG_ROOT:-results/realdata_soft_center}"
COMMON=(
    --priors "${PRIORS:-hs,rhs,new-rhs,cox-hs,cox-rhs,lasso,sslasso}"
    "${SAMPLER_COMMON[@]}"
    --hs-tau-scale 1.0
    --hs-local-scale 1.0 --rhs-slab-df 4.0 --rhs-slab-s2 2.0
    --rhs-tau-scale 1.0 --rhs-local-scale 1.0 --rhs-pseudo-sigma-df 3.0
    --rhs-pseudo-sigma-scale 10.0 --lasso-lambda2-shape 1.0 --lasso-lambda2-rate 0.1
    --sslasso-spike-scale 1.0 --sslasso-slab-scale 10.0 --sslasso-theta-a 1.0
    --sslasso-theta-b 1.0 --prox-l1-lambda 1e-3 --alpha-prior-shape 1.0
    --alpha-prior-scale 1.0 --prox-root-tol 1e-10 --prox-root-maxiter 100
    --use-car false --car-distance bray --car-weight-quantile 0.99
    --car-precision 1.0 --sample-car-strength true --car-strength-prior-a 1.0
    --car-strength-prior-b 1.0 --car-degree-scale mean_active --car-kappa-init 0.5
    --p0 auto --p0-alpha 0.05 --constraint "$constraint"
    --prox-sum0-lambda 1e-6 --standardize false
    --tie-method efron
    --diagnose true --diagnose-format pdf --diagnose-top-k-beta 10
    --diagnose-max-lag 50 --tune_priorsparams false --tune_algparams false
    --cv-folds 5 --tune-seed 42
 )
IFS=',' read -ra datasets <<< "${RUN_DATASETS:-T1D}"
[[ "${RUN_DATASETS:-}" != all ]] || IFS=',' read -ra datasets <<< "$ALL_REALDATA_DATASETS"
for dataset in "${datasets[@]}"; do
    realdata_dataset "$dataset"
    run_logged "${constraint}_${DATA_NAME}" "${LOG_ROOT:-$OUT_ROOT/logs}" run_julia scripts/lib/realdata/analyze.jl --workflow standard \
        --input "$DATA_INPUT" --outdir "$OUT_ROOT/$DATA_NAME" --time-col "$DATA_TIME" --status-col "$DATA_STATUS" \
        --comp-cols none --exclude-cols "$DATA_EXCLUDE" \
        --prior-configs "$SOFT_CONFIG_ROOT/$DATA_CONFIG/tuning/best_prior_configs.json" "${COMMON[@]}" "$@"
done
