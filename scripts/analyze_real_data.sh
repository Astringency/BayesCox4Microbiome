#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/common.sh
source scripts/lib/realdata/datasets.sh
case "${1:-}" in
    --help) echo 'Usage: analyze_real_data.sh [t1d|standard|loo|cv-replay|convergence] [options]; default: complete T1D analysis and PH diagnostics'; exit 0;;
    t1d) shift; exec bash scripts/lib/realdata/t1d_batch.sh "$@";;
    convergence) shift; exec bash scripts/lib/realdata/convergence_grid.sh "$@";;
    loo|cv-replay) workflow="$1"; shift; run_julia scripts/lib/realdata/analyze.jl --workflow "$workflow" "$@"; exit;;
    standard) shift;;
    *) exec bash scripts/lib/realdata/t1d_batch.sh "$@";;
esac
constraint="${CONSTRAINT:-prox}"
OUT_ROOT="${OUT_ROOT:-results/realdata_${constraint}}"
SOFT_CONFIG_ROOT="${SOFT_CONFIG_ROOT:-results/realdata_soft_center}"
COMMON=(
    --priors "${PRIORS:-hs,rhs,new-rhs,cox-hs,cox-rhs,lasso,sslasso}" --sampler "${SAMPLER:-nuts}" --samples "${SAMPLES:-5000}"
    --warmup "${WARMUP:-4000}" --thin 1 --chains "${CHAINS:-4}"
    --multichains true --target-accept 0.8 --noncentered true
    --seed "${SEED:-1}" --progress true --nuts-backend explicit
    --max-depth 10 --hmc-n-leapfrog 100
    --prox-block-min-step-size 1e-8 --prox-block-max-step-size 1.0 --prox-block-adapt-every 1
    --prox-block-divergence-threshold 1000.0 --prox-block-metric unit_diag --hs-tau-scale 1.0
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
