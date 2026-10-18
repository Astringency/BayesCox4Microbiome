#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source scripts/lib/common.sh
source scripts/lib/realdata/datasets.sh
OUT_ROOT="${OUT_ROOT:-results/realdata}"
TUNING_ROOT="${TUNING_ROOT:-results/realdata_tuning}"; FALLBACK_TUNING_ROOT="${FALLBACK_TUNING_ROOT:-results/realdata}"
IFS=',' read -ra constraints <<< "${CONSTRAINTS:-soft_center,prox}"
IFS=',' read -ra datasets <<< "${RUN_DATASETS:-$ALL_REALDATA_DATASETS}"
[[ "${RUN_DATASETS:-}" != all ]] || IFS=',' read -ra datasets <<< "$ALL_REALDATA_DATASETS"
COMMON=(
    --priors "${PRIORS:-hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso,prox-lasso,prox-sslasso}" --sampler nuts --nuts-backend explicit
    --samples "${SAMPLES:-800}" --warmup "${WARMUP:-800}" --thin 1 --chains "${CHAINS:-4}" --multichains false
    --target-accept "${TARGET_ACCEPT:-0.9}" --max-depth "${MAX_DEPTH:-10}" --seed "${SEED:-20260618}" --progress true
    --standardize false --tie-method efron
    --diagnose false --tune_priorsparams false --tune_algparams false --use-car false
    --hs-tau-scale 1.0 --hs-local-scale 1.0 --rhs-slab-df 4.0 --rhs-slab-s2 2.0
    --rhs-tau-scale 1.0 --rhs-local-scale 1.0 --rhs-pseudo-sigma-df 3.0 --rhs-pseudo-sigma-scale 10.0
    --lasso-lambda2-shape 1.0 --lasso-lambda2-rate 10.0 --sslasso-spike-scale 1.0 --sslasso-slab-scale 10.0
    --sslasso-theta-a 1.0 --sslasso-theta-b 1.0 --prox-l1-lambda 1e-3 --alpha-prior-shape 1.0
    --prox-root-tol 1e-10 --prox-root-maxiter 100
)
for constraint in "${constraints[@]}"; do
    constraint="${constraint//[[:space:]]/}"; [[ -n "$constraint" ]] || continue
    noncentered="${SOFT_CENTER_NONCENTERED:-true}"; sum0="${SOFT_CENTER_PROX_SUM0_LAMBDA:-1e-6}"
    if [[ "$constraint" == prox ]]; then noncentered="${PROX_NONCENTERED:-false}"; sum0="${PROX_SUM0_LAMBDA:-1e-6}"; fi
    for dataset in "${datasets[@]}"; do
        [[ -n "${dataset//[[:space:]]/}" ]] || continue
        realdata_dataset "$dataset"
        configs=none
        for base in "$TUNING_ROOT/$constraint" "$FALLBACK_TUNING_ROOT/$constraint" "$TUNING_ROOT/soft_center" "$FALLBACK_TUNING_ROOT/soft_center"; do
            candidate="$base/$DATA_NAME/tuning/best_prior_configs.json"
            if [[ -f "$candidate" ]]; then configs="$candidate"; break; fi
        done
        run_logged "${constraint}_${DATA_NAME}" "${LOG_ROOT:-$OUT_ROOT/logs}" run_julia scripts/analyze_real_data.jl \
            --input "$DATA_INPUT" --outdir "$OUT_ROOT/$constraint/$DATA_NAME" --time-col "$DATA_TIME" --status-col "$DATA_STATUS" \
            --comp-cols none --exclude-cols "$DATA_EXCLUDE" --constraint "$constraint" --prox-sum0-lambda "$sum0" \
            --noncentered "$noncentered" --p0 auto --alpha-prior-scale "$DATA_P" --prior-configs "$configs" "${COMMON[@]}" "$@"
    done
done
