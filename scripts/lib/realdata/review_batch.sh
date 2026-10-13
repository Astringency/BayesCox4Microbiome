#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source scripts/lib/common.sh
t1d_args=("$@")
# An explicit source selects one existing arm and preserves its specification.
for value in "${t1d_args[@]}"; do
    if [[ "$value" == --source-arm || "$value" == --source-arm=* ]]; then
        run_logged t1d-source "${LOG_ROOT:-results/realdata_review/logs}" run_julia scripts/analyze_real_data.jl --workflow t1d-review "${t1d_args[@]}"
        exit
    fi
done
OUT_ROOT="${OUT_ROOT:-results/realdata_review/$(date +%Y%m%d_%H%M%S)}"
IFS=',' read -ra t1d_priors <<< "${PRIORS:-cox-rhs,prox-sslasso}"
IFS=',' read -ra t1d_constraints <<< "${CONSTRAINTS:-prox,soft_center}"
IFS=',' read -ra t1d_adjustments <<< "${ADJUSTMENTS:-unadjusted,adjusted}"
t1d_arm() {
    local constraint="$1" prior="$2" adjustment="$3" adjusted=false
    [[ "$adjustment" != adjusted ]] || adjusted=true
    run_logged "${constraint}_${prior}_${adjustment}" "${LOG_ROOT:-$OUT_ROOT/logs}" run_julia scripts/analyze_real_data.jl --workflow t1d-review \
        --input data/T1D.csv --outdir "$OUT_ROOT" --constraint "$constraint" \
        --prior "$prior" --adjusted "$adjusted" --stage all "${t1d_args[@]}" || return
    # PH output is appropriate only after an all/final request completes.
    local stage=all i
    for ((i=0;i<${#t1d_args[@]};i++)); do
        case "${t1d_args[i]}" in --stage) stage="${t1d_args[i+1]}";; --stage=*) stage="${t1d_args[i]#*=}";; esac
    done
    if [[ "${RUN_PH:-true}" == true && ( "$stage" == all || "$stage" == final ) ]]; then
        local arm="$OUT_ROOT/$constraint/$adjustment/$prior"
        run_logged "${constraint}_${prior}_${adjustment}_ph" "${LOG_ROOT:-$OUT_ROOT/logs}" run_command "${RSCRIPT_BIN:-Rscript}" scripts/check_t1d_ph.R \
            "$arm/ph_design.csv" "$arm/ph_coefficients.csv" "$arm/ph_diagnostics"
    fi
}
for constraint in "${t1d_constraints[@]}"; do
for prior in "${t1d_priors[@]}"; do
for adjustment in "${t1d_adjustments[@]}"; do
    [[ "$adjustment" == adjusted || "$adjustment" == unadjusted ]] || { echo 'Invalid adjustment' >&2; exit 2; }
    start_job t1d_arm "$constraint" "$prior" "$adjustment"
done; done; done
wait_jobs
