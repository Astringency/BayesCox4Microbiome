#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source scripts/lib/common.sh
t1d_args=("$@")
# An explicit source selects one existing arm and preserves its specification.
for value in "${t1d_args[@]}"; do
    if [[ "$value" == --source-arm || "$value" == --source-arm=* ]]; then
        run_logged t1d-source "${LOG_ROOT:-results/realdata/T1D/logs}" run_julia scripts/lib/realdata/analyze.jl --workflow t1d "${t1d_args[@]}"
        exit
    fi
done
OUT_ROOT="${OUT_ROOT:-results/realdata/T1D/$(date +%Y%m%d_%H%M%S)}"
# Explicit single-arm CLI selectors also control the batch dimensions and root.
for ((i=0;i<${#t1d_args[@]};i++)); do
    key="${t1d_args[i]}"
    value="${key#*=}"
    if [[ "$key" != *=* ]]; then value="${t1d_args[i+1]:-}"; fi
    case "${key%%=*}" in
        --outdir) OUT_ROOT="$value";;
        --prior) PRIORS="$value";;
        --constraint) CONSTRAINTS="$value";;
        --adjusted)
            case "${value,,}" in true|1|yes|y) ADJUSTMENTS=adjusted;; *) ADJUSTMENTS=unadjusted;; esac;;
    esac
done
IFS=',' read -ra t1d_priors <<< "${PRIORS:-cox-rhs,prox-sslasso}"
IFS=',' read -ra t1d_constraints <<< "${CONSTRAINTS:-prox,soft_center}"
IFS=',' read -ra t1d_adjustments <<< "${ADJUSTMENTS:-unadjusted,adjusted}"
t1d_arm() {
    local constraint="$1" prior="$2" adjustment="$3" adjusted=false
    [[ "$adjustment" != adjusted ]] || adjusted=true
    run_logged "${constraint}_${prior}_${adjustment}" "${LOG_ROOT:-$OUT_ROOT/logs}" run_julia scripts/lib/realdata/analyze.jl --workflow t1d \
        --input data/T1D.csv --outdir "$OUT_ROOT" --constraint "$constraint" \
        --prior "$prior" --adjusted "$adjusted" --stage all "${t1d_args[@]}" || return

}
for constraint in "${t1d_constraints[@]}"; do
for prior in "${t1d_priors[@]}"; do
for adjustment in "${t1d_adjustments[@]}"; do
    [[ "$adjustment" == adjusted || "$adjustment" == unadjusted ]] || { echo 'Invalid adjustment' >&2; exit 2; }
    start_job t1d_arm "$constraint" "$prior" "$adjustment"
done; done; done
wait_jobs

# Cross all completed arms after workers finish; per-arm comparisons are also saved.
stage=all
run_ph="${RUN_PH:-true}"
for ((i=0;i<${#t1d_args[@]};i++)); do
    case "${t1d_args[i]}" in
        --stage) stage="${t1d_args[i+1]}";; --stage=*) stage="${t1d_args[i]#*=}";;
        --run-ph) run_ph="${t1d_args[i+1]}";; --run-ph=*) run_ph="${t1d_args[i]#*=}";;
    esac
done
if [[ "$run_ph" == true && ( "$stage" == all || "$stage" == final ) ]]; then
    run_logged ph-comparison "${LOG_ROOT:-$OUT_ROOT/logs}" run_julia scripts/lib/diagnostics/ph.jl \
        --root "$OUT_ROOT" --aggregate-only true
fi
