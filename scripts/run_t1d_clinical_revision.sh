#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 GKSwstype=100
JULIA_BIN="${JULIA_BIN:-julia}"
REVISION_OUT="${REVISION_OUT:-./results/revision_20261011}"
METHOD="${METHOD:-cox-rhs}"; ADJUSTED="${ADJUSTED:-true}"
label=unadjusted; [[ "$ADJUSTED" != true ]] || label=adjusted
"$JULIA_BIN" -t "${CHAINS:-4}" --startup-file=no --project=. scripts/run_t1d_clinical_revision.jl \
    --outdir "$REVISION_OUT/T1D_clinical" --prior "$METHOD" --adjusted "$ADJUSTED" \
    --historical-root "${T1D_HISTORICAL_ROOT:-./results/realdata/prox/T1D}" \
    --samples "${REAL_SAMPLES:-800}" --warmup "${REAL_WARMUP:-800}" --chains "${CHAINS:-4}" \
    --outer-folds 5 --inner-folds 5 --seed 1 --leapfrog 100
Rscript scripts/check_t1d_ph.R data/T1D.csv "$REVISION_OUT/T1D_clinical/$label/$METHOD/PH" \
    "$REVISION_OUT/T1D_clinical/$label/$METHOD/ph_input.csv"
