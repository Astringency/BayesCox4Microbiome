#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 GKSwstype=100
JULIA_BIN="${JULIA_BIN:-julia}"
HISTORICAL_ROOT="${HISTORICAL_ROOT:-/large_storage/zhangxf/BayesCox/results}"
REVISION_OUT="${REVISION_OUT:-./results/revision_20261011}"
"$JULIA_BIN" --startup-file=no --project=. scripts/lib/diagnostics/summarize.jl \
    --roots "$HISTORICAL_ROOT" --outdir "$REVISION_OUT/historical_diagnostics" \
    --priors cox-rhs,prox-sslasso --resume true "$@"
