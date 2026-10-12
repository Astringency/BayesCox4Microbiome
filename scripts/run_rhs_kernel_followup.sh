#!/usr/bin/env bash
set -euo pipefail
tools_dir="$(cd "$(dirname "$0")" && pwd)"
python3 "$tools_dir/run_rhs_kernel_followup.py" \
    --revision-root "${REVISION_ROOT:?set the isolated revision deployment root}" \
    --julia "${JULIA_BIN:-julia}" --workers "${RHS_WORKERS:-5}"
