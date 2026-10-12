#!/usr/bin/env bash
# Stop the serial clinical drivers first; reuse the SAME frozen project/output.
set -euo pipefail
tools_dir="$(cd "$(dirname "$0")" && pwd)"
python3 "$tools_dir/run_t1d_partition_pool.py" \
    --project "${BCX_FROZEN_CODE:?set the existing frozen project directory}" \
    --revision-out "${REVISION_OUT:?set its existing output root}" \
    --julia "${JULIA_BIN:-julia}" --workers "${CV_WORKERS:-8}" --method "${METHOD:-prox-sslasso}"
