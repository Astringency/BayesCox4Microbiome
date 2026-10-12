#!/usr/bin/env bash
# Summarize saved S4 fits only; never launch sampling or touch previous reports.
set -euo pipefail
tools_dir="$(cd "$(dirname "$0")" && pwd)"
root="${1:?usage: refresh_s4_reports.sh S4_ROOT}"
out="$root/reports"
mkdir -p "$out"
python3 "$tools_dir/aggregate_revision_diagnostics.py" \
    --new-root "$root/car" --outdir "$out/diagnostics"
python3 "$tools_dir/audit_revision_data_identity.py" \
    --diagnostics "$out/diagnostics/diagnostics_by_run.csv" --outdir "$out/data_identity"
python3 "$tools_dir/aggregate_revision_metrics.py" --profile s4 \
    --diagnostics "$out/diagnostics/diagnostics_by_run.csv" --outdir "$out/performance"
python3 "$tools_dir/summarize_revision_contrasts.py" --profile s4 \
    --metrics-by-run "$out/performance/metrics_by_run.csv" --outdir "$out/performance"
python3 "$tools_dir/verify_s4_results.py" --root "$root"
