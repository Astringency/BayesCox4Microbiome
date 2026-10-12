#!/usr/bin/env bash
# Read-only refresh; does not launch or repeat any sampling.
set -euo pipefail
tools_dir="$(cd "$(dirname "$0")" && pwd)"
root="${1:?usage: refresh_revision_reports.sh REVISION_ROOT}"
mkdir -p "$root/logs"
if command -v flock >/dev/null 2>&1; then
    exec 9>"$root/logs/refresh_reports.lock"
    flock 9
fi
out="$root/results"
if [[ -d "$root/logs/zero_tail_assist" ]]; then
    python3 "$tools_dir/audit_zero_tail_assistance.py" --revision-root "$root" \
        --outdir "$out/zero_tail_assistance_audit"
fi
replacement_args=()
if [[ -f "$out/resolved_fixed_rhs_baseline.json" ]]; then
    replacement_args=(--replacement-manifest "$out/resolved_fixed_rhs_baseline.json")
fi
python3 "$tools_dir/backfill_revision_execution_settings.py" --revision-results "$out"
python3 "$tools_dir/aggregate_revision_diagnostics.py" \
    --historical-diagnostics "$out/historical_diagnostics" --new-root "$out" \
    "${replacement_args[@]}" \
    --kernel-audit "$out/legacy_kernel_audit.json" --outdir "$out/diagnostic_aggregates_v2"
python3 "$tools_dir/audit_revision_data_identity.py" \
    --diagnostics "$out/diagnostic_aggregates_v2/diagnostics_by_run.csv" \
    --outdir "$out/data_identity_audit"
python3 "$tools_dir/aggregate_revision_metrics.py" \
    --diagnostics "$out/diagnostic_aggregates_v2/diagnostics_by_run.csv" \
    --historical-metrics "$out/historical_metrics/metrics_by_run.csv" \
    --outdir "$out/performance_aggregates"
python3 "$tools_dir/summarize_revision_contrasts.py" \
    --metrics-by-run "$out/performance_aggregates/metrics_by_run.csv" \
    --outdir "$out/performance_aggregates"
python3 "$tools_dir/summarize_t1d_clinical_revision.py" \
    --rhs-root "$out/T1D_clinical" --ssl-root "$out/corrected_ssl/T1D_clinical" \
    --outer-audit-root "$out/T1D_outer_audit" \
    --outdir "$out/T1D_summary"
python3 "$tools_dir/verify_t1d_saved_cindex.py" \
    --revision-root "$root" --data "${T1D_DATA:-$root/code_v5/data/T1D.csv}" \
    --outdir "$out/T1D_saved_score_audit"
python3 "$tools_dir/verify_revision_saved_outputs.py" \
    --revision-root "$root" --outdir "$out/saved_outputs_audit"
