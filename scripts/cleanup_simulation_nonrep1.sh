#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/cleanup_simulation_nonrep1.sh [--root PATH] [--delete] [--force]

Default:
  Dry-run only. It prints non-rep1 files/directories that would be removed.

Options:
  --root PATH   Simulation root. Default: results/simulation/softcenter
  --delete      Actually remove matched files and directories.
  --force       Do not require core reusable outputs before cleanup.
  --dry-run     Print targets without deleting them.
  -h, --help    Show this message.

The script removes only extras inside run directories whose basename starts
with repN_ where N != 1:
  diagnose/
  result_bundle.jld2
  *sample*long*.csv
  *sample*long*.csv.gz
USAGE
}

ROOT="results/simulation/softcenter"
APPLY=0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      [[ $# -ge 2 ]] || { echo "ERROR: --root requires a path." >&2; exit 2; }
      ROOT="$2"
      shift 2
      ;;
    --delete|--apply)
      APPLY=1
      shift
      ;;
    --dry-run)
      APPLY=0
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -d "$ROOT" ]] || { echo "ERROR: simulation root does not exist: $ROOT" >&2; exit 1; }

bytes_for_path() {
  local path="$1"
  if [[ -d "$path" ]]; then
    du -sb -- "$path" | awk '{print $1}'
  else
    stat -c '%s' -- "$path"
  fi
}

human_bytes() {
  local bytes="$1"
  awk -v b="$bytes" 'BEGIN {
    split("B KiB MiB GiB TiB", unit, " ");
    i = 1;
    while (b >= 1024 && i < 5) { b /= 1024; i++ }
    printf "%.2f %s", b, unit[i]
  }'
}

has_core_outputs() {
  local run="$1"
  [[ -s "$run/samples_wide.csv" &&
     -s "$run/posterior_summary.csv" &&
     -s "$run/metrics.csv" &&
     -s "$run/false_discovery.csv" ]]
}

matched_runs=0
skipped_runs=0
target_count=0
total_bytes=0

while IFS= read -r -d '' run; do
  base="$(basename "$run")"
  if [[ "$base" =~ ^rep([0-9]+)_ ]] && (( BASH_REMATCH[1] != 1 )); then
    ((matched_runs += 1))

    if (( FORCE == 0 )) && ! has_core_outputs "$run"; then
      echo "SKIP missing core reusable outputs: $run"
      ((skipped_runs += 1))
      continue
    fi

    targets=()
    [[ -d "$run/diagnose" ]] && targets+=("$run/diagnose")
    [[ -f "$run/result_bundle.jld2" ]] && targets+=("$run/result_bundle.jld2")

    while IFS= read -r -d '' sample_file; do
      targets+=("$sample_file")
    done < <(
      find "$run" -maxdepth 1 -type f \
        \( -name '*sample*long*.csv' -o -name '*sample*long*.csv.gz' \) \
        -print0
    )

    for target in "${targets[@]}"; do
      size="$(bytes_for_path "$target")"
      target_count=$((target_count + 1))
      total_bytes=$((total_bytes + size))
      if (( APPLY == 1 )); then
        echo "REMOVE $(human_bytes "$size") $target"
        rm -rf -- "$target"
      else
        echo "DRYRUN $(human_bytes "$size") $target"
      fi
    done
  fi
done < <(find "$ROOT" -type d -path '*/runs/rep*' -print0)

echo
echo "Matched non-rep1 run dirs: $matched_runs"
echo "Skipped run dirs: $skipped_runs"
echo "Cleanup targets: $target_count"
echo "Potential freed space: $(human_bytes "$total_bytes")"

if (( APPLY == 0 )); then
  echo "Dry-run complete. Re-run with --delete to remove these targets."
else
  echo "Cleanup complete."
fi
