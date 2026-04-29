#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ $# -eq 0 ]]; then
  julia "+1.11.7" --project="$REPO_DIR" "$SCRIPT_DIR/run_diagnose.jl"
else
  julia "+1.11.7" --project="$REPO_DIR" "$SCRIPT_DIR/run_diagnose.jl" "$@"
fi
