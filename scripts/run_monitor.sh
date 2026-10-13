#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/common.sh
if [[ "${1:-}" == --help ]]; then
    echo 'Usage: bash scripts/run_monitor.sh [--root DIR] [Julia options]; ROOTS=path1,path2 for multiple roots'
    exit 0
fi
for value in "$@"; do
    if [[ "$value" == --root ]]; then run_julia scripts/monitor_simulation.jl "$@"; exit; fi
done
IFS=',' read -ra roots <<< "${ROOTS:-results/simulation/prox,results/simulation/soft_center}"
for root in "${roots[@]}"; do
    run_julia scripts/monitor_simulation.jl --root "$root" "$@"
done
