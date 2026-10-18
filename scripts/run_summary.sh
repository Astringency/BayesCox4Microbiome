#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/common.sh
if [[ "${1:-}" == --help ]]; then
    echo 'Usage: bash scripts/run_summary.sh [--root DIR] [Julia options]'
    echo '       bash scripts/run_summary.sh --plots-only --manifest diagnostics_by_run.csv [--rep 1|all] [--suite main] [--output-dir DIR]'
    echo '       bash scripts/run_summary.sh --plots --root DIR [Julia options]'
    echo 'PLOT_REP selects the repetition after --plots (default 1). ROOTS supports multiple legacy roots.'
    exit 0
fi
if [[ "${1:-}" == --plots-only ]]; then
    shift
    exec bash scripts/plot_simulation.sh "$@"
fi
with_plots=0
if [[ "${1:-}" == --plots ]]; then with_plots=1; shift; fi
args=("$@")
explicit_root=''
for ((i=0;i<${#args[@]};i++)); do
    if [[ "${args[i]}" == --root ]]; then explicit_root="${args[i+1]}"; fi
    if [[ "${args[i]}" == --root=* ]]; then explicit_root="${args[i]#--root=}"; fi
done
if [[ -n "$explicit_root" ]]; then
    run_julia scripts/lib/simulation/summarize.jl "${args[@]}"
    if ((with_plots)); then bash scripts/plot_simulation.sh --root "$explicit_root" --rep "${PLOT_REP:-1}"; fi
    exit 0
fi
IFS=',' read -ra roots <<< "${ROOTS:-results/simulation/prox,results/simulation/soft_center}"
for root in "${roots[@]}"; do
    run_julia scripts/lib/simulation/summarize.jl --root "$root" "${args[@]}"
    if ((with_plots)); then bash scripts/plot_simulation.sh --root "$root" --rep "${PLOT_REP:-1}"; fi
done
