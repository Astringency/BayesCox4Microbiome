#!/usr/bin/env bash
# Shared execution helpers. JULIA_BIN is a binary path, not a shell command.
BCX_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export GKSwstype="${GKSwstype:-100}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
BCX_JULIA=("${JULIA_BIN:-julia}")
if [[ -z "${JULIA_BIN:-}" ]]; then BCX_JULIA+=("${JULIA_VERSION:-+1.11.7}"); fi
BCX_PIDS=()

run_command() {
    if [[ "${DRY_RUN:-false}" == true ]]; then
        printf '%q ' "$@"; printf '\n'
    else
        "$@"
    fi
}

run_julia() {
    run_command "${BCX_JULIA[@]}" -t "${THREADS:-${CHAINS:-4}}" --startup-file=no \
        "--project=$BCX_PROJECT_ROOT" "$@"
}

run_logged() {
    local label="$1" root="$2" code; shift 2
    local log="$root/${label}_$(date +%Y%m%d_%H%M%S)_${BASHPID}.log"
    if [[ "${DRY_RUN:-false}" == true ]]; then "$@"; printf '  log: %s\n' "$log"; return; fi
    mkdir -p "$root"
    printf '[运行] %s → %s\n' "$label" "$log"
    if "$@" > "$log" 2>&1; then
        printf '[完成] %s\n' "$label"
    else
        code=$?; printf '[失败 %s] %s → %s\n' "$code" "$label" "$log" >&2; return "$code"
    fi
}

wait_jobs() {
    local pid failed=0
    for pid in "${BCX_PIDS[@]}"; do
        if wait "$pid"; then :; else failed=1; fi
    done
    BCX_PIDS=()
    return "$failed"
}

start_job() {
    local workers="${WORKERS:-1}"
    [[ "$workers" =~ ^[1-9][0-9]*$ ]] || { echo 'WORKERS must be a positive integer' >&2; return 2; }
    if [[ "${DRY_RUN:-false}" == true ]]; then "$@"; return; fi
    "$@" &
    BCX_PIDS+=("$!")
    if (( ${#BCX_PIDS[@]} >= workers )); then wait_jobs; fi
}
