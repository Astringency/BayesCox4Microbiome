#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."

unset LD_LIBRARY_PATH
unset LIBRARY_PATH
unset CPATH
unset C_INCLUDE_PATH
unset CPLUS_INCLUDE_PATH
unset PKG_CONFIG_PATH

export GKSwstype="${GKSwstype:-100}"

ROOT="${ROOT:-}"
ROOTS="${ROOTS:-}"

args=("$@")
if [[ ${#args[@]} -gt 0 && "${args[0]}" != --* ]]; then
  ROOT="${args[0]}"
  args=("${args[@]:1}")
fi

if [[ -n "$ROOTS" ]]; then
  args=(--roots "$ROOTS" "${args[@]}")
elif [[ -n "$ROOT" ]]; then
  args=(--root "$ROOT" "${args[@]}")
fi

exec "${RSCRIPT_BIN:-Rscript}" ./scripts/lib/plotting/plot_simulation.R "${args[@]}"
