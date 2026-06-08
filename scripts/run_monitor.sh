#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."

unset LD_LIBRARY_PATH
unset LIBRARY_PATH
unset CPATH
unset C_INCLUDE_PATH
unset CPLUS_INCLUDE_PATH
unset PKG_CONFIG_PATH

export GKSwstype=100

echo ">>> SIMULATION MONITOR prox"
julia +1.11.7 --project=. ./scripts/monitor_simulation.jl --root /large_storage/zhangxf/BayesCox/results/simulation/prox/

echo ">>> SIMULATION MONITOR soft centers"
julia +1.11.7 --project=. ./scripts/monitor_simulation.jl --root /large_storage/zhangxf/BayesCox/results/simulation/soft_center/
