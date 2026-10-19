#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/common.sh

# T1D is the only bundled real-data workflow. Defaults match results/review/T1D.
SAMPLES="${SAMPLES:-5000}"
WARMUP="${WARMUP:-4000}"
CHAINS="${CHAINS:-4}"
SEED="${SEED:-1}"
TARGET_ACCEPT="${TARGET_ACCEPT:-0.95}"
HMC_N_LEAPFROG="${HMC_N_LEAPFROG:-300}"

T1D_SAMPLER=(
    --samples "$SAMPLES" --warmup "$WARMUP" --chains "$CHAINS" --seed "$SEED"
    --target-accept "$TARGET_ACCEPT" --leapfrog "$HMC_N_LEAPFROG"
)

case "${1:-}" in
    --help)
        echo 'Usage: analyze_real_data.sh [t1d] [options]'
        echo 'Runs the eight T1D arms: cox-rhs/prox-sslasso x prox/soft_center x unadjusted/adjusted.'
        echo 'Options are forwarded to scripts/lib/realdata/t1d.jl; use --outdir to choose the output root.'
        exit 0
        ;;
    t1d) shift ;;
    ''|-*) ;;
    *) echo "Unknown real-data workflow: $1 (only T1D is bundled)" >&2; exit 2 ;;
esac

exec bash scripts/lib/realdata/t1d_batch.sh "${T1D_SAMPLER[@]}" "$@"
