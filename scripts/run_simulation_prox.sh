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

current_time=$(date "+%Y%m%d_%H%M%S")
pids=()
status=0

# MAIN LOOP
n_vals=(200 400)
p_vals=(50 150)
non_vals=(6 12)
# non_vals=(2)
baseline_vals=("weibull" "exponential" "gompertz")

pids=()

for n in "${n_vals[@]}"; do
    for p in "${p_vals[@]}"; do
        for nonzero in "${non_vals[@]}"; do
            for baseline in "${baseline_vals[@]}"; do
                echo "[SIMULATION] n=${n}, p=${p}, ${nonzero} nonzero coeffs with ${baseline} baseline."

                julia "+1.11.7" -t 4 --project=. scripts/run_simulation.jl \
                    --sampler hmc \
                    --nuts-backend explicit \
                    --hmc-n-leapfrog 20 \
                    --samples 5000 \
                    --warmup 2000 \
                    --thin 1 \
                    --chains 4 \
                    --multichains true \
                    --reps 50 \
                    --n "$n" \
                    --p "$p" \
                    --nonzero "$nonzero" \
                    --baseline "$baseline" \
                    --constraint prox \
                    --priors hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso \
                    --target-accept 0.95 \
                    --noncentered true \
                    --prox-sum0-lambda 1e-3 \
                    --prox-l1-lambda 1e-5 \
                    --alpha-prior-shape 1 \
                    --alpha-prior-scale 1 \
                    --seed 1 \
                    --use-car true \
                    --car-weight-quantile 0.99 \
                    --car-precision 1 \
                    --sample-car-strength true \
                    --car-strength-prior-a 1 \
                    --car-strength-prior-b 1 \
                    --car-distance "bray" \
                    --p0 truth \
                    --p0-alpha 0.05 \
                    --outdir "/large_storage/zhangxf/BayesCox/results/simulation/prox_c/n${n}_p${p}_nonzero${nonzero}_${baseline}" &
                pids+=("$!")
            done
        done
    done
done

# /large_storage/zhangxf/BayesCox
# prox-lasso,prox-sslasso

wait "${pids[@]}"

if [[ "$status" -ne 0 ]]; then
    echo "SIMULATION FAILED!" >&2
    exit "$status"
fi

echo "ALL SIMULATION DONE!"
