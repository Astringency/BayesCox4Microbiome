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

# TEST
# test_n=400
# test_p=150
# pids=()
# julia "+1.11.7" -t 4 --project=. scripts/run_simulation.jl \
#     --sampler ula \
#     --samples 5000 \
#     --warmup 2000 \
#     --chains 4 \
#     --multichains \
#     --reps 1 \
#     --n "$test_n" \
#     --p "$test_p" \
#     --nonzero 12 \
#     --baseline weibull \
#     --priors prox-lasso \
#     --constraint prox \
#     --target-accept 0.8 \
#     --mala-step-size 1e-3 \
#     --mala-target-accept 0.7 \
#     --seed 1 \
#     --use-car true \
#     --car-distance bray \
#     --p0 truth \
#     --p0-alpha 0.05 \
#     --outdir "results/simulation/test_n${test_n}_p${test_p}_non12_weibull_${current_time}" &
# pids+=("$!")

# MAIN LOOP
n_vals=(200 400)
p_vals=(50 150)
non_vals=(6 12)
baseline_vals=("weibull" "exponential" "gompertz")

pids=()

for n in "${n_vals[@]}"; do
  for p in "${p_vals[@]}"; do
    for nonzero in "${non_vals[@]}"; do

      (
        for baseline in "${baseline_vals[@]}"; do
          echo "[SIMULATION] n=${n}, p=${p}, ${nonzero} nonzero coeffs with ${baseline} baseline."

          julia "+1.11.7" -t 4 --project=. scripts/run_simulation.jl \
            --sampler nuts \
            --nuts-backend explicit \
            --samples 5000 \
            --warmup 2000 \
            --chains 4 \
            --reps 50 \
            --n "$n" \
            --p "$p" \
            --nonzero "$nonzero" \
            --baseline "$baseline" \
            --constraint soft_center \
            --priors hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso \
            --target-accept 0.95 \
            --seed 1 \
            --use-car true \
            --car-distance bray \
            --p0 truth \
            --p0-alpha 0.05 \
            --outdir "/large_storage/zhangxf/BayesCox/results/simulation/softcenter/n${n}_p${p}_nonzero${nonzero}_${baseline}"
        done
      ) &

      pids+=("$!")

    done
  done
done

wait "${pids[@]}"

if [[ "$status" -ne 0 ]]; then
    echo "SIMULATION FAILED!" >&2
    exit "$status"
fi

echo "ALL SIMULATION DONE!"
