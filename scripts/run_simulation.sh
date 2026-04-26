#!/usr/bin/env bash

set -eu
cd "$(dirname "$0")/.."

current_time=$(date "+%Y-%m-%d %H:%M:%S")

# TEST
test_n=400
test_p=50
julia "+1.11.7" -t 4 --project=. scripts/run_simulation.jl \
    --sampler mala \
    --samples 5000 \
    --warmup 2000 \
    --chains 4 \
    --reps 1 \
    --n $test_n \
    --p $test_p \
    --nonzero 12 \
    --baseline weibull \
    --priors lasso \
    --constraint prox \
    --target-accept 0.8 \
    --seed 1 \
    --use-car true \
    --car-distance bray \
    --p0 truth \
    --p0-alpha 0.05 \
    --outdir results/simulation/test_n${test_n}_p${test_p}_non12_weibull_${current_time} &

# MAIN LOOP

# n_vals=(200 400)
# p_vals=(50 150 300)
# baseline_vals=("weibull" "exponential" "gompertz")
#
# for n in "${n_vals[@]}"; do
#     for p in "${p_vals[@]}"; do
#         for baseline in "${baseline_vals[@]}"; do
#
#             echo "[SIMULATION] n=${n}, p=${p} with ${baseline} baseline."
#
#             julia -t 4 --project=. scripts/run_simulation.jl \
#                 --sampler nuts \
#                 --samples 5000 \
#                 --warmup 2000 \
#                 --chains 4 \
#                 --reps 100 \
#                 --n $n \
#                 --p $p \
#                 --nonzero 6,12 \
#                 --baseline $baseline \
#                 --priors hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso \
#                 --target-accept 0.9 \
#                 --seed 1 \
#                 --use-car true \
#                 --car-distance bray \
#                 --p0 truth \
#                 --p0-alpha 0.05 \
#                 --outdir results/simulation/n${n}_p${p}_${baseline} &
#
#         done
#     done
# done

wait

echo "ALL SIMULATION DONE!"
