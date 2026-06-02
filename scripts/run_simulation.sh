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

# MAIN LOOP
n_vals=(400)
# n_vals=(200)400 1000 2000 3000
# p_vals=(50 150)
p_vals=(50)
# non_vals=(6 12)
non_vals=(6)
# baseline_vals=("weibull" "exponential" "gompertz")
baseline_vals=("gompertz")
sampler_vals=("mala") # hmc, nuts, mala, ula
# hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs

for sampler in "${sampler_vals[@]}"; do
    for n in "${n_vals[@]}"; do
        for p in "${p_vals[@]}"; do
            for nonzero in "${non_vals[@]}"; do
                for baseline in "${baseline_vals[@]}"; do
                    echo "[SIMULATION] n=${n}, p=${p}, ${nonzero} nonzero coeffs with ${baseline} baseline."

                    julia "+1.11.7" -t 4 --project=. ./scripts/run_simulation.jl \
                        --outdir "./results/simulation/test/n${n}_p${p}_nonzero${nonzero}_${baseline}_${current_time}" \
                        --reps 1 \
                        --n "$n" \
                        --p "$p" \
                        --nonzero "$nonzero" \
                        --baseline "$baseline" \
                        --priors prox-sslasso \
                        --sampler "$sampler" \
                        --nuts-backend explicit \
                        --hmc-n-leapfrog 20 \
                        --samples 5000 \
                        --warmup 5000 \
                        --thin 1 \
                        --chains 4 \
                        --multichains true \
                        --target-accept 0.95 \
                        --noncentered true \
                        --seed 1 \
                        --use-car true \
                        --progress true \
                        --car-weight-quantile 0.99 \
                        --car-precision 1 \
                        --sample-car-strength true \
                        --car-strength-prior-a 1 \
                        --car-strength-prior-b 1 \
                        --car-distance "bray" \
                        --p0 truth \
                        --p0-alpha 0.05 \
                        --constraint prox \
                        --prox-sum0-lambda 1e-5 \
                        --prox-l1-lambda 1e-5 \
                        --prox-root-tol 1e-10 \
                        --prox-root-maxiter 100 \
                        --mala-step-size 1e-2 \
                        --mala-target-accept 0.574 \
                        --mala-adapt-every 1 \
                        --mala-min-step 1e-8 \
                        --mala-max-step 1.0 \
                        --mala-precondition true \
                        --mala-stepsize-schedule adaptive_sgld \
                        --mala-sgld-c0 1.0 \
                        --mala-sgld-adapt-c0 true \
                        --diagnose true \
                        --debug false \
                        --diagnose-format "pdf" \
                        --diagnose-top-k-beta 10 \
                        --diagnose-max-lag 50 \
                        --resume true \
                        --pseudocount 0.0 \
                        --standardize false \
                        --plot-first-rep-only true \
                        --sample-thin 10 \
                        --first-rep-sample-thin 1 \
                        --save-wide-csv true \
                        --save-jld2 false \
                        --alpha-prior-shape "$((p+1))" \
                        --alpha-prior-rate 100 \
                        --rhs-slab-df 4 \
                        --rhs-slab-s2 2 \
                        --hs-tau-scale 1 \
                        --hs-local-scale 1 \
                        --rhs-tau-scale 1 \
                        --rhs-local-scale 1 \
                        --rhs-pseudo-sigma-df 3 \
                        --rhs-pseudo-sigma-scale 10 \
                        --lasso-lambda2-shape 1 \
                        --lasso-lambda2-rate 0.1 \
                        --sslasso-spike-scale 0.04 \
                        --sslasso-slab-scale 0.5 \
                        --sslasso-theta-a 1 \
                        --sslasso-theta-b 1
                done
            done
        done
    done
done

wait

# --outdir "/large_storage/zhangxf/BayesCox/results/simulation/softcenter/n${n}_p${p}_nonzero${nonzero}_${baseline}"
echo "ALL SIMULATION DONE!"



