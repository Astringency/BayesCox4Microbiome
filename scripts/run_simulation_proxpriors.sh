#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/simulation_progress.sh
simulation_setup "$@"

# Grid and sampling settings; edit here or override through environment variables.
IFS=',' read -ra n_vals <<< "${N_VALUES:-200,400}"
IFS=',' read -ra p_vals <<< "${P_VALUES:-50,150}"
IFS=',' read -ra non_vals <<< "${NONZERO_VALUES:-6,12}"
IFS=',' read -ra baseline_vals <<< "${BASELINES:-weibull,gompertz}"
IFS=',' read -ra constraint_vals <<< "${CONSTRAINTS:-prox}"
priors="${PRIORS:-prox-lasso}"; reps="${REPS:-50}"
init=zero; [[ ",$priors," != *,prox-sslasso,* ]] || init=rand
OUT_ROOT="${OUT_ROOT:-results/simulation/proxpriors_$SIM_STAMP}"

for n in "${n_vals[@]}"; do
for p in "${p_vals[@]}"; do
for nonzero in "${non_vals[@]}"; do
for baseline in "${baseline_vals[@]}"; do
for constraint in "${constraint_vals[@]}"; do
    label="n${n}_p${p}_nz${nonzero}_${baseline}_${constraint}"
    outdir="$OUT_ROOT/$constraint/n${n}_p${p}_nonzero${nonzero}_${baseline}"
    simulation_job "$label" "$outdir" \
        --n "$n" --p "$p" --nonzero "$nonzero" --baseline "$baseline" \
        --reps "$reps" --priors "$priors" --constraint "$constraint" \
        --sampler "${SAMPLER:-nuts}" --nuts-backend explicit --hmc-n-leapfrog "${HMC_N_LEAPFROG:-300}" \
        --samples "${SAMPLES:-5000}" --warmup "${WARMUP:-2000}" --thin 1 --chains "${CHAINS:-4}" \
        --multichains true --init "${INIT:-$init}" --target-accept "${TARGET_ACCEPT:-0.8}" --noncentered true --seed "${SEED:-1}" \
        --use-car "${USE_CAR:-true}" --progress true \
        --car-weight-quantile 0.99 --car-precision 1 --sample-car-strength "${SAMPLE_CAR_STRENGTH:-${USE_CAR:-true}}" \
        --car-strength-prior-a 1 --car-strength-prior-b "${CAR_STRENGTH_PRIOR_B:-$p}" --car-distance bray --p0 "${P0:-truth}" \
        --prox-sum0-lambda "${PROX_SUM0_LAMBDA:-1e-3}" --prox-l1-lambda "${PROX_L1_LAMBDA:-1e-6}" \
        --prox-root-tol 1e-10 --prox-root-maxiter 100 --alpha-prior-kind invgamma --alpha-prior-shape 1 --alpha-prior-scale "$p" \
        --rhs-slab-df 4 --rhs-slab-s2 2 --rhs-tau-scale 1 --rhs-local-scale 1 --hs-tau-scale 1 --hs-local-scale 1 \
        --lasso-lambda2-shape 1 --lasso-lambda2-rate 0.1 --sslasso-spike-scale 0.04 --sslasso-slab-scale 0.5 \
        --sslasso-theta-a 1 --sslasso-theta-b 1 --standardize false \
        --mala-step-size "${MALA_STEP_SIZE:-1e-2}" --mala-precondition false --mala-adapt-every 1 --mala-stepsize-schedule adaptive_sgld \
        --diagnose "${DIAGNOSE:-true}" --diagnose-format pdf --diagnose-top-k-beta 10 --diagnose-max-lag 50 \
        --resume true --plot-first-rep-only true --sample-thin "${SAMPLE_THIN:-10}" --first-rep-sample-thin 1 \
        --save-wide-csv true --save-jld2 false
done; done; done; done; done
simulation_finish
