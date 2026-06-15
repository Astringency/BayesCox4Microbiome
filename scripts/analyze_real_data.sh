#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export GKSwstype="${GKSwstype:-100}"

# ---------------------------------------------------------------------------
# Julia runner
# ---------------------------------------------------------------------------
JULIA=(julia +1.11.7 -t 4 --project=.)

# ---------------------------------------------------------------------------
# Real-data CLI defaults from scripts/analyze_real_data.jl
# ---------------------------------------------------------------------------
# REALDATA_PRIORS=(--priors hs,rhs,new-rhs,cox-hs,cox-rhs,lasso,sslasso)
REALDATA_PRIORS=(--priors hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso,prox-lasso,prox-sslasso)
# REALDATA_PRIORS=(--priors est-rhs)

# ---------------------------------------------------------------------------
# Sampler interfaces: general, NUTS, HMC, MALA, MALA-SGLD, Proximal block
# ---------------------------------------------------------------------------
SAMPLER_GENERAL=(
    --sampler nuts
    --samples 5000
    --warmup 4000
    --thin 1
    --chains 4
    --multichains true
    --target-accept 0.8
    --noncentered true
    --seed 1
    --progress true
)

SAMPLER_NUTS=(
    --nuts-backend explicit
    --max-depth 10
)

# Optional: add --hmc-step-size <value> here if needed.
SAMPLER_HMC=(
    --hmc-n-leapfrog 100
)

SAMPLER_MALA=(
    --mala-step-size 1e-4
    --mala-target-accept 0.574
    --mala-adapt-every 50
    --mala-min-step 1e-8
    --mala-max-step 1.0
    --mala-precondition false
    --mala-precondition-mode auto
    --mala-precondition-diag false
    --mala-metric-ridge 1e-6
    --mala-metric-max-cholesky-tries 8
    --mala-stepsize-schedule adaptive_mcmc
)

SAMPLER_MALA_SGLD=(
    --mala-sgld-c0 1.0
    --mala-sgld-target-accept 0.574
    --mala-sgld-accept-low 0.40
    --mala-sgld-accept-high 0.60
    --mala-sgld-adapt-c0 true
    --mala-sgld-adapt-every 50
    --mala-sgld-c0-min 1e-12
    --mala-sgld-c0-max 1e12
)

# Optional: add --prox-block-step-size <value> here if needed.
SAMPLER_PROX_BLOCK=(
    --prox-block-min-step-size 1e-8
    --prox-block-max-step-size 1.0
    --prox-block-adapt-every 1
    --prox-block-divergence-threshold 1000.0
    --prox-block-metric unit_diag
)

# ---------------------------------------------------------------------------
# Prior interfaces: horseshoe, RHS, lasso, spike-and-slab lasso, proximal L1, CAR
# ---------------------------------------------------------------------------
PRIOR_HORSESHOE=(
    --hs-tau-scale 1.0
    --hs-local-scale 1.0
)

PRIOR_RHS=(
    --rhs-slab-df 4.0
    --rhs-slab-s2 2.0
    --rhs-tau-scale 1.0
    --rhs-local-scale 1.0
    --rhs-pseudo-sigma-df 3.0
    --rhs-pseudo-sigma-scale 10.0
)

# Optional: add --lasso-lambda <value> here if needed.
PRIOR_LASSO=(
    --lasso-lambda2-shape 1.0
    --lasso-lambda2-rate 0.1
)

PRIOR_SSLASSO=(
    --sslasso-spike-scale 1.0
    --sslasso-slab-scale 10.0
    --sslasso-theta-a 1.0
    --sslasso-theta-b 1.0
)

PRIOR_PROXIMAL_L1=(
    --prox-l1-lambda 1e-3
    --alpha-prior-shape 1.0
    --alpha-prior-scale 1.0
    --prox-root-tol 1e-10
    --prox-root-maxiter 100
)

PRIOR_CAR=(
    --use-car false
    --car-distance bray
    --car-weight-quantile 0.99
    --car-precision 1.0
    --sample-car-strength true
    --car-strength-prior-a 1.0
    --car-strength-prior-b 1.0
    --car-degree-scale mean_active
    --car-kappa-init 0.5
)

# ---------------------------------------------------------------------------
# Cox model and diagnostics interfaces
# ---------------------------------------------------------------------------
# COX_MODEL=(
#     --p0 auto
#     --p0-alpha 0.05
#     --constraint soft_center
#     --prox-sum0-lambda 1e-6
#     --pseudocount 1e-6
#     --standardize false
#     --tie-method efron
#     --approx false
#     --approx-point zero
# )

COX_MODEL=(
    --p0 auto
    --p0-alpha 0.05
    --constraint prox
    --prox-sum0-lambda 1e-6
    --pseudocount 1e-6
    --standardize false
    --tie-method efron
    --approx false
    --approx-point zero
)

DIAGNOSTICS=(
    --diagnose true
    --diagnose-format pdf
    --diagnose-top-k-beta 10
    --diagnose-max-lag 50
)

TUNING=(
    --tune_priorsparams true
    --tune_algparams false
    --cv-folds 5
    --tune-seed 42
)

# Keep COMMON as the single reusable argument bundle used by all analyses.
COMMON=(
    "${REALDATA_PRIORS[@]}"
    "${SAMPLER_GENERAL[@]}"
    "${SAMPLER_NUTS[@]}"
    "${SAMPLER_HMC[@]}"
    "${SAMPLER_MALA[@]}"
    "${SAMPLER_MALA_SGLD[@]}"
    "${SAMPLER_PROX_BLOCK[@]}"
    "${PRIOR_HORSESHOE[@]}"
    "${PRIOR_RHS[@]}"
    "${PRIOR_LASSO[@]}"
    "${PRIOR_SSLASSO[@]}"
    "${PRIOR_PROXIMAL_L1[@]}"
    "${PRIOR_CAR[@]}"
    "${COX_MODEL[@]}"
    "${DIAGNOSTICS[@]}"
    "${TUNING[@]}"
)

# ---------------------------------------------------------------------------
# Dataset-specific interfaces
# ---------------------------------------------------------------------------
T1D_DATA=(
    --input data/T1D.csv
    --outdir results/realdata/T1D
    --time-col time
    --status-col status
    --comp-cols none
    --exclude-cols Sex,Antibiotics
)

HCT_DATA=(
    --input data/HCT.csv
    --outdir results/realdata/HCT
    --time-col time
    --status-col status
    --comp-cols none
    --exclude-cols sample_id
)

HCT_DATA_001=(
    --input data/HCT_001.csv
    --outdir results/realdata/HCT_001
    --time-col time
    --status-col status
    --comp-cols none
    --exclude-cols sample_id
)

TCGA_EXCLUDE="dataset,patient,cancer_type,sample_id,sample_type,pfi_time,pfi,dfi_time,dfi,dss_time,dss,os_censor,pfi_censor,dfi_censor,dss_censor"

run_realdata() {
    "${JULIA[@]}" scripts/analyze_real_data.jl "$@"
}

run_tcga() {
    local cancer="$1"
    local tcga_data=(
        --input "data/TCGA/cibersort/TCGA_${cancer}_CIBERSORT_survival.csv"
        --outdir "results/realdata/TCGA_${cancer}_CIBERSORT_OS"
        --time-col os_time
        --status-col os
        --comp-cols none
        --exclude-cols "${TCGA_EXCLUDE}"
    )

    run_realdata "${tcga_data[@]}" "${COMMON[@]}" "$@"
}

optional_arg() {
    # Usage:
    #   optional_arg ARRAY_NAME cli-key value
    # Example:
    #   optional_arg SAMPLER_HMC_STEP hmc-step-size "${HMC_STEP_SIZE:-}"
    local array_name="$1"
    local key="$2"
    local value="${3:-}"

    if [[ -n "${value}" ]]; then
        eval "${array_name}=(--${key} \"\${value}\")"
        else
        eval "${array_name}=()"
    fi
}

# ---------------------------------------------------------------------------
# 1. T1D data: data/T1D.csv (173, 30)
# ---------------------------------------------------------------------------
# echo ">>> RUN T1D DATA ANALYSIS"
# run_realdata "${T1D_DATA[@]}" "${COMMON[@]}" "$@"

# ---------------------------------------------------------------------------
# 2. HCT data: data/HCT.csv (63, 139/33)
# ---------------------------------------------------------------------------
# echo ">>> RUN HCT DATA ANALYSIS"
# run_realdata "${HCT_DATA[@]}" "${COMMON[@]}" "$@"

echo ">>> RUN PROCESSED HCT DATA ANALYSIS"
run_realdata "${HCT_DATA_001[@]}" "${COMMON[@]}" "$@"

# ---------------------------------------------------------------------------
# 3. TCGA CIBERSORT data: 8 cancer-specific OS analyses (n, 22)
# ---------------------------------------------------------------------------
# echo ">>> RUN TCGA DATA ANALYSIS"
# for cancer in BLCA BRCA COAD HNSC LIHC LUAD OV STAD; do
#     run_tcga "${cancer}" "$@"
# done


echo "ALL DONE."
