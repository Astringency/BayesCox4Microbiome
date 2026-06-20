#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export GKSwstype="${GKSwstype:-100}"

JULIA_BIN="${JULIA_BIN:-julia +1.11.7}"
THREADS="${THREADS:-4}"
SAMPLES="${SAMPLES:-800}"
WARMUP="${WARMUP:-800}"
CHAINS="${CHAINS:-4}"
TARGET_ACCEPT="${TARGET_ACCEPT:-0.9}"
MAX_DEPTH="${MAX_DEPTH:-10}"
SEED="${SEED:-20260618}"
PROX_SUM0_LAMBDA="${PROX_SUM0_LAMBDA:-1e-6}"
SOFT_CENTER_PROX_SUM0_LAMBDA="${SOFT_CENTER_PROX_SUM0_LAMBDA:-1e-6}"
PROX_NONCENTERED="${PROX_NONCENTERED:-false}"
SOFT_CENTER_NONCENTERED="${SOFT_CENTER_NONCENTERED:-true}"
OUT_ROOT="${OUT_ROOT:-results/realdata}"
TUNING_ROOT="${TUNING_ROOT:-results/realdata_tuning}"
FALLBACK_TUNING_ROOT="${FALLBACK_TUNING_ROOT:-results/realdata}"
CONSTRAINTS="${CONSTRAINTS:-soft_center,prox}"
RUN_DATASETS="${RUN_DATASETS:-all}"
PRIORS="${PRIORS:-hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso,prox-lasso,prox-sslasso}"

COMMON_ARGS=(
    --priors "$PRIORS"
    --sampler nuts
    --nuts-backend explicit
    --samples "$SAMPLES"
    --warmup "$WARMUP"
    --thin 1
    --chains "$CHAINS"
    --multichains false
    --target-accept "$TARGET_ACCEPT"
    --max-depth "$MAX_DEPTH"
    --seed "$SEED"
    --progress false
    --pseudocount 1e-6
    --standardize false
    --tie-method efron
    --approx false
    --approx-point zero
    --diagnose false
    --tune_priorsparams false
    --tune_algparams false
    --hs-tau-scale 1.0
    --hs-local-scale 1.0
    --rhs-slab-df 4.0
    --rhs-slab-s2 2.0
    --rhs-tau-scale 1.0
    --rhs-local-scale 1.0
    --rhs-pseudo-sigma-df 3.0
    --rhs-pseudo-sigma-scale 10.0
    --lasso-lambda2-shape 1.0
    --lasso-lambda2-rate 10.0
    --sslasso-spike-scale 1.0
    --sslasso-slab-scale 10.0
    --sslasso-theta-a 1.0
    --sslasso-theta-b 1.0
    --prox-l1-lambda 1e-3
    --alpha-prior-shape 1.0
    --prox-root-tol 1e-10
    --prox-root-maxiter 100
    --use-car false
)

dataset_spec() {
    case "$1" in
        T1D) echo "data/T1D.csv|time|status|Sex,Antibiotics|30" ;;
        HCT) echo "data/HCT.csv|time|status|sample_id|139" ;;
        HCT_001) echo "data/HCT_001.csv|time|status|sample_id|33" ;;
        HCT_bcorsis) echo "data/HCT_bcorsis.csv|time|status|sample_id|15" ;;
        TCGA_BLCA) echo "data/TCGA/cibersort/TCGA_BLCA_CIBERSORT_survival.csv|os_time|os|dataset,patient,cancer_type,sample_id,sample_type,pfi_time,pfi,dfi_time,dfi,dss_time,dss,os_censor,pfi_censor,dfi_censor,dss_censor|22" ;;
        TCGA_BRCA) echo "data/TCGA/cibersort/TCGA_BRCA_CIBERSORT_survival.csv|os_time|os|dataset,patient,cancer_type,sample_id,sample_type,pfi_time,pfi,dfi_time,dfi,dss_time,dss,os_censor,pfi_censor,dfi_censor,dss_censor|22" ;;
        TCGA_COAD) echo "data/TCGA/cibersort/TCGA_COAD_CIBERSORT_survival.csv|os_time|os|dataset,patient,cancer_type,sample_id,sample_type,pfi_time,pfi,dfi_time,dfi,dss_time,dss,os_censor,pfi_censor,dfi_censor,dss_censor|22" ;;
        TCGA_HNSC) echo "data/TCGA/cibersort/TCGA_HNSC_CIBERSORT_survival.csv|os_time|os|dataset,patient,cancer_type,sample_id,sample_type,pfi_time,pfi,dfi_time,dfi,dss_time,dss,os_censor,pfi_censor,dfi_censor,dss_censor|22" ;;
        TCGA_LIHC) echo "data/TCGA/cibersort/TCGA_LIHC_CIBERSORT_survival.csv|os_time|os|dataset,patient,cancer_type,sample_id,sample_type,pfi_time,pfi,dfi_time,dfi,dss_time,dss,os_censor,pfi_censor,dfi_censor,dss_censor|22" ;;
        TCGA_LUAD) echo "data/TCGA/cibersort/TCGA_LUAD_CIBERSORT_survival.csv|os_time|os|dataset,patient,cancer_type,sample_id,sample_type,pfi_time,pfi,dfi_time,dfi,dss_time,dss,os_censor,pfi_censor,dfi_censor,dss_censor|22" ;;
        TCGA_OV) echo "data/TCGA/cibersort/TCGA_OV_CIBERSORT_survival.csv|os_time|os|dataset,patient,cancer_type,sample_id,sample_type,pfi_time,pfi,dfi_time,dfi,dss_time,dss,os_censor,pfi_censor,dfi_censor,dss_censor|22" ;;
        TCGA_STAD) echo "data/TCGA/cibersort/TCGA_STAD_CIBERSORT_survival.csv|os_time|os|dataset,patient,cancer_type,sample_id,sample_type,pfi_time,pfi,dfi_time,dfi,dss_time,dss,os_censor,pfi_censor,dfi_censor,dss_censor|22" ;;
        *) return 2 ;;
    esac
}

out_dataset_name() {
    case "$1" in
        TCGA_*) echo "${1}_CIBERSORT_OS" ;;
        *) echo "$1" ;;
    esac
}

if [[ "$RUN_DATASETS" == "all" ]]; then
    RUN_DATASETS="T1D,HCT,HCT_001,HCT_bcorsis,TCGA_BLCA,TCGA_BRCA,TCGA_COAD,TCGA_HNSC,TCGA_LIHC,TCGA_LUAD,TCGA_OV,TCGA_STAD"
fi

IFS=',' read -r -a DATASETS <<< "$RUN_DATASETS"
IFS=',' read -r -a CONSTRAINT_LIST <<< "$CONSTRAINTS"

for raw_constraint in "${CONSTRAINT_LIST[@]}"; do
    constraint="${raw_constraint//[[:space:]]/}"
    [[ -z "$constraint" ]] && continue
    if [[ "$constraint" == "prox" ]]; then
        noncentered="$PROX_NONCENTERED"
        prox_sum0_lambda="$PROX_SUM0_LAMBDA"
    else
        noncentered="$SOFT_CENTER_NONCENTERED"
        prox_sum0_lambda="$SOFT_CENTER_PROX_SUM0_LAMBDA"
    fi

    for raw_dataset in "${DATASETS[@]}"; do
        dataset="${raw_dataset//[[:space:]]/}"
        [[ -z "$dataset" ]] && continue
        spec="$(dataset_spec "$dataset")"
        IFS='|' read -r input time_col status_col exclude_cols p <<< "$spec"
        out_name="$(out_dataset_name "$dataset")"
        prior_configs="${TUNING_ROOT}/${constraint}/${out_name}/tuning/best_prior_configs.json"
        if [[ ! -f "$prior_configs" ]]; then
            prior_configs="${FALLBACK_TUNING_ROOT}/${constraint}/${out_name}/tuning/best_prior_configs.json"
        fi
        if [[ ! -f "$prior_configs" ]]; then
            prior_configs="${TUNING_ROOT}/soft_center/${out_name}/tuning/best_prior_configs.json"
        fi
        if [[ ! -f "$prior_configs" ]]; then
            prior_configs="${FALLBACK_TUNING_ROOT}/soft_center/${out_name}/tuning/best_prior_configs.json"
        fi
        [[ -f "$prior_configs" ]] || prior_configs="none"

        echo ">>> constraint=${constraint} dataset=${out_name} p=${p} p0=auto"
        ${JULIA_BIN} -t "$THREADS" --project=. scripts/analyze_real_data.jl \
            --input "$input" \
            --outdir "${OUT_ROOT}/${constraint}/${out_name}" \
            --time-col "$time_col" \
            --status-col "$status_col" \
            --comp-cols none \
            --exclude-cols "$exclude_cols" \
            --constraint "$constraint" \
            --prox-sum0-lambda "$prox_sum0_lambda" \
            --noncentered "$noncentered" \
            --p0 auto \
            --alpha-prior-scale "$p" \
            --prior-configs "$prior_configs" \
            "${COMMON_ARGS[@]}"
    done
done

echo "ALL DONE."
