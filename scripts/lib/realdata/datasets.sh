#!/usr/bin/env bash
ALL_REALDATA_DATASETS=T1D,HCT,HCT_001,HCT_bcorsis,TCGA_BLCA,TCGA_BRCA,TCGA_COAD,TCGA_HNSC,TCGA_LIHC,TCGA_LUAD,TCGA_OV,TCGA_STAD
realdata_dataset() {
    local dataset="${1//[[:space:]]/}"
    DATA_INPUT="data/$dataset.csv"; DATA_NAME="$dataset"; DATA_CONFIG="$dataset"
    DATA_TIME=time; DATA_STATUS=status; DATA_EXCLUDE=sample_id
    case "$dataset" in
        T1D) DATA_EXCLUDE=Sex,Antibiotics; DATA_P=30;;
        HCT) DATA_P=139;; HCT_001) DATA_P=33;;
        HCT_bcorsis) DATA_P=15; DATA_CONFIG=HCT_001;;
        TCGA_*)
            DATA_INPUT="data/TCGA/cibersort/${dataset}_CIBERSORT_survival.csv"
            DATA_NAME="${dataset}_CIBERSORT_OS"; DATA_CONFIG="$DATA_NAME"; DATA_P=22
            DATA_TIME=os_time; DATA_STATUS=os
            DATA_EXCLUDE=dataset,patient,cancer_type,sample_id,sample_type,pfi_time,pfi,dfi_time,dfi,dss_time,dss,os_censor,pfi_censor,dfi_censor,dss_censor;;
        *) echo "Unknown dataset: $dataset" >&2; return 2;;
    esac
}
