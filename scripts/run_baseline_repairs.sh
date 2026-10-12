#!/usr/bin/env bash
# Only explicitly selected repetitions. Historical-kernel exclusions also apply;
# use run_rhs_kernel_followup.sh for the complete audited revision baseline.
set -euo pipefail
cd "$(dirname "$0")/.."
revision_base="${REVISION_OUT:-./results/revision_20261011}"
export REP_IDS="${REP_IDS:?provide explicitly audited replicate IDs}"
export FORCE_NEW_FITS=true PRIORS=cox-rhs VALUES=1e-3
export REVISION_OUT="$revision_base/repairs"
bash scripts/run_sensitivity_revision.sh lambda
echo "Selected repairs saved; assemble the complete corrected baseline with resolve_revision_fixed_baseline.py."
