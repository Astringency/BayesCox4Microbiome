#!/usr/bin/env bash
# Paired reference/information-calibrated Cox-RHS fits on identical training seeds.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_sensitivity_revision.sh" scale "$@"
