#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export MPLCONFIGDIR="${REPO_DIR}/simulations/snn_long_run/report/.mplconfig"
export PYTHONPATH="${REPO_DIR}"
python "${SCRIPT_DIR}/generate_snn_long_run_report.py"
