#!/usr/bin/env bash
set -euo pipefail
config="${1:-configs/arr_followup_v1.yaml}"
.venv/bin/python -m offline_evaluation.arr_followup.prepare_second_builder --config "$config"
echo "Dry-run only: inspect second_builder/manifest.csv before authorizing a separate provider/model experiment."
