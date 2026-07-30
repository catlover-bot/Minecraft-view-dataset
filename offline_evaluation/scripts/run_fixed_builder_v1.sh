#!/usr/bin/env bash
set -euo pipefail
config="${1:-configs/fixed_builder_v1.yaml}"
.venv/bin/python -m offline_evaluation.fixed_builder.freeze_state --config "$config"
.venv/bin/python -m offline_evaluation.fixed_builder.audit_prompts --config "$config"
bash offline_evaluation/scripts/run_fixed_builder_api.sh "$config"
bash offline_evaluation/scripts/run_fixed_builder_analysis.sh "$config"
