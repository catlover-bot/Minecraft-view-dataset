#!/usr/bin/env bash
set -euo pipefail
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-offline}"
PY=".venv/bin/python"
CFG="configs/offline_eval.yaml"

"$PY" -m offline_evaluation.audit --repo-root .
"$PY" -m offline_evaluation.validate_executor_v2 --config "$CFG"
"$PY" -m offline_evaluation.data_coverage --config "$CFG"
"$PY" -m offline_evaluation.reconcile_metrics --config "$CFG"
"$PY" -m offline_evaluation.analyze_materials_v2 --config "$CFG"
"$PY" -m offline_evaluation.analyze_correlations_v2 --config "$CFG" --bootstrap-samples 10000
"$PY" -m offline_evaluation.analyze_alignment_v2 --config "$CFG" --sensitivity-scenes 50
"$PY" -m offline_evaluation.structured_ir_recovery.prepare --config "$CFG"
"$PY" -m offline_evaluation.structured_ir_recovery.prepare --config "$CFG" --execute-offline
"$PY" -m offline_evaluation.analyze_paired_ir_v2 --config "$CFG" --bootstrap-samples 10000
"$PY" -m offline_evaluation.fixed_builder_v2.prepare --config "$CFG"
"$PY" -m offline_evaluation.generate_arr_v2 --config "$CFG"
"$PY" -m pytest -q offline_evaluation/tests
