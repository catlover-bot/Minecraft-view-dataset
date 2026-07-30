#!/usr/bin/env bash
set -euo pipefail
config="${1:-configs/arr_followup_v1.yaml}"
.venv/bin/python -m offline_evaluation.arr_followup.analyze_existing --config "$config"
.venv/bin/python -m offline_evaluation.arr_followup.audit_and_grounding --config "$config"
.venv/bin/python -m offline_evaluation.arr_followup.direct_image_to_build --config "$config" collect
.venv/bin/python -m offline_evaluation.arr_followup.direct_image_to_build --config "$config" evaluate
.venv/bin/python -m offline_evaluation.arr_followup.stochastic_repeat --config "$config" collect
.venv/bin/python -m offline_evaluation.arr_followup.stochastic_repeat --config "$config" evaluate
MPLCONFIGDIR=/tmp/mpl-arr-followup .venv/bin/python -m offline_evaluation.arr_followup.generate_outputs --config "$config"
