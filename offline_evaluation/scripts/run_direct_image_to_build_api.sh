#!/usr/bin/env bash
set -euo pipefail
config="${1:-configs/arr_followup_v1.yaml}"
.venv/bin/python -m offline_evaluation.arr_followup.direct_image_to_build --config "$config" prepare
.venv/bin/python -m offline_evaluation.arr_followup.direct_image_to_build --config "$config" run
.venv/bin/python -m offline_evaluation.arr_followup.direct_image_to_build --config "$config" collect
