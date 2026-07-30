#!/usr/bin/env bash
set -euo pipefail
config="${1:-configs/arr_followup_v1.yaml}"
.venv/bin/python -m offline_evaluation.arr_followup.stochastic_repeat --config "$config" prepare
.venv/bin/python -m offline_evaluation.arr_followup.stochastic_repeat --config "$config" run
.venv/bin/python -m offline_evaluation.arr_followup.stochastic_repeat --config "$config" collect
