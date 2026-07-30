#!/usr/bin/env bash
set -euo pipefail
config="${1:-configs/fixed_builder_v1.yaml}"
.venv/bin/python -m offline_evaluation.fixed_builder.evaluate --config "$config"
.venv/bin/python -m offline_evaluation.fixed_builder.score_descriptions --config "$config"
.venv/bin/python -m offline_evaluation.fixed_builder.analyze_correlations --config "$config"
.venv/bin/python -m offline_evaluation.fixed_builder.analyze_description_models --config "$config"
.venv/bin/python -m offline_evaluation.fixed_builder.analyze --config "$config"
.venv/bin/python -m offline_evaluation.fixed_builder.generate_paper_outputs --config "$config"
.venv/bin/python -m offline_evaluation.fixed_builder.api_usage --config "$config"
.venv/bin/python -m offline_evaluation.fixed_builder.generate_arr_claims --config "$config"
