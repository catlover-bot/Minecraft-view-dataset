#!/usr/bin/env bash
set -euo pipefail
config="${1:-configs/fixed_builder_v1.yaml}"
.venv/bin/python -m offline_evaluation.fixed_builder.generate_gemini_descriptions --config "$config"
.venv/bin/python -m offline_evaluation.fixed_builder.collect_manifests --config "$config"
.venv/bin/python -m offline_evaluation.fixed_builder.prepare_inputs --config "$config"
.venv/bin/python -m offline_evaluation.fixed_builder.run_builder --config "$config" --representation free_form
.venv/bin/python -m offline_evaluation.fixed_builder.run_builder --config "$config" --representation structured
.venv/bin/python -m offline_evaluation.fixed_builder.collect_manifests --config "$config"
