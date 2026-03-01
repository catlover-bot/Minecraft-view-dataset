#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [[ -x /opt/homebrew/bin/python3 ]]; then
  PYTHON_BIN="/opt/homebrew/bin/python3"
else
  PYTHON_BIN="$(command -v python3)"
fi

if [[ $# -eq 0 ]]; then
  cat <<'EOF'
Usage:
  scripts/run_i2t2b_experiment.sh \
    --dataset_root datasets/buildings_100_v1 \
    [--provider openai|anthropic|mock] \
    [--dotenv .env] \
    [--output_tag my_tag] \
    [--no_split_by_model] \
    [--limit 10] \
    [--overwrite]
EOF
  exit 1
fi

cd "${ROOT_DIR}"
"${PYTHON_BIN}" "${ROOT_DIR}/tools/run_i2t2b_experiment.py" "$@"
