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
    [--provider openai|anthropic|gemini|mock] \
    [--dotenv .env] \
    [--output_tag my_tag] \
    [--no_split_by_model] \
    [--limit 10] \
    [--overwrite] \
    [--no_relocate_outputs] \
    [--relocate_out_root outputs/i2t2b]
EOF
  exit 1
fi

DATASET_ROOT=""
RELOCATE_OUTPUTS=1
RELOCATE_OUT_ROOT="outputs/i2t2b"
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_root)
      DATASET_ROOT="${2:-}"
      PASS_ARGS+=("$1" "${2:-}")
      shift 2
      ;;
    --no_relocate_outputs)
      RELOCATE_OUTPUTS=0
      shift
      ;;
    --relocate_out_root)
      RELOCATE_OUT_ROOT="${2:-}"
      shift 2
      ;;
    *)
      PASS_ARGS+=("$1")
      shift
      ;;
  esac
done

cd "${ROOT_DIR}"
"${PYTHON_BIN}" "${ROOT_DIR}/tools/run_i2t2b_experiment.py" "${PASS_ARGS[@]}"

if [[ "${RELOCATE_OUTPUTS}" -eq 1 && -n "${DATASET_ROOT}" ]]; then
  "${ROOT_DIR}/scripts/relocate_i2t2b_outputs.sh" \
    --dataset_root "${DATASET_ROOT}" \
    --out_root "${RELOCATE_OUT_ROOT}"
fi
