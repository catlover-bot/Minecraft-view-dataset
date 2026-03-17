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

DATASET_ROOT=""
PRED_ROOT=""
RENDERER_SUBDIR=""
AGENT_SUBDIR=""
OUT=""
THRESHOLDS_JSON="${ROOT_DIR}/tools/thresholds_levels.example.json"
LIMIT=0
BUILDING_PATTERN="building_*"
FAIL_ON_MISSING_RENDERER=1
FAIL_ON_MISSING_AGENT=1

usage() {
  cat <<'EOF'
Usage:
  scripts/evaluate_renderer_agent_gap.sh \
    --dataset_root datasets/buildings_100_v1 \
    --pred_root outputs/i2t2b/buildings_100_v1 \
    --renderer_subdir rebuild_world_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_tuned \
    --agent_subdir rebuild_world_agentexec_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_tuned \
    [--out outputs/i2t2b/buildings_100_v1/metrics/rebuild/execution_gap_openai_tuned.json] \
    [--thresholds_json tools/thresholds_levels.example.json] \
    [--limit 10] \
    [--building_pattern building_*] \
    [--allow_missing_renderer_pred] \
    [--allow_missing_agent_pred]

Notes:
  - renderer: 理想側（上限）スコアを出す予測サブディレクトリ
  - agent: 実エージェント建築結果の予測サブディレクトリ
  - execution_gap = renderer - agent
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --pred_root)
      PRED_ROOT="$2"
      shift 2
      ;;
    --renderer_subdir)
      RENDERER_SUBDIR="$2"
      shift 2
      ;;
    --agent_subdir)
      AGENT_SUBDIR="$2"
      shift 2
      ;;
    --out)
      OUT="$2"
      shift 2
      ;;
    --thresholds_json)
      THRESHOLDS_JSON="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --building_pattern)
      BUILDING_PATTERN="$2"
      shift 2
      ;;
    --allow_missing_renderer_pred)
      FAIL_ON_MISSING_RENDERER=0
      shift
      ;;
    --allow_missing_agent_pred)
      FAIL_ON_MISSING_AGENT=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[evaluate_renderer_agent_gap] Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${DATASET_ROOT}" || -z "${RENDERER_SUBDIR}" || -z "${AGENT_SUBDIR}" ]]; then
  echo "[evaluate_renderer_agent_gap] required: --dataset_root --renderer_subdir --agent_subdir" >&2
  usage
  exit 1
fi

DATASET_ROOT_ABS="$("${PYTHON_BIN}" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${DATASET_ROOT}")"
if [[ ! -d "${DATASET_ROOT_ABS}" ]]; then
  echo "[evaluate_renderer_agent_gap] dataset_root not found: ${DATASET_ROOT_ABS}" >&2
  exit 1
fi

if [[ -z "${PRED_ROOT}" ]]; then
  dataset_name="$(basename "${DATASET_ROOT_ABS}")"
  PRED_ROOT="${ROOT_DIR}/outputs/i2t2b/${dataset_name}"
fi
PRED_ROOT_ABS="$("${PYTHON_BIN}" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${PRED_ROOT}")"

if [[ -z "${OUT}" ]]; then
  OUT="${PRED_ROOT_ABS}/metrics/rebuild/execution_gap.json"
fi
OUT_ABS="$("${PYTHON_BIN}" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${OUT}")"

mkdir -p "$(dirname "${OUT_ABS}")"

CMD=(
  "${PYTHON_BIN}" "${ROOT_DIR}/tools/evaluate_execution_gap.py"
  "--gt_root" "${DATASET_ROOT_ABS}"
  "--pred_root" "${PRED_ROOT_ABS}"
  "--renderer_pred_subdir" "${RENDERER_SUBDIR}"
  "--agent_pred_subdir" "${AGENT_SUBDIR}"
  "--out" "${OUT_ABS}"
  "--building_pattern" "${BUILDING_PATTERN}"
)

if [[ -n "${THRESHOLDS_JSON}" ]]; then
  CMD+=("--thresholds_json" "${THRESHOLDS_JSON}")
fi
if [[ "${LIMIT}" -gt 0 ]]; then
  CMD+=("--limit" "${LIMIT}")
fi
if [[ "${FAIL_ON_MISSING_RENDERER}" -eq 0 ]]; then
  CMD+=("--no_fail_on_missing_renderer_pred")
fi
if [[ "${FAIL_ON_MISSING_AGENT}" -eq 0 ]]; then
  CMD+=("--no_fail_on_missing_agent_pred")
fi

echo "[evaluate_renderer_agent_gap] dataset_root: ${DATASET_ROOT_ABS}"
echo "[evaluate_renderer_agent_gap] pred_root: ${PRED_ROOT_ABS}"
echo "[evaluate_renderer_agent_gap] renderer_subdir: ${RENDERER_SUBDIR}"
echo "[evaluate_renderer_agent_gap] agent_subdir: ${AGENT_SUBDIR}"
echo "[evaluate_renderer_agent_gap] out: ${OUT_ABS}"

"${CMD[@]}"

echo "[evaluate_renderer_agent_gap] done."
