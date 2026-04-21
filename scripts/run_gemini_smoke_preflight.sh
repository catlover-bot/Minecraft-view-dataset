#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [[ -x /opt/homebrew/bin/python3 ]]; then
  PYTHON_BIN="/opt/homebrew/bin/python3"
else
  PYTHON_BIN="$(command -v python3)"
fi

LIMIT="${LIMIT:-10}"
BUILDING_PATTERN="${BUILDING_PATTERN:-building_*}"
DOTENV_PATH="${DOTENV_PATH:-.env}"
DATASETS="${DATASETS:-buildings_100_v1,buildings_100_v4}"
OUT_ROOT="${OUT_ROOT:-outputs/i2t2b}"
THRESHOLDS_JSON="${THRESHOLDS_JSON:-tools/thresholds_levels.example.json}"
GEMINI_MODEL_TAG="${GEMINI_MODEL_TAG:-}"
OVERWRITE="${OVERWRITE:-0}"

MAX_EMPTY_OPS_RATE="${MAX_EMPTY_OPS_RATE:-0.10}"
MAX_FALLBACK_RATE="${MAX_FALLBACK_RATE:-0.60}"
MAX_STRICT_BLOCKING_RATE="${MAX_STRICT_BLOCKING_RATE:-0.60}"
MIN_IOU="${MIN_IOU:-0.18}"
MIN_F1="${MIN_F1:-0.30}"
MIN_MATERIAL_MATCH="${MIN_MATERIAL_MATCH:-0.10}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)
      LIMIT="${2:-10}"
      shift 2
      ;;
    --building_pattern)
      BUILDING_PATTERN="${2:-building_*}"
      shift 2
      ;;
    --dotenv)
      DOTENV_PATH="${2:-.env}"
      shift 2
      ;;
    --datasets)
      DATASETS="${2:-buildings_100_v1,buildings_100_v4}"
      shift 2
      ;;
    --out_root)
      OUT_ROOT="${2:-outputs/i2t2b}"
      shift 2
      ;;
    --thresholds_json)
      THRESHOLDS_JSON="${2:-tools/thresholds_levels.example.json}"
      shift 2
      ;;
    --gemini_model_tag)
      GEMINI_MODEL_TAG="${2:-}"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --max_empty_ops_rate)
      MAX_EMPTY_OPS_RATE="${2:-0.10}"
      shift 2
      ;;
    --max_fallback_rate)
      MAX_FALLBACK_RATE="${2:-0.60}"
      shift 2
      ;;
    --max_strict_blocking_rate)
      MAX_STRICT_BLOCKING_RATE="${2:-0.60}"
      shift 2
      ;;
    --min_iou)
      MIN_IOU="${2:-0.18}"
      shift 2
      ;;
    --min_f1)
      MIN_F1="${2:-0.30}"
      shift 2
      ;;
    --min_material_match)
      MIN_MATERIAL_MATCH="${2:-0.10}"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  scripts/run_gemini_smoke_preflight.sh [options]

Options:
  --limit 10
  --datasets buildings_100_v1,buildings_100_v4
  --dotenv .env
  --gemini_model_tag gemini_gemini_3_1_pro_preview
  --overwrite

Gate thresholds:
  --max_empty_ops_rate 0.10
  --max_fallback_rate 0.60
  --max_strict_blocking_rate 0.60
  --min_iou 0.18
  --min_f1 0.30
  --min_material_match 0.10
EOF
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${GEMINI_MODEL_TAG}" ]]; then
  GEMINI_MODEL_TAG="$(
    DOTENV_PATH="${DOTENV_PATH}" "${PYTHON_BIN}" - <<'PY'
import os
import re
from tools.llm_config import load_llm_config

def slugify(s: str) -> str:
    x = re.sub(r"[^a-z0-9]+", "_", (s or "").strip().lower())
    x = re.sub(r"_+", "_", x).strip("_")
    return x or "unknown"

cfg = load_llm_config(os.environ.get("DOTENV_PATH") or None)
print(f"gemini_{slugify(cfg.gemini_model or 'gemini_model')}")
PY
  )"
fi

LIMIT_PAD="$(printf '%03d' "${LIMIT}")"
REPORT_DIR="reports/final/gemini_preflight"
mkdir -p "${REPORT_DIR}"

echo "[gemini-preflight] python: ${PYTHON_BIN}"
echo "[gemini-preflight] gemini_model_tag: ${GEMINI_MODEL_TAG}"
echo "[gemini-preflight] datasets: ${DATASETS}"
echo "[gemini-preflight] limit: ${LIMIT}"

IFS=',' read -r -a DATASET_ARR <<< "${DATASETS}"
for dataset in "${DATASET_ARR[@]}"; do
  dataset="$(echo "${dataset}" | xargs)"
  [[ -z "${dataset}" ]] && continue
  dataset_root="datasets/${dataset}"
  pred_root="${OUT_ROOT}/${dataset}"

  if [[ ! -d "${dataset_root}" ]]; then
    echo "[gemini-preflight] dataset not found: ${dataset_root}" >&2
    exit 1
  fi

  description_subdir="description_${GEMINI_MODEL_TAG}"
  plan_subdir="rebuild_plan_schema_material_v5_repair_${GEMINI_MODEL_TAG}"
  refined_plan_subdir="${plan_subdir}_self_refine_no_gt_tuned"
  rebuild_subdir="rebuild_world_schema_material_v5_repair_${GEMINI_MODEL_TAG}_self_refine_no_gt_tuned"
  desc_metrics_rel="metrics/description/description_metrics_${GEMINI_MODEL_TAG}_smoke_l${LIMIT_PAD}.json"
  rebuild_metrics_rel="metrics/rebuild/metrics_levels_${GEMINI_MODEL_TAG}_smoke_l${LIMIT_PAD}.json"

  echo "[gemini-preflight] === ${dataset} ==="
  cmd=(
    "${PYTHON_BIN}" "tools/run_i2t2b_experiment.py"
    "--dataset_root" "${dataset_root}"
    "--provider" "gemini"
    "--dotenv" "${DOTENV_PATH}"
    "--description_subdir" "${description_subdir}"
    "--plan_subdir" "${plan_subdir}"
    "--rebuild_subdir" "${rebuild_subdir}"
    "--enable_self_refine_no_gt"
    "--self_refine_plan_subdir" "${refined_plan_subdir}"
    "--plan_prompt_profile" "prompts/rebuild_plan_strict_material_v3.json"
    "--plan_critic_revise"
    "--plan_strict_schema"
    "--plan_enforce_role_fixed"
    "--plan_require_material_budget"
    "--plan_material_budget_tolerance" "0.35"
    "--plan_role_fix_min_confidence" "0.78"
    "--plan_prefer_description_palette"
    "--plan_max_operations" "260"
    "--building_pattern" "${BUILDING_PATTERN}"
    "--limit" "${LIMIT}"
    "--thresholds_json" "${THRESHOLDS_JSON}"
    "--description_metrics_out" "${desc_metrics_rel}"
    "--rebuild_metrics_out" "${rebuild_metrics_rel}"
  )
  if [[ "${OVERWRITE}" == "1" ]]; then
    cmd+=("--overwrite")
  fi
  "${cmd[@]}"

  "${ROOT_DIR}/scripts/relocate_i2t2b_outputs.sh" \
    --dataset_root "${dataset_root}" \
    --out_root "${OUT_ROOT}"

  gate_out="${REPORT_DIR}/${dataset}_gate_${GEMINI_MODEL_TAG}_l${LIMIT_PAD}.json"
  "${PYTHON_BIN}" tools/check_i2t2b_smoke_gate.py \
    --gt_root "${dataset_root}" \
    --pred_root "${pred_root}" \
    --description_subdir "${description_subdir}" \
    --plan_subdir "${plan_subdir}" \
    --rebuild_subdir "${rebuild_subdir}" \
    --metrics_json "${pred_root}/${rebuild_metrics_rel}" \
    --building_pattern "${BUILDING_PATTERN}" \
    --limit "${LIMIT}" \
    --max_missing_description_rate 0.00 \
    --max_missing_plan_rate 0.00 \
    --max_missing_rebuild_rate 0.00 \
    --max_empty_operations_rate "${MAX_EMPTY_OPS_RATE}" \
    --max_fallback_rate "${MAX_FALLBACK_RATE}" \
    --max_strict_blocking_rate "${MAX_STRICT_BLOCKING_RATE}" \
    --min_iou "${MIN_IOU}" \
    --min_f1 "${MIN_F1}" \
    --min_material_match "${MIN_MATERIAL_MATCH}" \
    --out_json "${gate_out}"

  echo "[gemini-preflight] gate passed: ${gate_out}"
done

echo "[gemini-preflight] done."
