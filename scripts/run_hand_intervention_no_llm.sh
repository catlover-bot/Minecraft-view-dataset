#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/opt/python@3.13/bin/python3.13}"
LIMIT="${LIMIT:-10}"
PORT="${PORT:-10000}"
MAX_RETRIES="${MAX_RETRIES:-4}"
BUILDING_PATTERN="${BUILDING_PATTERN:-building_*}"
CASES="${CASES:-v1_openai,v1_claude,v4_openai,v4_claude}"
VARIANTS="${VARIANTS:-baseline,twostage_off,overbuild_guard,underbuild_relax,material_reproject,mission_stable_exec}"
HAND_PLACE_MAX_PASSES="${HAND_PLACE_MAX_PASSES:-3}"
HAND_PLACE_TP_HEIGHT_OFFSET="${HAND_PLACE_TP_HEIGHT_OFFSET:-2.2}"
HAND_PLACE_USE_PULSE_SEC="${HAND_PLACE_USE_PULSE_SEC:-0.06}"
HAND_PLACE_HOTBAR_SLOT="${HAND_PLACE_HOTBAR_SLOT:-0}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_hand_intervention_no_llm.sh

Env overrides:
  LIMIT=10
  PORT=10000
  MAX_RETRIES=4
  BUILDING_PATTERN=building_*
  CASES=v1_openai,v1_claude,v4_openai,v4_claude
  VARIANTS=baseline,twostage_off,overbuild_guard,underbuild_relax,material_reproject,mission_stable_exec
  HAND_PLACE_MAX_PASSES=3
  HAND_PLACE_TP_HEIGHT_OFFSET=2.2
  HAND_PLACE_USE_PULSE_SEC=0.06
  HAND_PLACE_HOTBAR_SLOT=0

Notes:
  - This script does NOT call any LLM APIs.
  - It reuses existing renderer outputs and evaluates hand-placement execution only.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

limit_tag="$(printf "%03d" "$LIMIT")"
metrics_dir="reports/final/intervention_metrics_hand"
mkdir -p "$metrics_dir"

contains_csv() {
  local csv="$1"
  local item="$2"
  local IFS=','
  for x in $csv; do
    if [[ "$x" == "$item" ]]; then
      return 0
    fi
  done
  return 1
}

run_variant() {
  local dataset_root="$1"
  local gt_root="$2"
  local source_subdir="$3"
  local out_subdir="$4"
  local metrics_out="$5"

  scripts/recover_agentexec_variant.sh \
    --dataset_root "$dataset_root" \
    --gt_root "$gt_root" \
    --source_subdir "$source_subdir" \
    --out_subdir "$out_subdir" \
    --metrics_out "$metrics_out" \
    --port "$PORT" \
    --max_retries "$MAX_RETRIES" \
    --limit "$LIMIT" \
    --building_pattern "$BUILDING_PATTERN" \
    --placement_mode hand_place \
    --hand_place_max_passes "$HAND_PLACE_MAX_PASSES" \
    --hand_place_tp_height_offset "$HAND_PLACE_TP_HEIGHT_OFFSET" \
    --hand_place_use_pulse_sec "$HAND_PLACE_USE_PULSE_SEC" \
    --hand_place_hotbar_slot "$HAND_PLACE_HOTBAR_SLOT"
}

run_case() {
  local case_key="$1"
  local dataset_root="$2"
  local gt_root="$3"
  local baseline_renderer="$4"
  local short_tag="$5"

  echo "[run_hand_intervention_no_llm] case=$case_key"

  local variant
  local source_subdir
  local out_subdir
  local metrics_out
  for variant in baseline twostage_off overbuild_guard underbuild_relax material_reproject mission_stable_exec; do
    if ! contains_csv "$VARIANTS" "$variant"; then
      continue
    fi
    if [[ "$variant" == "baseline" || "$variant" == "mission_stable_exec" ]]; then
      source_subdir="$baseline_renderer"
    else
      source_subdir="rebuild_world_ab_${variant}_${short_tag}_l${limit_tag}"
    fi
    out_subdir="rebuild_world_agentexec_hand_ab_${variant}_${short_tag}_l${limit_tag}"
    metrics_out="${metrics_dir}/${case_key}.${variant}.agent.json"
    echo "[run_hand_intervention_no_llm]   variant=$variant"
    run_variant "$dataset_root" "$gt_root" "$source_subdir" "$out_subdir" "$metrics_out"
  done
}

echo "[run_hand_intervention_no_llm] limit=$LIMIT port=$PORT cases=$CASES variants=$VARIANTS"

if contains_csv "$CASES" "v1_openai"; then
  run_case \
    "v1_openai" \
    "outputs/i2t2b/buildings_100_v1" \
    "datasets/buildings_100_v1" \
    "rebuild_world_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_tuned" \
    "openai_gpt5mini"
fi

if contains_csv "$CASES" "v1_claude"; then
  run_case \
    "v1_claude" \
    "outputs/i2t2b/buildings_100_v1" \
    "datasets/buildings_100_v1" \
    "rebuild_world_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001_self_refine_no_gt_tuned" \
    "claude_haiku45"
fi

if contains_csv "$CASES" "v4_openai"; then
  run_case \
    "v4_openai" \
    "outputs/i2t2b/buildings_100_v4" \
    "datasets/buildings_100_v4" \
    "rebuild_world_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_tuned" \
    "openai_gpt5mini"
fi

if contains_csv "$CASES" "v4_claude"; then
  run_case \
    "v4_claude" \
    "outputs/i2t2b/buildings_100_v4" \
    "datasets/buildings_100_v4" \
    "rebuild_world_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001_self_refine_no_gt_tuned" \
    "claude_haiku45"
fi

echo "[run_hand_intervention_no_llm] done. metrics_dir=${metrics_dir}"
