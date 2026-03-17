#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3}"

LIMIT="${LIMIT:-10}"
PORT="${PORT:-10000}"
PLACEMENT_MODE="${PLACEMENT_MODE:-chat_commands}"
BUILDING_PATTERN="${BUILDING_PATTERN:-building_*}"
THRESHOLDS_JSON="${THRESHOLDS_JSON:-tools/thresholds_levels.example.json}"
CASES="${CASES:-v1_openai,v1_claude,v4_openai,v4_claude}"
VARIANTS="${VARIANTS:-baseline,twostage_off,overbuild_guard,underbuild_relax,material_reproject,mission_stable_exec}"
OUT_JSON="${OUT_JSON:-reports/final/intervention_ab_real_limit10.json}"
OUT_MD="${OUT_MD:-reports/final/intervention_ab_real_limit10.md}"

EXTRA_ARGS=("$@")

echo "[run_real_failure_intervention_ab.sh] python: $PYTHON_BIN"
echo "[run_real_failure_intervention_ab.sh] limit: $LIMIT port: $PORT placement_mode: $PLACEMENT_MODE"
echo "[run_real_failure_intervention_ab.sh] cases: $CASES"
echo "[run_real_failure_intervention_ab.sh] variants: $VARIANTS"

"$PYTHON_BIN" tools/run_real_failure_intervention_ab.py \
  --limit "$LIMIT" \
  --port "$PORT" \
  --placement_mode "$PLACEMENT_MODE" \
  --building_pattern "$BUILDING_PATTERN" \
  --thresholds_json "$THRESHOLDS_JSON" \
  --cases "$CASES" \
  --variants "$VARIANTS" \
  --out_json "$OUT_JSON" \
  --out_md "$OUT_MD" \
  "${EXTRA_ARGS[@]}"
