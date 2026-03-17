#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

LIMIT="${LIMIT:-10}"
PORT="${PORT:-10000}"
CASES="${CASES:-v1_openai,v1_claude,v4_openai,v4_claude}"
VARIANTS="${VARIANTS:-baseline,twostage_off,overbuild_guard,underbuild_relax,material_reproject,mission_stable_exec}"
WAIT_SEC="${WAIT_SEC:-60}"
LOG_FILE="${LOG_FILE:-logs/next_steps_after_hand_suite.log}"

mkdir -p "$(dirname "$LOG_FILE")"

suite_pattern="tools/run_execution_gap_suite.py --agentexec_mode hand --limit ${LIMIT}"

echo "[next-steps] start: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
echo "[next-steps] waiting pattern: $suite_pattern" | tee -a "$LOG_FILE"

while pgrep -f "$suite_pattern" >/dev/null 2>&1; do
  echo "[next-steps] hand suite still running... $(date -u +%H:%M:%S)" | tee -a "$LOG_FILE"
  sleep "$WAIT_SEC"
done

echo "[next-steps] hand suite finished. start intervention run." | tee -a "$LOG_FILE"

PLACEMENT_MODE=hand_place \
LIMIT="$LIMIT" \
PORT="$PORT" \
CASES="$CASES" \
VARIANTS="$VARIANTS" \
OUT_JSON="reports/final/intervention_ab_hand_limit${LIMIT}.json" \
OUT_MD="reports/final/intervention_ab_hand_limit${LIMIT}.md" \
scripts/run_real_failure_intervention_ab.sh \
  --overwrite_agentexec \
  --overwrite_variants | tee -a "$LOG_FILE"

echo "[next-steps] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
