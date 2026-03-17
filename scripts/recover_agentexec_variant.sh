#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/opt/python@3.13/bin/python3.13}"
PORT="${PORT:-10000}"
MAX_RETRIES="${MAX_RETRIES:-3}"
LIMIT="${LIMIT:-10}"
PLACEMENT_MODE="${PLACEMENT_MODE:-chat_commands}"
HAND_PLACE_MAX_PASSES="${HAND_PLACE_MAX_PASSES:-3}"
HAND_PLACE_TP_HEIGHT_OFFSET="${HAND_PLACE_TP_HEIGHT_OFFSET:-2.2}"
HAND_PLACE_USE_PULSE_SEC="${HAND_PLACE_USE_PULSE_SEC:-0.06}"
HAND_PLACE_HOTBAR_SLOT="${HAND_PLACE_HOTBAR_SLOT:-0}"

DATASET_ROOT=""
GT_ROOT=""
SOURCE_SUBDIR=""
OUT_SUBDIR=""
METRICS_OUT=""
BUILDING_PATTERN="building_*"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_root) DATASET_ROOT="$2"; shift 2 ;;
    --gt_root) GT_ROOT="$2"; shift 2 ;;
    --source_subdir) SOURCE_SUBDIR="$2"; shift 2 ;;
    --out_subdir) OUT_SUBDIR="$2"; shift 2 ;;
    --metrics_out) METRICS_OUT="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --max_retries) MAX_RETRIES="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --building_pattern) BUILDING_PATTERN="$2"; shift 2 ;;
    --placement_mode) PLACEMENT_MODE="$2"; shift 2 ;;
    --hand_place_max_passes) HAND_PLACE_MAX_PASSES="$2"; shift 2 ;;
    --hand_place_tp_height_offset) HAND_PLACE_TP_HEIGHT_OFFSET="$2"; shift 2 ;;
    --hand_place_use_pulse_sec) HAND_PLACE_USE_PULSE_SEC="$2"; shift 2 ;;
    --hand_place_hotbar_slot) HAND_PLACE_HOTBAR_SLOT="$2"; shift 2 ;;
    *)
      echo "[recover_agentexec_variant] unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$DATASET_ROOT" || -z "$GT_ROOT" || -z "$SOURCE_SUBDIR" || -z "$OUT_SUBDIR" || -z "$METRICS_OUT" ]]; then
  echo "[recover_agentexec_variant] missing required args." >&2
  echo "required: --dataset_root --gt_root --source_subdir --out_subdir --metrics_out" >&2
  exit 2
fi

if [[ "$PLACEMENT_MODE" != "chat_commands" && "$PLACEMENT_MODE" != "hand_place" ]]; then
  echo "[recover_agentexec_variant] invalid --placement_mode: $PLACEMENT_MODE (chat_commands|hand_place)" >&2
  exit 2
fi

restart_malmo() {
  local pids
  pids="$(lsof -tiTCP:${PORT} -sTCP:LISTEN || true)"
  if [[ -n "$pids" ]]; then
    # shellcheck disable=SC2086
    kill $pids || true
    sleep 1
  fi
  MALMO_DIR="${MALMO_DIR:-$ROOT_DIR/MalmoPlatform}" scripts/start_malmo_client_mac.sh --port "$PORT"
  scripts/wait_for_malmo_port.sh --port "$PORT" --timeout 240
}

BUILDINGS=()
while IFS= read -r path; do
  [[ -z "$path" ]] && continue
  BUILDINGS+=("$(basename "$path")")
done < <(find "$DATASET_ROOT" -maxdepth 1 -type d -name "$BUILDING_PATTERN" | sort | head -n "$LIMIT")
if [[ ${#BUILDINGS[@]} -eq 0 ]]; then
  echo "[recover_agentexec_variant] no buildings matched under $DATASET_ROOT pattern=$BUILDING_PATTERN" >&2
  exit 2
fi

restart_malmo

for b in "${BUILDINGS[@]}"; do
  ok=0
  for attempt in $(seq 1 "$MAX_RETRIES"); do
    echo "[recover_agentexec_variant] building=$b attempt=$attempt/$MAX_RETRIES"
    if "$PYTHON_BIN" tools/generate_agentexec_world_real.py \
      --dataset_root "$DATASET_ROOT" \
      --source_subdir "$SOURCE_SUBDIR" \
      --out_subdir "$OUT_SUBDIR" \
      --port "$PORT" \
      --building_pattern "$b" \
      --limit 1 \
      --placement_mode "$PLACEMENT_MODE" \
      --hand_place_max_passes "$HAND_PLACE_MAX_PASSES" \
      --hand_place_tp_height_offset "$HAND_PLACE_TP_HEIGHT_OFFSET" \
      --hand_place_use_pulse_sec "$HAND_PLACE_USE_PULSE_SEC" \
      --hand_place_hotbar_slot "$HAND_PLACE_HOTBAR_SLOT" \
      --overwrite; then
      ok=1
      break
    fi
    echo "[recover_agentexec_variant] retry after client restart: building=$b" >&2
    restart_malmo
  done
  if [[ "$ok" -ne 1 ]]; then
    echo "[recover_agentexec_variant] FAILED building=$b after $MAX_RETRIES attempts" >&2
    exit 3
  fi
  sleep 0.6
done

"$PYTHON_BIN" tools/evaluate_rebuild_metrics.py \
  --gt_root "$GT_ROOT" \
  --pred_root "$DATASET_ROOT" \
  --pred_source rebuild_world \
  --pred_subdir "$OUT_SUBDIR" \
  --out "$METRICS_OUT" \
  --building_pattern "$BUILDING_PATTERN" \
  --fail_on_missing_pred \
  --limit "$LIMIT" \
  --thresholds_json tools/thresholds_levels.example.json

echo "[recover_agentexec_variant] done source=$SOURCE_SUBDIR out=$OUT_SUBDIR mode=$PLACEMENT_MODE metrics=$METRICS_OUT"
