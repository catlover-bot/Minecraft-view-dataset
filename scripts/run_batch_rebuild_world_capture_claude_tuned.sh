#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/malmo_env_mac.sh"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [[ -x /opt/homebrew/bin/python3 ]]; then
  PYTHON_BIN="/opt/homebrew/bin/python3"
else
  PYTHON_BIN="$(command -v python3)"
fi

PORT=10000
VIEWS=8
IMAGE_W=960
IMAGE_H=540
FOV=70
WAIT_TIMEOUT=240
START_INDEX=0
LIMIT=0
MAX_RETRIES=2

usage() {
  cat <<'EOF'
Usage:
  scripts/run_batch_rebuild_world_capture_claude_tuned.sh \
    [--port PORT] \
    [--views N] \
    [--image_size W H] \
    [--fov FOV] \
    [--wait_timeout SEC] \
    [--start_index N] \
    [--limit N] \
    [--retries N]

Notes:
  - Targets both outputs/i2t2b/buildings_100_v1 and outputs/i2t2b/buildings_100_v4
  - Captures only missing/incomplete Claude tuned rebuild-world outputs.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    --views)
      VIEWS="$2"
      shift 2
      ;;
    --image_size)
      IMAGE_W="$2"
      IMAGE_H="$3"
      shift 3
      ;;
    --fov)
      FOV="$2"
      shift 2
      ;;
    --wait_timeout)
      WAIT_TIMEOUT="$2"
      shift 2
      ;;
    --start_index)
      START_INDEX="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --retries)
      MAX_RETRIES="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[run_batch_rebuild_world_capture_claude_tuned] Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

BASE_DIRS=(
  "${ROOT_DIR}/outputs/i2t2b/buildings_100_v1"
  "${ROOT_DIR}/outputs/i2t2b/buildings_100_v4"
)

MODEL_SUFFIX="schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001_self_refine_no_gt_tuned"

build_queue() {
  local tmp="$1"
  : > "${tmp}"
  local base
  for base in "${BASE_DIRS[@]}"; do
    [[ -d "${base}" ]] || continue
    local b
    for b in "${base}"/building_*; do
      [[ -d "${b}" ]] || continue
      local rw="${b}/rebuild_world_${MODEL_SUFFIX}"
      local cap="${b}/capture_rebuild_world_${MODEL_SUFFIX}"
      [[ -d "${rw}" ]] || continue
      if [[ -f "${cap}/meta.json" ]] && [[ -d "${cap}/images" ]]; then
        local png_count
        png_count="$(find "${cap}/images" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' ')"
        if [[ "${png_count}" -ge "${VIEWS}" ]]; then
          continue
        fi
      fi
      echo "${rw}" >> "${tmp}"
    done
  done
  sort "${tmp}" -o "${tmp}"
}

restart_client() {
  local pid_file="${ROOT_DIR}/logs/malmo_client.pid"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
      sleep 1
      if kill -0 "${pid}" >/dev/null 2>&1; then
        kill -9 "${pid}" >/dev/null 2>&1 || true
      fi
    fi
    rm -f "${pid_file}"
  fi
  if command -v lsof >/dev/null 2>&1; then
    local listen_pids
    listen_pids="$(lsof -tiTCP:${PORT} -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "${listen_pids}" ]]; then
      for pid in ${listen_pids}; do
        kill -9 "${pid}" >/dev/null 2>&1 || true
      done
      sleep 1
    fi
  fi
}

cd "${ROOT_DIR}"
mkdir -p logs
QUEUE_FILE="logs/claude_rebuild_capture_queue.txt"

build_queue "${QUEUE_FILE}"
TOTAL="$(wc -l < "${QUEUE_FILE}" | tr -d ' ')"
echo "[run_batch_rebuild_world_capture_claude_tuned] pending targets: ${TOTAL}"
echo "[run_batch_rebuild_world_capture_claude_tuned] queue file: ${QUEUE_FILE}"

if [[ "${TOTAL}" -eq 0 ]]; then
  echo "[run_batch_rebuild_world_capture_claude_tuned] nothing to do."
  exit 0
fi

done_count=0
fail_count=0
seen=0

while IFS= read -r rw_dir; do
  [[ -n "${rw_dir}" ]] || continue
  seen=$((seen + 1))
  if [[ "${seen}" -le "${START_INDEX}" ]]; then
    continue
  fi
  if [[ "${LIMIT}" -gt 0 ]] && [[ "${done_count}" -ge "${LIMIT}" ]]; then
    break
  fi

  echo
  echo "[run_batch_rebuild_world_capture_claude_tuned] (${seen}/${TOTAL}) target=${rw_dir}"
  ok=0
  for ((attempt=1; attempt<=MAX_RETRIES; attempt++)); do
    echo "[run_batch_rebuild_world_capture_claude_tuned] attempt=${attempt}/${MAX_RETRIES}"
    if "${SCRIPT_DIR}/run_rebuild_world_capture.sh" \
        --rebuild_world_dir "${rw_dir}" \
        --port "${PORT}" \
        --views "${VIEWS}" \
        --image_size "${IMAGE_W}" "${IMAGE_H}" \
        --fov "${FOV}" \
        --wait_timeout "${WAIT_TIMEOUT}"; then
      ok=1
      break
    fi
    echo "[run_batch_rebuild_world_capture_claude_tuned] capture failed on attempt=${attempt}; restarting client."
    restart_client
  done

  if [[ "${ok}" -eq 1 ]]; then
    done_count=$((done_count + 1))
  else
    fail_count=$((fail_count + 1))
    echo "[run_batch_rebuild_world_capture_claude_tuned] failed: ${rw_dir}"
  fi
  echo "[run_batch_rebuild_world_capture_claude_tuned] done=${done_count} fail=${fail_count}"
done < "${QUEUE_FILE}"

echo
echo "[run_batch_rebuild_world_capture_claude_tuned] finished."
echo "[run_batch_rebuild_world_capture_claude_tuned] done=${done_count} fail=${fail_count}"
