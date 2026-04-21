#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_ROOT="${ROOT_DIR}/outputs/i2t2b/buildings_100_v1"
PORT=10000
VIEWS=8
IMAGE_W=960
IMAGE_H=540
FOV=70
WAIT_TIMEOUT=240
START_INDEX=0
LIMIT=0
RETRIES=2
MODE="all" # direct|structured|all

usage() {
  cat <<'EOF'
Usage:
  scripts/overwrite_gemini_v1_captures_with_minecraft.sh \
    [--dataset_root outputs/i2t2b/buildings_100_v1] \
    [--mode direct|structured|all] \
    [--port 10000] \
    [--views 8] \
    [--image_size 960 540] \
    [--fov 70] \
    [--wait_timeout 240] \
    [--start_index 0] \
    [--limit 0] \
    [--retries 2]

Notes:
  - Existing capture directories are removed and recreated (overwrite).
  - This script targets Gemini v1 outputs only.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_root) DATASET_ROOT="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --views) VIEWS="$2"; shift 2 ;;
    --image_size) IMAGE_W="$2"; IMAGE_H="$3"; shift 3 ;;
    --fov) FOV="$2"; shift 2 ;;
    --wait_timeout) WAIT_TIMEOUT="$2"; shift 2 ;;
    --start_index) START_INDEX="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --retries) RETRIES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[overwrite_gemini_v1] unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "${MODE}" != "direct" && "${MODE}" != "structured" && "${MODE}" != "all" ]]; then
  echo "[overwrite_gemini_v1] --mode must be direct|structured|all" >&2
  exit 1
fi

if [[ -z "${MALMO_DIR:-}" ]]; then
  if [[ -d "${ROOT_DIR}/MalmoPlatform" ]]; then
    export MALMO_DIR="${ROOT_DIR}/MalmoPlatform"
  else
    echo "[overwrite_gemini_v1] MALMO_DIR is not set and ${ROOT_DIR}/MalmoPlatform not found." >&2
    exit 1
  fi
fi

if [[ -z "${JAVA_HOME:-}" ]]; then
  export JAVA_HOME="$(/usr/libexec/java_home -v 1.8)"
fi

DIRECT_RW="rebuild_world_schema_material_v5_repair_gemini_gemini_3_1_pro_preview_common_v8_struct_self_refine_no_gt_tuned"
DIRECT_CAP="capture_rebuild_world_schema_material_v5_repair_gemini_gemini_3_1_pro_preview_common_v8_struct_self_refine_no_gt_tuned"
STRUCT_RW="rebuild_world_structured_ir_gemini_main_orig_20260419"
STRUCT_CAP="capture_rebuild_world_structured_ir_gemini_main_orig_20260419"

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

build_queue() {
  local queue="$1"
  : > "${queue}"
  local b
  for b in "${DATASET_ROOT}"/building_*; do
    [[ -d "${b}" ]] || continue
    if [[ "${MODE}" == "direct" || "${MODE}" == "all" ]]; then
      local rw="${b}/${DIRECT_RW}"
      local cap="${b}/${DIRECT_CAP}"
      [[ -d "${rw}" ]] && echo "${rw}|${cap}" >> "${queue}"
    fi
    if [[ "${MODE}" == "structured" || "${MODE}" == "all" ]]; then
      local rw2="${b}/${STRUCT_RW}"
      local cap2="${b}/${STRUCT_CAP}"
      [[ -d "${rw2}" ]] && echo "${rw2}|${cap2}" >> "${queue}"
    fi
  done
  sort "${queue}" -o "${queue}"
}

cd "${ROOT_DIR}"
mkdir -p logs
QUEUE_FILE="logs/gemini_v1_minecraft_capture_overwrite_queue.txt"
build_queue "${QUEUE_FILE}"
TOTAL="$(wc -l < "${QUEUE_FILE}" | tr -d ' ')"

echo "[overwrite_gemini_v1] dataset_root=${DATASET_ROOT}"
echo "[overwrite_gemini_v1] mode=${MODE} total_targets=${TOTAL}"
echo "[overwrite_gemini_v1] queue=${QUEUE_FILE}"

if [[ "${TOTAL}" -eq 0 ]]; then
  echo "[overwrite_gemini_v1] nothing to do."
  exit 0
fi

done_count=0
fail_count=0
seen=0

while IFS='|' read -r rw cap; do
  [[ -n "${rw}" ]] || continue
  seen=$((seen + 1))
  if [[ "${seen}" -le "${START_INDEX}" ]]; then
    continue
  fi
  if [[ "${LIMIT}" -gt 0 ]] && [[ "${done_count}" -ge "${LIMIT}" ]]; then
    break
  fi

  echo
  echo "[overwrite_gemini_v1] (${seen}/${TOTAL}) target=${rw}"

  rm -rf "${cap}"

  ok=0
  for ((attempt=1; attempt<=RETRIES; attempt++)); do
    echo "[overwrite_gemini_v1] attempt=${attempt}/${RETRIES}"
    if MALMO_DIR="${MALMO_DIR}" JAVA_HOME="${JAVA_HOME}" \
      "${SCRIPT_DIR}/run_rebuild_world_capture.sh" \
        --rebuild_world_dir "${rw}" \
        --out "${cap}" \
        --port "${PORT}" \
        --views "${VIEWS}" \
        --image_size "${IMAGE_W}" "${IMAGE_H}" \
        --fov "${FOV}" \
        --wait_timeout "${WAIT_TIMEOUT}"; then
      ok=1
      break
    fi
    echo "[overwrite_gemini_v1] capture failed. restarting Malmo client."
    restart_client
  done

  if [[ "${ok}" -eq 1 ]]; then
    done_count=$((done_count + 1))
  else
    fail_count=$((fail_count + 1))
    echo "[overwrite_gemini_v1] failed: ${rw}"
  fi
  echo "[overwrite_gemini_v1] done=${done_count} fail=${fail_count}"
done < "${QUEUE_FILE}"

echo
echo "[overwrite_gemini_v1] finished done=${done_count} fail=${fail_count}"
