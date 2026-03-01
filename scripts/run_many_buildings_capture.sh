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

OUT="datasets/buildings_100_v1"
PORT=10000
COUNT=100
START_STYLE_ID=0
VIEWS=12
IMAGE_W=960
IMAGE_H=540
FOV=70
SEED=1234
SPACING=180
WAIT_TIMEOUT=420
MAX_RETRIES=3
PROFILE="complex"
SKIP_EXISTING=""

usage() {
  cat <<'EOF'
Usage:
  scripts/run_many_buildings_capture.sh \
    [--out OUT_DIR] \
    [--port PORT] \
    [--count N] \
    [--start_style_id N] \
    [--views N] \
    [--image_size W H] \
    [--fov FOV] \
    [--seed SEED] \
    [--profile complex|simple] \
    [--spacing BLOCKS] \
    [--retries N] \
    [--skip_existing] \
    [--wait_timeout SEC]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --count)
      COUNT="$2"
      shift 2
      ;;
    --start_style_id)
      START_STYLE_ID="$2"
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
    --seed)
      SEED="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --spacing)
      SPACING="$2"
      shift 2
      ;;
    --retries)
      MAX_RETRIES="$2"
      shift 2
      ;;
    --skip_existing)
      SKIP_EXISTING="1"
      shift
      ;;
    --wait_timeout)
      WAIT_TIMEOUT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[run_many_buildings_capture] Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

is_listening() {
  local ok=1
  if command -v nc >/dev/null 2>&1; then
    nc -z 127.0.0.1 "${PORT}" >/dev/null 2>&1 && ok=0
    if [[ "${ok}" -ne 0 ]]; then
      nc -z ::1 "${PORT}" >/dev/null 2>&1 && ok=0
    fi
  fi
  if [[ "${ok}" -ne 0 ]] && command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1 && ok=0
  fi
  return "${ok}"
}

cd "${ROOT_DIR}"
mkdir -p logs

stop_malmo_client_if_running() {
  local pid_file="${ROOT_DIR}/logs/malmo_client.pid"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
      sleep 1
    fi
    rm -f "${pid_file}"
  fi
  if ! command -v lsof >/dev/null 2>&1; then
    return 0
  fi
  for _ in 1 2 3 4; do
    local pids
    pids="$(lsof -tiTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -z "${pids}" ]]; then
      return 0
    fi
    for pid in ${pids}; do
      kill "${pid}" >/dev/null 2>&1 || true
    done
    sleep 1
  done
  local stubborn
  stubborn="$(lsof -tiTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "${stubborn}" ]]; then
    for pid in ${stubborn}; do
      kill -9 "${pid}" >/dev/null 2>&1 || true
    done
    sleep 1
  fi
  if is_listening; then
    echo "[run_many_buildings_capture] Failed to stop existing listener on :${PORT}." >&2
    return 1
  fi
  return 0
}

ensure_client_ready() {
  if ! is_listening; then
    echo "[run_many_buildings_capture] Malmo client is not listening on :${PORT}. Launching client..."
    "${SCRIPT_DIR}/start_malmo_client_mac.sh" --port "${PORT}"
  else
    echo "[run_many_buildings_capture] :${PORT} is already LISTEN."
  fi
  if ! "${SCRIPT_DIR}/wait_for_malmo_port.sh" --host 127.0.0.1 --port "${PORT}" --timeout "${WAIT_TIMEOUT}"; then
    echo "[run_many_buildings_capture] Failed waiting for Malmo port." >&2
    echo "[run_many_buildings_capture] Check logs: ${ROOT_DIR}/logs/malmo_client.log" >&2
    return 1
  fi
  return 0
}

restart_client() {
  echo "[run_many_buildings_capture] Restarting Malmo client on :${PORT} ..."
  if ! stop_malmo_client_if_running; then
    return 1
  fi
  "${SCRIPT_DIR}/start_malmo_client_mac.sh" --port "${PORT}"
  if ! "${SCRIPT_DIR}/wait_for_malmo_port.sh" --host 127.0.0.1 --port "${PORT}" --timeout "${WAIT_TIMEOUT}"; then
    echo "[run_many_buildings_capture] Failed waiting for Malmo port after restart." >&2
    echo "[run_many_buildings_capture] Check logs: ${ROOT_DIR}/logs/malmo_client.log" >&2
    return 1
  fi
  return 0
}

if ! ensure_client_ready; then
  exit 1
fi

OUT_ABS="$("${PYTHON_BIN}" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${OUT}")"
mkdir -p "${OUT_ABS}"
mkdir -p "${OUT_ABS}/logs"

echo "[run_many_buildings_capture] output: ${OUT_ABS}"

is_complete_building() {
  local building_dir="$1"
  "${PYTHON_BIN}" - "${building_dir}" "${VIEWS}" <<'PY'
import json
import os
import sys

building_dir = sys.argv[1]
expected_views = int(sys.argv[2])
meta_path = os.path.join(building_dir, "meta.json")
bbox_path = os.path.join(building_dir, "gt", "bbox.json")
voxels_path = os.path.join(building_dir, "gt", "voxels.npy")
images_dir = os.path.join(building_dir, "images")

ok = (
    os.path.isfile(meta_path)
    and os.path.isfile(bbox_path)
    and os.path.isfile(voxels_path)
    and os.path.isdir(images_dir)
)
if not ok:
    raise SystemExit(1)

with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)
if len(meta.get("views", [])) < expected_views:
    raise SystemExit(1)

png_count = sum(1 for n in os.listdir(images_dir) if n.lower().endswith(".png"))
if png_count < expected_views:
    raise SystemExit(1)
PY
}

for ((i=0; i<COUNT; i++)); do
  STYLE_ID=$((START_STYLE_ID + i))
  BUILDING_DIR="${OUT_ABS}/building_$(printf '%03d' "${STYLE_ID}")"

  if [[ "${SKIP_EXISTING}" == "1" ]] && is_complete_building "${BUILDING_DIR}"; then
    echo "[run_many_buildings_capture] skip style_id=${STYLE_ID} (already complete)"
    continue
  fi

  success=0
  for ((attempt=1; attempt<=MAX_RETRIES; attempt++)); do
    echo "[run_many_buildings_capture] capture style_id=${STYLE_ID} attempt=${attempt}/${MAX_RETRIES} -> ${BUILDING_DIR}"
    set +e
    "${PYTHON_BIN}" "${ROOT_DIR}/tools/capture_one_building.py" \
      --out "${BUILDING_DIR}" \
      --port "${PORT}" \
      --views "${VIEWS}" \
      --image_size "${IMAGE_W}" "${IMAGE_H}" \
      --fov "${FOV}" \
      --seed "${SEED}" \
      --profile "${PROFILE}" \
      --style_id "${STYLE_ID}"
    status=$?
    set -e

    if [[ "${status}" -eq 0 ]] && is_complete_building "${BUILDING_DIR}"; then
      success=1
      break
    fi

    echo "[run_many_buildings_capture] Capture failed for style_id=${STYLE_ID} (attempt ${attempt}/${MAX_RETRIES})." >&2
    echo "[run_many_buildings_capture] Check these logs:" >&2
    echo "  - ${BUILDING_DIR}/logs/capture.log" >&2
    echo "  - ${ROOT_DIR}/logs/malmo_client.log" >&2

    if [[ "${attempt}" -lt "${MAX_RETRIES}" ]]; then
      if ! restart_client; then
        exit 1
      fi
    fi
  done

  if [[ "${success}" -ne 1 ]]; then
    echo "[run_many_buildings_capture] Giving up style_id=${STYLE_ID} after ${MAX_RETRIES} attempts." >&2
    exit 1
  fi
done

"${PYTHON_BIN}" - "${OUT_ABS}" "${COUNT}" "${START_STYLE_ID}" "${SEED}" "${VIEWS}" "${IMAGE_W}" "${IMAGE_H}" "${FOV}" "${SPACING}" <<'PY'
import datetime
import glob
import json
import os
import sys

out_root = sys.argv[1]
count = int(sys.argv[2])
start_style_id = int(sys.argv[3])
seed = int(sys.argv[4])
views = int(sys.argv[5])
image_w = int(sys.argv[6])
image_h = int(sys.argv[7])
fov = float(sys.argv[8])
spacing = int(sys.argv[9])

items = []
for style_id in range(start_style_id, start_style_id + count):
    building_dir = os.path.join(out_root, f"building_{style_id:03d}")
    meta_path = os.path.join(building_dir, "meta.json")
    if not os.path.isfile(meta_path):
        continue
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    items.append(
        {
            "style_id": style_id,
            "style": meta.get("style"),
            "palette": meta.get("palette"),
            "bbox": meta.get("bbox"),
            "views": len(meta.get("views", [])),
            "path": os.path.relpath(building_dir, out_root),
        }
    )

collection_meta = {
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "seed": seed,
    "count": count,
    "start_style_id": start_style_id,
    "views_per_building": views,
    "image_size": {"width": image_w, "height": image_h},
    "fov": fov,
    "spacing": spacing,
    "items": items,
}
with open(os.path.join(out_root, "meta_collection.json"), "w", encoding="utf-8") as f:
    json.dump(collection_meta, f, ensure_ascii=False, indent=2)
PY

echo "[run_many_buildings_capture] Capture completed successfully."
echo "[run_many_buildings_capture] Dataset root: ${OUT_ABS}"
