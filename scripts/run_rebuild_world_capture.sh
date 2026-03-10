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

REBUILD_WORLD_DIR=""
OUT=""
PORT=10000
VIEWS=8
IMAGE_W=960
IMAGE_H=540
FOV=70
SHIFT_X=0
SHIFT_Y=4
SHIFT_Z=0
WAIT_TIMEOUT=240

usage() {
  cat <<'EOF'
Usage:
  scripts/run_rebuild_world_capture.sh \
    --rebuild_world_dir outputs/i2t2b/buildings_100_v1/building_000/rebuild_world_xxx \
    [--out OUT_DIR] \
    [--port PORT] \
    [--views N] \
    [--image_size W H] \
    [--fov FOV] \
    [--shift_x N] \
    [--shift_y N] \
    [--shift_z N] \
    [--wait_timeout SEC]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rebuild_world_dir)
      REBUILD_WORLD_DIR="$2"
      shift 2
      ;;
    --out)
      OUT="$2"
      shift 2
      ;;
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
    --shift_x)
      SHIFT_X="$2"
      shift 2
      ;;
    --shift_y)
      SHIFT_Y="$2"
      shift 2
      ;;
    --shift_z)
      SHIFT_Z="$2"
      shift 2
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
      echo "[run_rebuild_world_capture] Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${REBUILD_WORLD_DIR}" ]]; then
  echo "[run_rebuild_world_capture] --rebuild_world_dir is required." >&2
  usage
  exit 1
fi

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

if ! is_listening; then
  echo "[run_rebuild_world_capture] Malmo client is not listening on :${PORT}. Launching client..."
  "${SCRIPT_DIR}/start_malmo_client_mac.sh" --port "${PORT}"
else
  echo "[run_rebuild_world_capture] :${PORT} is already LISTEN."
fi

if ! "${SCRIPT_DIR}/wait_for_malmo_port.sh" --host 127.0.0.1 --port "${PORT}" --timeout "${WAIT_TIMEOUT}"; then
  echo "[run_rebuild_world_capture] Failed waiting for Malmo port." >&2
  echo "[run_rebuild_world_capture] Check logs: ${ROOT_DIR}/logs/malmo_client.log" >&2
  exit 1
fi

REBUILD_WORLD_DIR_ABS="$("${PYTHON_BIN}" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${REBUILD_WORLD_DIR}")"

if [[ -z "${OUT}" ]]; then
  parent_dir="$(dirname "${REBUILD_WORLD_DIR_ABS}")"
  world_name="$(basename "${REBUILD_WORLD_DIR_ABS}")"
  OUT="${parent_dir}/capture_${world_name}"
fi
OUT_ABS="$("${PYTHON_BIN}" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${OUT}")"

mkdir -p "${OUT_ABS}"
echo "[run_rebuild_world_capture] source rebuild_world: ${REBUILD_WORLD_DIR_ABS}"
echo "[run_rebuild_world_capture] output: ${OUT_ABS}"

set +e
"${PYTHON_BIN}" "${ROOT_DIR}/tools/capture_rebuild_world.py" \
  --rebuild_world_dir "${REBUILD_WORLD_DIR_ABS}" \
  --out "${OUT_ABS}" \
  --port "${PORT}" \
  --views "${VIEWS}" \
  --image_size "${IMAGE_W}" "${IMAGE_H}" \
  --fov "${FOV}" \
  --shift_x "${SHIFT_X}" \
  --shift_y "${SHIFT_Y}" \
  --shift_z "${SHIFT_Z}"
status=$?
set -e

if [[ "${status}" -ne 0 ]]; then
  echo "[run_rebuild_world_capture] Capture failed." >&2
  echo "[run_rebuild_world_capture] Check these logs:" >&2
  echo "  - ${OUT_ABS}/logs/capture.log" >&2
  echo "  - ${ROOT_DIR}/logs/malmo_client.log" >&2
  exit "${status}"
fi

echo "[run_rebuild_world_capture] Capture completed successfully."
echo "[run_rebuild_world_capture] Output root: ${OUT_ABS}"

