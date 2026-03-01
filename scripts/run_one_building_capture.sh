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

OUT="datasets/one_building_v1"
PORT=10000
VIEWS=12
IMAGE_W=960
IMAGE_H=540
FOV=70
SEED=1234
STYLE_ID=""
PROFILE="complex"
WAIT_TIMEOUT=240

usage() {
  cat <<'EOF'
Usage:
  scripts/run_one_building_capture.sh \
    [--out OUT_DIR] \
    [--port PORT] \
    [--views N] \
    [--image_size W H] \
    [--fov FOV] \
    [--seed SEED] \
    [--style_id STYLE_ID] \
    [--profile complex|simple] \
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
    --style_id)
      STYLE_ID="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
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
      echo "[run_one_building_capture] Unknown arg: $1" >&2
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

if ! is_listening; then
  echo "[run_one_building_capture] Malmo client is not listening on :${PORT}. Launching client..."
  "${SCRIPT_DIR}/start_malmo_client_mac.sh" --port "${PORT}"
else
  echo "[run_one_building_capture] :${PORT} is already LISTEN."
fi

if ! "${SCRIPT_DIR}/wait_for_malmo_port.sh" --host 127.0.0.1 --port "${PORT}" --timeout "${WAIT_TIMEOUT}"; then
  echo "[run_one_building_capture] Failed waiting for Malmo port." >&2
  echo "[run_one_building_capture] Check logs: ${ROOT_DIR}/logs/malmo_client.log" >&2
  exit 1
fi

OUT_ABS="$("${PYTHON_BIN}" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${OUT}")"

mkdir -p "${OUT_ABS}"
echo "[run_one_building_capture] output: ${OUT_ABS}"

set +e
"${PYTHON_BIN}" "${ROOT_DIR}/tools/capture_one_building.py" \
  --out "${OUT_ABS}" \
  --port "${PORT}" \
  --views "${VIEWS}" \
  --image_size "${IMAGE_W}" "${IMAGE_H}" \
  --fov "${FOV}" \
  --seed "${SEED}" \
  --profile "${PROFILE}" \
  ${STYLE_ID:+--style_id "${STYLE_ID}"}
status=$?
set -e

if [[ "${status}" -ne 0 ]]; then
  echo "[run_one_building_capture] Capture failed." >&2
  echo "[run_one_building_capture] Check these logs:" >&2
  echo "  - ${OUT_ABS}/logs/capture.log" >&2
  echo "  - ${ROOT_DIR}/logs/malmo_client.log" >&2
  exit "${status}"
fi

echo "[run_one_building_capture] Capture completed successfully."
echo "[run_one_building_capture] Dataset root: ${OUT_ABS}"
