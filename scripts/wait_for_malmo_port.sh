#!/usr/bin/env bash
set -euo pipefail

HOST="127.0.0.1"
PORT=10000
TIMEOUT=240
INTERVAL=1
QUIET=0

usage() {
  cat <<'EOF'
Usage:
  scripts/wait_for_malmo_port.sh [--host HOST] [--port PORT] [--timeout SEC] [--interval SEC] [--quiet]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    --interval)
      INTERVAL="$2"
      shift 2
      ;;
    --quiet)
      QUIET=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[wait_for_malmo_port] Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

is_listening() {
  local ok=1
  if command -v nc >/dev/null 2>&1; then
    nc -z "${HOST}" "${PORT}" >/dev/null 2>&1 && ok=0
    if [[ "${ok}" -ne 0 && ("${HOST}" == "127.0.0.1" || "${HOST}" == "localhost") ]]; then
      nc -z ::1 "${PORT}" >/dev/null 2>&1 && ok=0
    fi
  fi
  if [[ "${ok}" -ne 0 ]] && command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1 && ok=0
  fi
  return "${ok}"
}

start_epoch="$(date +%s)"
next_report_epoch="${start_epoch}"

while true; do
  if is_listening; then
    if [[ "${QUIET}" -eq 0 ]]; then
      echo "[wait_for_malmo_port] ${HOST}:${PORT} is LISTEN."
    fi
    exit 0
  fi

  now_epoch="$(date +%s)"
  elapsed=$((now_epoch - start_epoch))
  if (( elapsed >= TIMEOUT )); then
    echo "[wait_for_malmo_port] TIMEOUT: ${HOST}:${PORT} did not become LISTEN within ${TIMEOUT}s." >&2
    echo "[wait_for_malmo_port] Check logs: $(pwd)/logs/malmo_client.log" >&2
    echo "[wait_for_malmo_port] Possible causes:" >&2
    echo "  1) Java 8 が設定されていない (JAVA_HOME / java -version)" >&2
    echo "  2) MALMO_DIR が誤っている" >&2
    echo "  3) Malmo 起動スクリプトのパス不一致" >&2
    echo "[wait_for_malmo_port] Suggested next actions:" >&2
    echo "  - tail -n 120 logs/malmo_client.log" >&2
    echo "  - echo \$MALMO_DIR && echo \$JAVA_HOME" >&2
    exit 1
  fi

  if [[ "${QUIET}" -eq 0 && "${now_epoch}" -ge "${next_report_epoch}" ]]; then
    echo "[wait_for_malmo_port] waiting for ${HOST}:${PORT} ... ${elapsed}s/${TIMEOUT}s"
    next_report_epoch=$((now_epoch + 5))
  fi

  sleep "${INTERVAL}"
done
