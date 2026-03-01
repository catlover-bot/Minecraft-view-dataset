#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/malmo_env_mac.sh"

PORT=10000
CLIENT_ARCH="${MALMO_CLIENT_ARCH:-auto}"

fail() {
  echo "[start_malmo_client_mac] ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  scripts/start_malmo_client_mac.sh [--port PORT]

Env:
  MALMO_CLIENT_ARCH=auto|x86_64|arm64
  MALMO_CLIENT_JAVA_HOME=/path/to/java8/home
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[start_malmo_client_mac] Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"
PID_FILE="${LOG_DIR}/malmo_client.pid"
LOG_FILE="${LOG_DIR}/malmo_client.log"

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

if is_listening; then
  echo "[start_malmo_client_mac] Port ${PORT} is already LISTEN; skip launch."
  exit 0
fi

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" >/dev/null 2>&1; then
    echo "[start_malmo_client_mac] Existing Malmo client process found (pid=${existing_pid}); skip launch."
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

launcher=""
for candidate in \
  "${MALMO_DIR}/Minecraft/launchClient.sh" \
  "${MALMO_DIR}/launchClient.sh" \
  "${MALMO_DIR}/Minecraft/run.sh"; do
  if [[ -f "${candidate}" ]]; then
    launcher="${candidate}"
    break
  fi
done

if [[ -z "${launcher}" ]]; then
  fail "launch script not found under MALMO_DIR=${MALMO_DIR}"
fi

launcher_dir="$(cd "$(dirname "${launcher}")" && pwd)"
launcher_base="$(basename "${launcher}")"

resolve_client_arch() {
  if [[ "${CLIENT_ARCH}" != "auto" ]]; then
    echo "${CLIENT_ARCH}"
    return
  fi
  if [[ "$(uname -m)" == "arm64" ]]; then
    echo "x86_64"
    return
  fi
  echo "$(uname -m)"
}

choose_java_home_for_arch() {
  local target_arch="$1"
  if [[ -n "${MALMO_CLIENT_JAVA_HOME:-}" ]]; then
    echo "${MALMO_CLIENT_JAVA_HOME}"
    return
  fi
  if [[ -x /usr/libexec/java_home ]]; then
    /usr/libexec/java_home -v 1.8 -a "${target_arch}" 2>/dev/null || true
  fi
}

effective_arch="$(resolve_client_arch)"
launcher_cmd=(bash "./${launcher_base}" -port "${PORT}")

case "${effective_arch}" in
  x86_64)
    if ! arch -x86_64 /usr/bin/uname -m >/dev/null 2>&1; then
      fail "Rosetta (x86_64 translation) が利用できません。'softwareupdate --install-rosetta --agree-to-license' を確認してください。"
    fi
    client_java_home="$(choose_java_home_for_arch x86_64)"
    if [[ -z "${client_java_home}" ]]; then
      fail "x86_64 Java 8 が見つかりません。MALMO_CLIENT_JAVA_HOME を設定するか x86_64 JDK8 をインストールしてください。"
    fi
    if [[ ! -x "${client_java_home}/bin/java" ]]; then
      fail "MALMO_CLIENT_JAVA_HOME が無効です: ${client_java_home}"
    fi
    launcher_cmd=(env JAVA_HOME="${client_java_home}" PATH="${client_java_home}/bin:${PATH}" arch -x86_64 bash "./${launcher_base}" -port "${PORT}")
    echo "[start_malmo_client_mac] Using x86_64 client JVM: ${client_java_home}"
    ;;
  arm64|aarch64)
    if [[ -n "${MALMO_CLIENT_JAVA_HOME:-}" ]]; then
      if [[ ! -x "${MALMO_CLIENT_JAVA_HOME}/bin/java" ]]; then
        fail "MALMO_CLIENT_JAVA_HOME が無効です: ${MALMO_CLIENT_JAVA_HOME}"
      fi
      launcher_cmd=(env JAVA_HOME="${MALMO_CLIENT_JAVA_HOME}" PATH="${MALMO_CLIENT_JAVA_HOME}/bin:${PATH}" bash "./${launcher_base}" -port "${PORT}")
    fi
    ;;
  *)
    fail "Unsupported MALMO_CLIENT_ARCH=${effective_arch} (auto|x86_64|arm64 を使用)"
    ;;
esac

(
  cd "${launcher_dir}"
  nohup "${launcher_cmd[@]}" >>"${LOG_FILE}" 2>&1 &
  echo $! >"${PID_FILE}"
)

echo "[start_malmo_client_mac] Launched Malmo client."
echo "[start_malmo_client_mac] pid=$(cat "${PID_FILE}")"
echo "[start_malmo_client_mac] log=${LOG_FILE}"
