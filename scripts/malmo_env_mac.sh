#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[malmo_env_mac] ERROR: $*" >&2
  if [[ "${BASH_SOURCE[0]-}" != "$0" ]]; then
    return 1
  fi
  exit 1
}

if [[ -z "${MALMO_DIR:-}" ]]; then
  fail "MALMO_DIR が未設定です。例: export MALMO_DIR=\"\$HOME/MalmoPlatform\""
fi
if [[ ! -d "${MALMO_DIR}" ]]; then
  fail "MALMO_DIR が存在しません: ${MALMO_DIR}"
fi

if [[ -z "${JAVA_HOME:-}" ]]; then
  if [[ -x /usr/libexec/java_home ]]; then
    JAVA_HOME="$(/usr/libexec/java_home -v 1.8 2>/dev/null || true)"
  fi
fi
if [[ -z "${JAVA_HOME:-}" ]]; then
  fail "JAVA_HOME が未設定です。Java 8 をインストールし、export JAVA_HOME=\"\$(/usr/libexec/java_home -v 1.8)\" を設定してください。"
fi
if [[ ! -d "${JAVA_HOME}" ]]; then
  fail "JAVA_HOME が無効です: ${JAVA_HOME}"
fi
export JAVA_HOME

if [[ -z "${MALMO_XSD_PATH:-}" ]]; then
  for candidate in \
    "${MALMO_DIR}/Schemas/Mission.xsd" \
    "${MALMO_DIR}/Malmo/Schemas/Mission.xsd" \
    "${MALMO_DIR}/../Schemas/Mission.xsd"; do
    if [[ -f "${candidate}" ]]; then
      MALMO_XSD_PATH="${candidate}"
      break
    fi
  done
fi
# Malmo expects MALMO_XSD_PATH to be a directory containing Mission.xsd.
if [[ -f "${MALMO_XSD_PATH:-}" ]]; then
  MALMO_XSD_PATH="$(cd "$(dirname "${MALMO_XSD_PATH}")" && pwd)"
fi
if [[ -n "${MALMO_XSD_PATH:-}" && ! -f "${MALMO_XSD_PATH}/Mission.xsd" ]]; then
  for candidate in \
    "${MALMO_DIR}/Schemas/Mission.xsd" \
    "${MALMO_DIR}/Malmo/Schemas/Mission.xsd" \
    "${MALMO_DIR}/../Schemas/Mission.xsd"; do
    if [[ -f "${candidate}" ]]; then
      MALMO_XSD_PATH="$(cd "$(dirname "${candidate}")" && pwd)"
      break
    fi
  done
fi
if [[ -z "${MALMO_XSD_PATH:-}" || ! -f "${MALMO_XSD_PATH}/Mission.xsd" ]]; then
  fail "MALMO_XSD_PATH を解決できません。例: export MALMO_XSD_PATH=\"\$MALMO_DIR/Schemas\""
fi
export MALMO_XSD_PATH

if [[ -z "${MALMO_PYTHON_PATH:-}" ]]; then
  for candidate in \
    "${MALMO_DIR}/Python_Examples" \
    "${MALMO_DIR}/build/install/Python_Examples" \
    "${MALMO_DIR}/build/Python_Examples" \
    "${MALMO_DIR}/MalmoEnv" \
    "${MALMO_DIR}"; do
    if [[ -d "${candidate}" ]]; then
      MALMO_PYTHON_PATH="${candidate}"
      break
    fi
  done
fi
if [[ -z "${MALMO_PYTHON_PATH:-}" || ! -d "${MALMO_PYTHON_PATH}" ]]; then
  fail "MALMO_PYTHON_PATH を解決できません。例: export MALMO_PYTHON_PATH=\"\$MALMO_DIR/Python_Examples\""
fi

if [[ -z "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${MALMO_PYTHON_PATH}"
else
  case ":${PYTHONPATH}:" in
    *":${MALMO_PYTHON_PATH}:"*) ;;
    *) export PYTHONPATH="${MALMO_PYTHON_PATH}:${PYTHONPATH}" ;;
  esac
fi
