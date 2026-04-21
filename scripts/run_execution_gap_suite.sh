#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [[ -x /opt/homebrew/bin/python3 ]]; then
  PYTHON_BIN="/opt/homebrew/bin/python3"
else
  PYTHON_BIN="$(command -v python3)"
fi

if [[ $# -eq 0 ]]; then
  cat <<'EOF'
Usage:
  scripts/run_execution_gap_suite.sh [options]

Examples:
  scripts/run_execution_gap_suite.sh
  scripts/run_execution_gap_suite.sh --limit 10 --overwrite_agentexec
  scripts/run_execution_gap_suite.sh --include_gemini_cases --dotenv .env
EOF
fi

cd "${ROOT_DIR}"
"${PYTHON_BIN}" "${ROOT_DIR}/tools/run_execution_gap_suite.py" "$@"
