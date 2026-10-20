#!/usr/bin/env bash
# Template installed at the plan-bound canonical root; never source this file.
set +x
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' 'Run this script with bash; sourcing is refused.' >&2
  return 64
fi
set -Eeuo pipefail
umask 077
readonly goal015c_repo='__GOAL015C_REPO__'
readonly goal015c_python='__GOAL015C_PYTHON__'
readonly goal015c_root='__GOAL015C_ROOT__'
readonly goal015c_plan="$goal015c_root/plan.json"
goal015c_args=(--plan "$goal015c_plan")
if (( $# )); then
  if [[ $# -ne 2 || "$1" != '--pricing-evidence' || -z "$2" ]]; then
    printf '%s\n' 'Usage: bash start-reviewed.sh [--pricing-evidence SAME_CONTRACT_REVIEWED_JSON]' >&2
    exit 64
  fi
  goal015c_args+=(--pricing-evidence "$2")
fi
if [[ "$0" != "$goal015c_root/start-reviewed.sh" || -L "$0" ]]; then
  printf '%s\n' 'Only the exact canonical launcher path may start this plan.' >&2
  exit 73
fi
goal015c_parent="$goal015c_root"
while [[ "$goal015c_parent" != '/' ]]; do
  if [[ -L "$goal015c_parent" || ! -d "$goal015c_parent" ]]; then
    printf '%s\n' 'Canonical run ancestry must contain regular directories.' >&2
    exit 73
  fi
  goal015c_parent="${goal015c_parent%/*}"
  [[ -n "$goal015c_parent" ]] || goal015c_parent='/'
done
goal015c_log=$(/usr/bin/mktemp -- "$goal015c_root/start-reviewed-$(/usr/bin/date -u +%Y%m%dT%H%M%SZ)-XXXXXXXX.log")
readonly goal015c_log
goal015c_finish() {
  local goal015c_exit=$?
  trap - EXIT
  unset OPENAI_API_KEY ANTHROPIC_API_KEY
  printf '\nWrapper exit status: %s\nLog: %s\nCompletion requires the status JSON and verified Draft proof. Task success is reported separately. Save failures require save-only recovery, never generation retry.\n' \
    "$goal015c_exit" "$goal015c_log" >&2
  printf 'Wrapper exit status: %s\nLog: %s\n' "$goal015c_exit" "$goal015c_log" >> "$goal015c_log" || true
  exit "$goal015c_exit"
}
trap goal015c_finish EXIT
goal015c_fail() {
  printf '%s\n' "$1" | /usr/bin/tee -a "$goal015c_log" >&2
  exit 1
}
cd -- "$goal015c_repo"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$goal015c_root/tmp"
if ! exec {goal015c_tty_read}</dev/tty {goal015c_tty_write}>/dev/tty; then
  goal015c_fail 'A real controlling terminal is required.'
fi
[[ -t "$goal015c_tty_read" && -t "$goal015c_tty_write" ]] || goal015c_fail 'Both terminal handles must be TTYs.'
goal015c_preflight() {
  "$goal015c_python" -B -m offline_evaluation.experiments.goal015c.cli preflight \
    "${goal015c_args[@]}" --require-unstarted 2>&1 | /usr/bin/tee -a "$goal015c_log"
}
goal015c_preflight || goal015c_fail 'Preflight refused this start before credential prompts.'
goal015c_require_key() {
  local goal015c_key_name="$1"
  case "$goal015c_key_name" in OPENAI_API_KEY|ANTHROPIC_API_KEY) ;; *) goal015c_fail 'Unsupported credential variable.' ;; esac
  if [[ -z "${!goal015c_key_name:-}" ]]; then
    printf '%s (hidden; empty/EOF cancels): ' "$goal015c_key_name" >&"$goal015c_tty_write"
    if ! IFS= read -r -s -u "$goal015c_tty_read" "$goal015c_key_name"; then
      printf '\n' >&"$goal015c_tty_write"
      goal015c_fail 'Credential entry cancelled or terminal read failed.'
    fi
    printf '\n' >&"$goal015c_tty_write"
    [[ -n "${!goal015c_key_name}" ]] || goal015c_fail 'Empty credential entry; no launch.'
  fi
  export "$goal015c_key_name"
}
goal015c_require_key OPENAI_API_KEY
goal015c_require_key ANTHROPIC_API_KEY
goal015c_preflight || goal015c_fail 'Preflight refused this start after credential entry; no launch.'
exec {goal015c_tty_read}<&-
exec {goal015c_tty_write}>&-
printf '%s\n' 'Invoking this new 12-call diagnostic once, followed by evaluation and a new verified Draft save. Exact real-TTY approval remains required. No automatic resume or retry.' \
  | /usr/bin/tee -a "$goal015c_log"
set +e
"$goal015c_python" -B -m offline_evaluation.experiments.goal015c.cli run \
  "${goal015c_args[@]}" --allow-create 2>&1 | /usr/bin/tee -a "$goal015c_log"
goal015c_pipeline_status=("${PIPESTATUS[@]}")
set -e
printf 'CLI process exit status: %s; log writer exit status: %s\n' \
  "${goal015c_pipeline_status[0]}" "${goal015c_pipeline_status[1]}" | /usr/bin/tee -a "$goal015c_log"
if (( goal015c_pipeline_status[1] != 0 )); then
  exit 74
fi
exit "${goal015c_pipeline_status[0]}"
