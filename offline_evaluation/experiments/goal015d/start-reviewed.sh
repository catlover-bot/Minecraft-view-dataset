#!/usr/bin/env bash
# Template installed at the plan-bound canonical root; never source this file.
set +x
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' 'Run this script with bash; sourcing is refused.' >&2
  return 64
fi
set -Eeuo pipefail
umask 077
readonly goal015d_repo='__GOAL015D_REPO__'
readonly goal015d_python='__GOAL015D_PYTHON__'
readonly goal015d_root='__GOAL015D_ROOT__'
readonly goal015d_plan="$goal015d_root/plan.json"
goal015d_args=(--plan "$goal015d_plan")
if (( $# )); then
  if [[ $# -ne 2 || "$1" != '--pricing-evidence' || -z "$2" ]]; then
    printf '%s\n' 'Usage: bash start-reviewed.sh [--pricing-evidence SAME_CONTRACT_REVIEWED_JSON]' >&2
    exit 64
  fi
  goal015d_args+=(--pricing-evidence "$2")
fi
if [[ "$0" != "$goal015d_root/start-reviewed.sh" || -L "$0" ]]; then
  printf '%s\n' 'Only the exact canonical launcher path may start this plan.' >&2
  exit 73
fi
goal015d_parent="$goal015d_root"
while [[ "$goal015d_parent" != '/' ]]; do
  if [[ -L "$goal015d_parent" || ! -d "$goal015d_parent" ]]; then
    printf '%s\n' 'Canonical run ancestry must contain regular directories.' >&2
    exit 73
  fi
  goal015d_parent="${goal015d_parent%/*}"
  [[ -n "$goal015d_parent" ]] || goal015d_parent='/'
done
goal015d_log=$(/usr/bin/mktemp -- "$goal015d_root/start-reviewed-$(/usr/bin/date -u +%Y%m%dT%H%M%SZ)-XXXXXXXX.log")
readonly goal015d_log
goal015d_finish() {
  local goal015d_exit=$?
  trap - EXIT
  unset OPENAI_API_KEY ANTHROPIC_API_KEY
  printf '\nWrapper exit status: %s\nLog: %s\nCompletion requires the status JSON and verified Draft proof. Task success is reported separately. Save failures require save-only recovery, never generation retry.\n' \
    "$goal015d_exit" "$goal015d_log" >&2
  printf 'Wrapper exit status: %s\nLog: %s\n' "$goal015d_exit" "$goal015d_log" >> "$goal015d_log" || true
  exit "$goal015d_exit"
}
trap goal015d_finish EXIT
goal015d_fail() {
  printf '%s\n' "$1" | /usr/bin/tee -a "$goal015d_log" >&2
  exit 1
}
cd -- "$goal015d_repo"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$goal015d_root/tmp"
if ! exec {goal015d_tty_read}</dev/tty {goal015d_tty_write}>/dev/tty; then
  goal015d_fail 'A real controlling terminal is required.'
fi
[[ -t "$goal015d_tty_read" && -t "$goal015d_tty_write" ]] || goal015d_fail 'Both terminal handles must be TTYs.'
goal015d_preflight() {
  "$goal015d_python" -B -m offline_evaluation.experiments.goal015d.cli preflight \
    "${goal015d_args[@]}" --require-unstarted 2>&1 | /usr/bin/tee -a "$goal015d_log"
}
goal015d_preflight || goal015d_fail 'Preflight refused this start before credential prompts.'
goal015d_require_key() {
  local goal015d_key_name="$1"
  case "$goal015d_key_name" in OPENAI_API_KEY|ANTHROPIC_API_KEY) ;; *) goal015d_fail 'Unsupported credential variable.' ;; esac
  if [[ -z "${!goal015d_key_name:-}" ]]; then
    printf '%s (hidden; empty/EOF cancels): ' "$goal015d_key_name" >&"$goal015d_tty_write"
    if ! IFS= read -r -s -u "$goal015d_tty_read" "$goal015d_key_name"; then
      printf '\n' >&"$goal015d_tty_write"
      goal015d_fail 'Credential entry cancelled or terminal read failed.'
    fi
    printf '\n' >&"$goal015d_tty_write"
    [[ -n "${!goal015d_key_name}" ]] || goal015d_fail 'Empty credential entry; no launch.'
  fi
  export "$goal015d_key_name"
}
goal015d_require_key OPENAI_API_KEY
goal015d_require_key ANTHROPIC_API_KEY
goal015d_preflight || goal015d_fail 'Preflight refused this start after credential entry; no launch.'
exec {goal015d_tty_read}<&-
exec {goal015d_tty_write}>&-
printf '%s\n' 'Invoking this new 16-call diagnostic once, followed by evaluation and a new verified Draft save. Exact real-TTY approval remains required. No automatic resume or retry.' \
  | /usr/bin/tee -a "$goal015d_log"
set +e
"$goal015d_python" -B -m offline_evaluation.experiments.goal015d.cli run \
  "${goal015d_args[@]}" --allow-create 2>&1 | /usr/bin/tee -a "$goal015d_log"
goal015d_pipeline_status=("${PIPESTATUS[@]}")
set -e
printf 'CLI process exit status: %s; log writer exit status: %s\n' \
  "${goal015d_pipeline_status[0]}" "${goal015d_pipeline_status[1]}" | /usr/bin/tee -a "$goal015d_log"
if (( goal015d_pipeline_status[1] != 0 )); then
  exit 74
fi
exit "${goal015d_pipeline_status[0]}"
