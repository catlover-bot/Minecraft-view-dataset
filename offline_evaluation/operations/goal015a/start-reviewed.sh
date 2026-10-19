#!/usr/bin/env bash
# Run with bash directly in the user's tmux pane; never source this file.
set +x
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' 'Refusing sourced invocation; run this file with bash.' >&2
  return 64
fi
set -Eeuo pipefail
umask 077

readonly goal015a_repo='/root/workspace/Minecraft-view-dataset'
readonly goal015a_python='/root/workspace/Minecraft-view-dataset/.venv-server-v2-20261013/bin/python'
readonly goal015a_root='/root/minecraft-runs/goal015a-20261019T141842Z-13cfbc79'
readonly goal015a_plan="$goal015a_root/plan.json"
# A later, explicitly reviewed parent-contract pricing file may be supplied here.
readonly goal015a_evidence="${GOAL015A_PRICING_EVIDENCE:-/root/minecraft-runs/goal015-20261017T170546Z/pricing-refresh-20261018T173449Z-fd5237eb/reviewed.json}"

goal015a_parent="$goal015a_root"
while [[ "$goal015a_parent" != '/' ]]; do
  if [[ -L "$goal015a_parent" || ! -d "$goal015a_parent" ]]; then
    printf '%s\n' 'Canonical run directory ancestry is not a regular directory path.' >&2
    exit 73
  fi
  goal015a_parent="${goal015a_parent%/*}"
  [[ -n "$goal015a_parent" ]] || goal015a_parent='/'
done
goal015a_log=$(/usr/bin/mktemp -- "$goal015a_root/start-reviewed-$(/usr/bin/date -u +%Y%m%dT%H%M%SZ)-XXXXXXXX.log")
readonly goal015a_log

goal015a_finish() {
  local goal015a_exit=$?
  trap - EXIT
  unset OPENAI_API_KEY ANTHROPIC_API_KEY
  printf '\nWrapper exit status: %s\nLog: %s\nOnly COMPLETE_SAVED with verified counts and GitHub proof establishes the completed workflow. Settled calls do not imply successful reconstructions.\n' \
    "$goal015a_exit" "$goal015a_log" >&2
  printf 'Wrapper exit status: %s\nLog: %s\n' "$goal015a_exit" "$goal015a_log" >> "$goal015a_log" || true
  exit "$goal015a_exit"
}
trap goal015a_finish EXIT
goal015a_fail() {
  printf '%s\n' "$1" | /usr/bin/tee -a "$goal015a_log" >&2
  exit 1
}
[[ $# -eq 0 ]] || goal015a_fail 'No arguments accepted; use only the fixed residual-eight plan.'
cd -- "$goal015a_repo"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$goal015a_root/tmp"

if ! exec {goal015a_tty_read}</dev/tty {goal015a_tty_write}>/dev/tty; then
  goal015a_fail 'A real controlling terminal is required.'
fi
[[ -t "$goal015a_tty_read" && -t "$goal015a_tty_write" ]] || goal015a_fail 'Both terminal handles must be TTYs.'
[[ -n "${TMUX:-}" && "${TMUX_PANE:-}" =~ ^%[0-9]+$ ]] || goal015a_fail 'Run directly inside the intended tmux pane.'
goal015a_pane_tty=$(/usr/bin/tmux display-message -p -t "$TMUX_PANE" '#{pane_tty}') || goal015a_fail 'Cannot verify the tmux pane.'
goal015a_process_tty=$(/bin/ps -o tty= -p "$$") || goal015a_fail 'Cannot verify the process controlling terminal.'
goal015a_process_tty="${goal015a_process_tty//[[:space:]]/}"
[[ "$goal015a_process_tty" != '?' && "/dev/$goal015a_process_tty" == "$goal015a_pane_tty" ]] \
  || goal015a_fail 'tmux pane TTY and this process controlling TTY differ.'

goal015a_preflight() {
  "$goal015a_python" -B - "$goal015a_repo" "$goal015a_plan" "$goal015a_evidence" <<'PY' 2>&1 | /usr/bin/tee -a "$goal015a_log"
from pathlib import Path
import json
import os
import subprocess
import sys

runtime = None
try:
    repo, plan_path, evidence_path = sys.argv[1:]
    sys.path.insert(0, repo)
    from offline_evaluation.operations.goal015a import runtime
    plan = runtime.verify_plan(plan_path)
    runtime.require(plan.get('fixture_only') is not True, 'Canonical production child required.')
    committed = subprocess.check_output(['git', 'show', 'HEAD:reports/server_only/goal015a/EXECUTION_BINDING.json'],
                                        cwd=repo, stderr=subprocess.DEVNULL)
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=repo).decode().strip()
    runtime.require(committed == runtime.EXECUTION_BINDING.read_bytes()
                    and branch == 'research/server-only-20261013', 'Committed child registry on dedicated branch required.')
    root = runtime.path_safe(plan['canonical_root'])
    runtime.require(root == Path(plan_path).parent, 'Canonical child root differs.')
    for relative in ('enabled', 'approval.json', 'authority.json', 'ledger.json', 'execution.lock',
                     'enabled/approval.json', 'enabled/authority.json', 'enabled/ledger.json'):
        runtime.require(not os.path.lexists(root/relative), 'Existing execution requires status review and explicit resume/finish; do not start.')
    runtime.require(runtime.path_safe(root/'tmp').is_dir(), 'Run temporary directory is required.')
    with runtime.paired_locks(plan):
        runtime.verify_parent(plan)
        evidence = runtime.pricing_fresh(plan, evidence_path)
        reviewed = runtime.bound(evidence)
        runtime.require(reviewed.get('same_plan_id') == plan['parent_plan_id'], 'Pricing evidence must bind the original parent contract.')
    print(json.dumps(runtime.preview(plan_path), ensure_ascii=False, sort_keys=True))
    print('Read-only preflight passed: fixed parent72, residual8, committed identity, current pricing and absent child execution.')
except Exception as exc:
    reason = str(exc) if runtime is not None and isinstance(exc, runtime.RuntimeErrorSafe) else type(exc).__name__
    print('Read-only preflight failed: ' + reason + '; no launch.', file=sys.stderr)
    raise SystemExit(1) from None
PY
}
goal015a_preflight || goal015a_fail 'Preflight refused this start before credential prompts.'

goal015a_require_key() {
  local goal015a_key_name="$1"
  case "$goal015a_key_name" in OPENAI_API_KEY|ANTHROPIC_API_KEY) ;; *) goal015a_fail 'Unsupported credential variable.' ;; esac
  if [[ -z "${!goal015a_key_name:-}" ]]; then
    printf '%s (hidden; empty/EOF cancels): ' "$goal015a_key_name" >&"$goal015a_tty_write"
    if ! IFS= read -r -s -u "$goal015a_tty_read" "$goal015a_key_name"; then
      printf '\n' >&"$goal015a_tty_write"
      goal015a_fail 'Credential entry cancelled or terminal read failed.'
    fi
    printf '\n' >&"$goal015a_tty_write"
    [[ -n "${!goal015a_key_name}" ]] || goal015a_fail 'Empty credential entry; no launch.'
  fi
  export "$goal015a_key_name"
}
goal015a_require_key OPENAI_API_KEY
goal015a_require_key ANTHROPIC_API_KEY
goal015a_preflight || goal015a_fail 'Preflight refused this start after credential entry; no launch.'
exec {goal015a_tty_read}<&-
exec {goal015a_tty_write}>&-

printf '%s\n' 'Invoking residual8 generation -> CPU evaluation -> verified Draft save once. Exact new real-TTY approval remains required; no automatic resume or retry.' \
  | /usr/bin/tee -a "$goal015a_log"
set +e
"$goal015a_python" -B -m offline_evaluation.operations.goal015a.cli run \
  --plan "$goal015a_plan" --pricing-evidence "$goal015a_evidence" 2>&1 | /usr/bin/tee -a "$goal015a_log"
goal015a_pipeline_status=("${PIPESTATUS[@]}")
set -e
printf 'CLI process exit status: %s; log writer exit status: %s\n' \
  "${goal015a_pipeline_status[0]}" "${goal015a_pipeline_status[1]}" | /usr/bin/tee -a "$goal015a_log"
if (( goal015a_pipeline_status[1] != 0 )); then
  exit 74
fi
exit "${goal015a_pipeline_status[0]}"
