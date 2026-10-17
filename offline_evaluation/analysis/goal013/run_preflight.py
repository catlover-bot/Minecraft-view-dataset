"""Run only the source-binding branch after committing the blocked protocol.

No model loader or association computation is provided. The only accepted
availability state is BLOCKED_NO_VALIDATED_OFFLINE_SCORER.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time

from . import source_binding as sources
from ..goal010.source_guard import locked_sources

ROOT = sources.ROOT
REPORTS = ROOT / 'reports/server_only/goal013'
BLOCKED = 'BLOCKED_NO_VALIDATED_OFFLINE_SCORER'
PREFIXES = ('offline_evaluation/analysis/goal013/', 'reports/server_only/goal013/')


def file_descriptor(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def verify_preflight_commit(commit):
    sources.require(bool(re.fullmatch('[0-9a-f]{40}', commit)), 'full code/spec commit required')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    sources.require(head == commit, 'run must use the committed preflight HEAD')
    names = subprocess.check_output(
        ['git', 'ls-tree', '-r', '--name-only', commit, '--', *PREFIXES], cwd=ROOT
    ).decode().splitlines()
    required = {
        'offline_evaluation/analysis/goal013/source_binding.py',
        'offline_evaluation/analysis/goal013/run_preflight.py',
        'reports/server_only/goal013/SCORING_PROTOCOL.md',
        'reports/server_only/goal013/MODEL_SELECTION_RULE.json',
        'reports/server_only/goal013/OFFLINE_SCORER_AUDIT.json',
    }
    sources.require(required <= set(names), 'code, protocol and availability evidence must be committed')
    descriptors = {}
    for name in names:
        path = sources.safe(ROOT/name, ROOT)
        blob = subprocess.check_output(['git', 'show', commit+':'+name], cwd=ROOT)
        descriptor = file_descriptor(path)
        sources.require(hashlib.sha256(blob).hexdigest() == descriptor['sha256'], 'committed preflight file changed')
        descriptors[name] = descriptor
    return descriptors


def require_blocked_scorer(audit, rule_bytes):
    sources.require(audit['status'] == BLOCKED, 'this runner handles the blocked branch only')
    sources.require(audit['computation_status'] == 'COMPUTATION_NOT_PERFORMED', 'scoring state is not blocked')
    sources.require(audit['validated_offline_scorer_count'] == 0, 'unexpected validated scorer')
    sources.require(audit['selected_candidate'] is None, 'no active checkpoint may be selected')
    for field in ('scored_description_count', 'scored_image_text_pair_count'):
        sources.require(type(audit[field]) is int and audit[field] == 0, 'preflight may not ingest any scores')
    actions = audit['actions']
    required_actions = {'research_generation_api_calls', 'model_lookup_api_calls', 'token_count_api_calls',
                        'judge_api_calls', 'inference_calls', 'human_experiments', 'secret_or_authentication_queries',
                        'downloads', 'package_installs', 'existing_venv_mutations', 'existing_cache_mutations'}
    sources.require(required_actions <= set(actions), 'required zero-action evidence is missing')
    sources.require(all(type(value) is int and value == 0 for value in actions.values()),
                    'preflight evidence includes forbidden or unverified actions')
    sources.require(audit['actual_correlation_n'] is None and audit['paired_denominators'] is None,
                    'unscored branch cannot contain association denominators')
    sources.require(audit['selection_rule']['sha256'] == hashlib.sha256(rule_bytes).hexdigest(),
                    'mechanical selection rule differs from audited rule')


def prepare_output(path):
    path = sources.safe(path)
    sources.require(path.parent == Path('/root/minecraft-runs')
                    and bool(re.fullmatch(r'goal013-\d{8}T\d{6}Z', path.name)),
                    'a unique canonical Goal013 run root is required')
    path.mkdir(exist_ok=False)
    return path


def restrict_writes(destination):
    """Guard common Python filesystem writes; this is not OS-level isolation."""
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND

    def permitted(path):
        sources.require(not isinstance(path, int), 'descriptor-only mutation refused')
        candidate = Path(os.fsdecode(path)).absolute()
        sources.require(candidate.is_relative_to(destination) and '..' not in candidate.parts,
                        'write outside the new Goal013 run refused')
        sources.safe(candidate, destination)

    def audit(event, args):
        if event == 'open':
            path, mode, flags = args
            writing = bool((flags or 0) & write_flags) or (isinstance(mode, str) and any(c in mode for c in 'wax+'))
            if writing:
                permitted(path)
        elif event in {'os.symlink', 'os.link'}:
            raise sources.SourceBindingError('link creation refused during source preflight')
        elif event in {'os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod', 'os.chown', 'os.utime', 'os.truncate'}:
            permitted(args[0])
        elif event in {'os.rename', 'os.replace'}:
            permitted(args[0]); permitted(args[1])
    sys.addaudithook(audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--code-commit', required=True)
    args = parser.parse_args()
    committed = verify_preflight_commit(args.code_commit)
    audit = json.loads((REPORTS/'OFFLINE_SCORER_AUDIT.json').read_text())
    require_blocked_scorer(audit, (REPORTS/'MODEL_SELECTION_RULE.json').read_bytes())
    destination = prepare_output(args.output_root)
    restrict_writes(destination)
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    resource.setrlimit(resource.RLIMIT_AS, (2*1024**3, 2*1024**3))
    signal.alarm(300)
    started = time.monotonic()
    cpu = time.process_time()
    base = {
        'schema': 'goal013_blocked_preflight_v1',
        'analysis_label': sources.LABEL,
        'code_protocol_commit': args.code_commit,
        'committed_file_descriptors': committed,
        'run_root': str(destination),
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'research_api_requests_sent': 0,
        'model_queries': 0, 'token_count_api_requests': 0, 'api_key_queries': 0,
        'judge_api_requests': 0, 'human_studies': 0, 'model_inference_calls': 0,
        'weights_downloaded': 0, 'package_installs': 0, 'existing_environment_mutations': 0,
        'score_id': None, 'selected_checkpoint': None, 'weight_sha256': None,
        'computation_status': 'COMPUTATION_NOT_PERFORMED',
        'scored_descriptions': 0, 'target_descriptions': 180, 'scored_image_text_pairs': 0,
        'actual_association_N': None, 'paired_denominators': None,
        'budgets': {'cpu_seconds': 120, 'wall_seconds': 300, 'address_space_bytes': 2*1024**3,
                    'threads': 1, 'overwrite': False},
        'OS_level_isolation_guaranteed': False,
    }
    write_json(destination/'STARTED.json', base)
    try:
        with locked_sources() as locks:
            result = sources.audit_sources()
            products = sources.write_outputs(result, destination/'phase0')
            sources.verify_inventory(result['receipt']['source_inventory'])
            receipt = {**base, 'status': BLOCKED, 'RQ2_GATE': 'BLOCKED_NO_SCORER',
                       'source_binding_status': 'PASS', 'phase0_overall_status': 'BLOCKED',
                       'source_locks': len(locks), 'source_products': products,
                       'bound_descriptions': len(result['rows']),
                       'independent_source_buildings': result['receipt']['independent_buildings'],
                       'phase1_executed': False, 'phase2_executed': False,
                       'cpu_seconds': time.process_time()-cpu,
                       'elapsed_seconds': time.monotonic()-started,
                       'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        # The source-lock context performs its final process check on exit.
        # Publish success only after that check has passed.
        receipt.update(cpu_seconds=time.process_time()-cpu,
                       elapsed_seconds=time.monotonic()-started,
                       max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(destination/'PREFLIGHT_RECEIPT.json', receipt)
    except Exception as exc:
        write_json(destination/'FAILED.json', {**base, 'status': 'SOURCE_BINDING_FAILED',
                                              'reason': str(exc), 'error_type': type(exc).__name__})
        raise
    signal.alarm(0)
    print(json.dumps({k: receipt[k] for k in ('status', 'RQ2_GATE', 'source_binding_status',
        'bound_descriptions', 'scored_descriptions', 'cpu_seconds', 'elapsed_seconds', 'max_rss_kib')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
