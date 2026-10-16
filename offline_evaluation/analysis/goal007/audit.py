"""Read-only Goal007 evidence gate. Never generates or computes quality scores.

Canonical files are opened read-only and locked using existing lock files. Only
new audit outputs and fresh GitHub GET downloads are written. Frozen original
validators are called directly; the complete-fixed960 aggregation guard is unused.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import csv
from decimal import Decimal
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess

from ...server_v2.generation import object_hash, read_json as _read_json, utc_now
from ...server_v2.data import sha256_file

ROOT = Path(__file__).resolve().parents[3]
VERSION = 'goal007_readonly_audit_v1'
C_PLAN = Path('/root/minecraft-runs/goal006c-remaining17-20261016T001844Z/plan.json')
EXPECTED = {
    'parent160': ('f54ebfaa041a18cbbfb60d2d27f9e9c4a50d436545c7ed1175c60751be58816f', 160),
    'original781': ('db5533bffb583d0b1efcbcdb7c1563edf2ac87606334e6ce6cd766c650075843', 781),
    'supplement3': ('206dfda6468a90b8d67700e3799571ff0de5aabe1263c15754efe242887f8922', 3),
    'supplement17': ('03340ae00dab4ab13be7cc8e9de379652a0ea92cf2cec2edda38e006c2668e4c', 17),
}
CHECKPOINTS = {
    'original781': {'directory': '/root/minecraft-checkpoints/goal006-db5533bffb58-0781-20261015T184855214460Z-e17ef2a5', 'release_id': 414157245, 'archive_sha256': 'c7897e5f121df0c08cc2a2a7e993f02fae3ac15fa5ca6f248e1e16cd14ca36a5'},
    'supplement3': {'directory': '/root/minecraft-checkpoints/goal006b-206dfda6468a-0003-20261015T231926513964Z-0aaff2c5', 'release_id': 414396750, 'archive_sha256': '85498f3b9492d4fc37d9778c1e2f7261bf7bddf0ca6d6e20bc5aeb2410592afe'},
    'supplement17': {'directory': '/root/minecraft-checkpoints/goal006c-03340ae00dab-0017-20261016T011456824559Z-2b98786f', 'release_id': 414429861, 'archive_sha256': '6515bc73c442fb8012c2e7349cafcf2a0af9d2c6e6826311aaecb7a39f4c3b55'},
}


def read_json(path):
    return _read_json(Path(path))


class AuditError(RuntimeError):
    pass


def require(condition, code):
    if not condition:
        raise AuditError(code)


def regular(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)) and path.is_file(), 'regular_file_required:' + str(path))
    return path


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')


def sources_from_chain(c_path=C_PLAN):
    """Resolve preceding sources strictly from each immutable parent reference."""
    c = read_json(regular(c_path))
    b_path = regular(c['identity']['source_snapshot']['plan_path'])
    b = read_json(b_path)
    a_path = regular(b['identity']['source_snapshot']['plan_path'])
    a = read_json(a_path)
    p_path = regular(a['identity']['parent']['plan_path'])
    p = read_json(p_path)
    paths = dict(zip(EXPECTED, (p_path, a_path, b_path, regular(c_path))))
    plans = dict(zip(EXPECTED, (p, a, b, c)))
    for label, plan in plans.items():
        require(plan['identity_hash'] == object_hash(plan['identity']) == EXPECTED[label][0], 'source_identity:' + label)
        require(plan['identity']['plan_path'] == str(paths[label]), 'source_not_canonical:' + label)
    for child, parent, key in ((c, b, 'source_snapshot'), (b, a, 'source_snapshot'), (a, p, 'parent')):
        ref = child['identity'][key]
        require(ref['plan_identity_hash'] == parent['identity_hash'] and ref['plan_sha256'] == sha256_file(parent['identity']['plan_path']), 'parent_reference_changed')
    require(c['identity']['original_source_snapshot'] == b['identity']['source_snapshot'], 'original_parent_reference_changed')
    return paths, plans


def _tree(path):
    path = Path(path)
    require(path.is_dir() and not path.is_symlink(), 'immutable_tree_missing:' + str(path))
    result = []
    for p in sorted(path.rglob('*')):
        require(not p.is_symlink(), 'symlink_in_immutable_tree')
        if p.is_file():
            result.append(str(regular(p)))
    return result


def inventory(paths, trees=()):
    tree_map = {str(Path(p).absolute()): _tree(p) for p in trees}
    names = sorted(set(map(str, paths)) | {p for rows in tree_map.values() for p in rows})
    return {'trees': tree_map, 'files': {str(regular(p)): {'sha256': sha256_file(p), 'bytes': Path(p).stat().st_size} for p in names}}


def verify_inventory(saved):
    current = inventory(saved['files'], saved['trees'])
    require(current == saved, 'immutable_inventory_changed')
    return len(current['files'])


def _collect_references(value, paths):
    if isinstance(value, dict):
        # These maps contain explicit paths, never credential discovery.
        for key in ('original_files_sha256', 'implementation_hashes'):
            for name in value.get(key, {}):
                paths.add(str((ROOT / name) if not Path(name).is_absolute() else Path(name)))
        for key, item in value.items():
            if isinstance(item, str) and ('path' in key or key in {'verification_report'}):
                p = Path(item) if Path(item).is_absolute() else ROOT / item
                if p.is_file():
                    paths.add(str(p))
            _collect_references(item, paths)
    elif isinstance(value, list):
        for item in value:
            _collect_references(item, paths)


def process_audit():
    """Report only PIDs and classifications, never argv, environment, or keys."""
    matched = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().decode('utf-8', 'replace').split('\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not args or not args[0]:
            continue
        executable = Path(args[0]).name
        kind = None
        if executable in {'gh', 'gh.exe'} and (('release' in args and any(x in args for x in ('upload', 'create', 'edit', 'delete')))
                or any(x in args for x in ('POST', 'PATCH', 'PUT', 'DELETE'))):
            kind = 'github_mutation'
        if executable.startswith('python'):
            relevant = [a for a in args[1:] if '\n' not in a and ('/' in a or a.startswith('offline_evaluation.')) and
                        any(x in a for x in ('server_v2', 'batch_v1', 'continuation_v1', 'resmoke_v1', 'operations/goal006', 'operations.goal006'))]
            if relevant and any(any(x in a for x in ('cli', 'launch', 'run_generation', 'manage')) for a in relevant):
                kind = 'generation_or_save_entrypoint'
        if kind:
            matched.append({'pid': int(proc.name), 'kind': kind})
    require(not matched, 'active_generation_or_save_process')
    return {'active_generation_or_save_processes': matched, 'host_pid1_comm': Path('/proc/1/comm').read_text().strip(), 'argv_or_environment_reported': False}


@contextmanager
def source_locks(plans):
    c = plans['supplement17']['identity']
    paths = list(dict.fromkeys([*c['parent']['lock_paths'], str(Path(c['enabled_directory']) / 'run.lock'), str(Path(c['authority_path']).with_suffix('.json.lock'))]))
    with ExitStack() as stack:
        for path in paths:
            stream = stack.enter_context(regular(path).open('rb'))
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise AuditError('active_source_lock:' + path) from None
        yield paths


@contextmanager
def no_python_network():
    """Block Python network connections; frozen archive gh subprocesses use GET."""
    old_connect, old_connect_ex = socket.socket.connect, socket.socket.connect_ex
    def denied(*args, **kwargs):
        raise AuditError('python_network_disabled')
    socket.socket.connect = socket.socket.connect_ex = denied
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex = old_connect, old_connect_ex


def _modules():
    from ...batch_v1 import plan as pp, runner as pr
    from ...continuation_v1 import plan as ap, runner as ar
    from ...operations.goal006b import planning as bp, runner as br
    from ...operations.goal006c import planning as cp, runner as cr
    return {'parent160': (pp, pr._load_enabled), 'original781': (ap, ar._load_enabled),
            'supplement3': (bp, br.load_enabled), 'supplement17': (cp, cr.load_enabled)}


def validate_original_sources(paths):
    plans, ledgers, checks = {}, {}, {}
    for label, (planning, load_enabled) in _modules().items():
        print('AUDIT original_validator ' + label, flush=True)
        plan = planning.load_plan(paths[label])
        if hasattr(planning, 'verify_parent'):
            planning.verify_parent(plan['identity']['parent'])
        else:
            planning.verify_source(plan['identity'])
        directory, ledger = load_enabled(plan)
        planning.current_pricing(plan['identity'], fresh=False)
        require(len(ledger['entries']) == EXPECTED[label][1], 'attempt_count_changed:' + label)
        plans[label], ledgers[label] = plan, ledger
        checks[label] = {'status': 'PASS', 'validator': load_enabled.__module__ + '.' + load_enabled.__name__,
                         'attempted': len(ledger['entries']), 'success': sum(e['status'] == 'success' for e in ledger['entries']),
                         'claims': len(list((directory / 'reservations').glob('*.json'))),
                         'halt_reason': ledger.get('halt_reason'), 'historical_pricing_verified': True}
    require(ledgers['original781']['halt_reason'] == 'unknown_usage:auth_failure', 'original_halt_changed')
    require(ledgers['supplement3']['halt_reason'] == 'unknown_usage:http_failure', 'supplemental_halt_changed')
    require(ledgers['supplement17']['halt_reason'] is None, 'unexpected_c_halt')
    return plans, ledgers, checks


def check_metric(metric):
    require(metric.get('metric_version') == 'server_v2.2_strict_translation', 'metric_version_changed')
    alignment = metric.get('alignment', {})
    require(alignment.get('mode') == 'strict_translation' and alignment.get('radius') is None
            and alignment.get('globally_optimal_unrestricted') is True and alignment.get('search_complete') is True
            and alignment.get('status') == 'PASS' and alignment.get('rotation') is False and alignment.get('scale') is False,
            'nonexact_alignment_cache')
    require(alignment.get('all_aligned_metrics_share_shift') is True and alignment.get('clipping') is False, 'inconsistent_metric_shift')


def validate_caches(plans, ledgers, manifest):
    from ...continuation_v1 import aggregate
    semantic = manifest['identity']['protocol_id']
    result = {}
    for label, expected in [('parent160', manifest['identity']['full_schedule'][:160]), ('original781', manifest['identity']['full_schedule'][160:])]:
        plan, ledger = plans[label], ledgers[label]
        path = Path(plan['identity']['enabled_directory']) / 'evaluation.json'
        evaluation = read_json(regular(path))
        aggregate._check_plan(plan, manifest, expected, semantic)
        require(len(evaluation['rows']) == len(expected), 'incomplete_existing_cache')
        rows = aggregate._joined_rows(plan, evaluation, ledger, semantic)
        scores = 0
        for index, row in enumerate(rows):
            entry = ledger['entries'][index] if index < len(ledger['entries']) else None
            if label == 'original781':
                require(row.get('cache_source') == {'plan_identity_hash': plan['identity_hash'], 'entry_hash': object_hash(entry),
                    'gt_hash': object_hash(plan['identity']['gt_by_scene'][row['scene_id']])}, 'cache_source_binding_changed')
            if row.get('metrics') is not None:
                require(entry is not None and entry['status'] == 'success', 'metrics_for_non_success')
                check_metric(row['metrics']); scores += 1
            elif entry and entry['status'] == 'success':
                require(row['status'] == 'ALIGNMENT_RESOURCE_LIMIT' and row.get('evaluation_failure', {}).get('kind') == 'ALIGNMENT_RESOURCE_LIMIT', 'successful_cache_score_missing')
        result[label] = {'status': 'PASS', 'path': str(path), 'sha256': sha256_file(path), 'rows': len(rows),
                         'metric_rows': scores, 'metric_version': 'server_v2.2_strict_translation', 'scores_recomputed': 0}
    return result


def attempt_rows(plans, ledgers):
    from ...continuation_v1.runner import _completed_response
    result, seen = [], set()
    for label, plan in plans.items():
        for record, entry in zip(plan['identity']['schedule'], ledgers[label]['entries']):
            key = (plan['identity_hash'], entry['attempt_id'], entry['schedule_index'], entry['provider'], entry['input_id'])
            require(key not in seen, 'duplicate_attempt_identity'); seen.add(key)
            known = entry.get('usage') is not None
            row = {'source': label, 'plan_identity_hash': plan['identity_hash'], 'attempt_id': entry['attempt_id'],
                   'global_index': entry['schedule_index'], **{k: record.get(k) for k in ('provider', 'input_id', 'scene_id', 'condition', 'description_model', 'input_sha256')},
                   'status': entry['status'], 'failure_kind': entry.get('failure_kind'), 'http_status_code': entry.get('http_status_code'),
                   'usage_known': known, 'completed_request_outcome': _completed_response(plan['identity'], record, entry),
                   'classification': ('operational_unknown' if not known else 'success' if entry['status'] == 'success' else 'model_output_failure'),
                   'intent': entry.get('intent', 'first_attempt'), 'retry': entry['retry'], 'source_attempt_id': entry.get('source_attempt_id'),
                   'request_payload_hash': entry['request_payload_hash'], 'response_model': entry.get('response_model'),
                   'budget_accounted_usd': str(Decimal(str(entry['charged_cost_usd']))),
                   'known_usage_calculated_usd': str(Decimal(str(entry['charged_cost_usd']))) if known else None,
                   'unknown_usage_hold_usd': None if known else str(Decimal(str(entry['charged_cost_usd'])))}
            for kind in ('raw.json', 'parsed.json', 'voxels.json'):
                artifact = entry['artifacts'].get(kind)
                stem = kind.split('.')[0]
                row[stem + '_sha256'] = artifact['sha256'] if artifact else None
                row[stem + '_path'] = str(Path(plan['identity']['enabled_directory']) / artifact['path']) if artifact else None
            result.append(row)
    return result


def accounting(plans, ledgers, rows):
    prior = plans['parent160']['identity']['parent']['usage']
    known = sum((Decimal(r['known_usage_calculated_usd']) for r in rows if r['usage_known']), Decimal(0))
    holds = [r for r in rows if not r['usage_known']]
    hold_sum = sum((Decimal(r['unknown_usage_hold_usd']) for r in holds), Decimal(0))
    total = Decimal(str(prior['cost_usd_upper_bound'])) + known + hold_sum
    retries = prior['retries'] + sum(r['retry'] for r in rows)
    require(len(rows) == 961 and prior['calls'] + len(rows) == 991, 'historical_call_count_mismatch')
    require(total == Decimal('21.56083625'), 'recomputed_budget_mismatch')
    require([(r['global_index'], r['http_status_code'], r['unknown_usage_hold_usd']) for r in holds] == [(940, 401, '0.165536'), (942, 520, '0.165536')], 'unknown_holds_changed')
    old, retry = [r for r in rows if r['global_index'] == 940]
    require(retry['source'] == 'supplement3' and retry['status'] == 'success' and retry['retry'] is True
            and retry['intent'] == 'manual_auth_retry' and retry['source_attempt_id'] == old['attempt_id']
            and retry['request_payload_hash'] == old['request_payload_hash'] and retry['input_id'] == old['input_id'], 'manual_retry_not_bound')
    require(Counter(r['global_index'] for r in rows) == Counter([*range(960), 940]), 'fixed_scope_coverage_mismatch')
    c_cost = sum((Decimal(r['known_usage_calculated_usd']) for r in rows if r['source'] == 'supplement17'), Decimal(0))
    require(c_cost == Decimal('0.34771625'), 'c_cost_mismatch')
    return {'historical_calls': 991, 'prior_smoke_calls': prior['calls'], 'fixed_scope_attempts': len(rows), 'historical_retries': retries,
            'known_usage_calculated_usd_including_smoke': str(Decimal(str(prior['cost_usd_upper_bound'])) + known),
            'unknown_usage_hold_usd': str(hold_sum), 'budget_accounted_usd': str(total), 'supplement17_known_usage_calculated_usd': str(c_cost),
            'unknown_usage_holds': [{k: r[k] for k in ('source', 'attempt_id', 'global_index', 'http_status_code', 'unknown_usage_hold_usd')} for r in holds],
            'invoice_amount_claimed': False, 'hold_release_performed': False, 'research_api_requests_sent': 0}


def validate_recovery(label, report, source):
    expected = CHECKPOINTS[label]
    require(report.get('status') == 'GITHUB_VERIFIED' and report.get('draft') is True and report.get('release_id') == expected['release_id'], 'checkpoint_not_verified:' + label)
    require(report.get('plan_identity_hash') == source['plan_identity_hash'] and report.get('verifiedledger_sha256') == source['ledger_sha256']
            and report.get('attempted_inputs') == EXPECTED[label][1] and report.get('current_attempted_inputs') == EXPECTED[label][1], 'checkpoint_prefix_changed:' + label)
    require(report.get('uploaded_assets') == [] and report.get('research_api_requests_sent') == 0, 'checkpoint_mutation_detected')
    assets = report.get('assets', [])
    require(len(assets) == 6 and len({a['name'] for a in assets}) == 6 and len({a['asset_id'] for a in assets}) == 6, 'checkpoint_six_unique_assets_required')
    for asset in assets:
        require(isinstance(asset['asset_id'], int) and asset['github_digest'] == 'sha256:' + asset['sha256'], 'checkpoint_digest_mismatch')
        path = Path(report['verification_directory']) / 'redownload' / asset['name']
        require(sha256_file(regular(path)) == asset['sha256'] and path.stat().st_size == asset['bytes'], 'checkpoint_download_changed')
    archive = next(a for a in assets if a['name'] == report.get('archive', 'continuation-point-in-time.tar.gz'))
    require(archive['sha256'] == expected['archive_sha256'] and bool(report.get('copied_plan_load_rejection')) and report.get('current_ledger_extends_restored_prefix') is True,
            'checkpoint_restore_provenance_missing')
    return report


def _old160_proof(plans, audit_dir):
    from ...server_v2.checkpoint import restore_archive
    reference = plans['original781']['identity']['checkpoint_sources']
    proof_path = regular(ROOT / reference['verification_report'])
    proof = read_json(proof_path)
    ref = reference['refs']['completed160']
    require(proof['status'] == 'GITHUB_VERIFIED_DRAFT' and proof['release_id'] == ref['release_id'], 'parent160_archive_proof_mismatch')
    download = Path(proof['download_directory'])
    for asset in proof['assets']:
        p = regular(download / asset['name'])
        require(sha256_file(p) == asset['sha256'] and p.stat().st_size == asset['bytes'], 'parent160_archive_asset_changed')
    restored = audit_dir / 'parent160-local-restored'
    restoration = restore_archive(download / ref['name'], restored, ref['sha256'])
    old_restore = Path(proof['restore_directory'])
    require({p.relative_to(restored).as_posix(): sha256_file(p) for p in restored.rglob('*') if p.is_file()} ==
            {p.relative_to(old_restore).as_posix(): sha256_file(p) for p in old_restore.rglob('*') if p.is_file()}, 'parent160_archive_restore_changed')
    canonical = Path(plans['parent160']['identity']['plan_path']).parent
    restored_root = restored / 'original/plan_root'
    require({p.relative_to(canonical).as_posix(): sha256_file(p) for p in canonical.rglob('*') if p.is_file()} ==
            {p.relative_to(restored_root).as_posix(): sha256_file(p) for p in restored_root.rglob('*') if p.is_file()}, 'parent160_restored_live_tree_mismatch')
    authority = regular(plans['parent160']['identity']['authority_path'])
    require(sha256_file(restored / 'original/authority' / authority.name) == sha256_file(authority), 'parent160_restored_authority_mismatch')
    from ...batch_v1 import plan as parent_planning
    try:
        parent_planning.load_plan(restored_root / 'plan.json')
    except Exception as exc:
        require('canonical' in str(exc), 'parent160_unexpected_copy_rejection')
    else:
        raise AuditError('parent160_copied_plan_accepted')
    return {'status': 'PASS', 'proof_path': str(proof_path), 'proof_sha256': sha256_file(proof_path), 'release_id': ref['release_id'],
            'asset_id': ref['asset_id'], 'archive_sha256': ref['sha256'], 'restoration': restoration,
            'fresh_local_safe_restore': True, 'fresh_remote_download': False}


def _output_descriptors(paths, plans, ledgers):
    result = {}
    for label, plan in plans.items():
        i = plan['identity']; enabled = Path(i['enabled_directory']); evaluation = enabled / 'evaluation.json'
        result[label] = {'plan_path': str(paths[label]), 'plan_identity_hash': plan['identity_hash'], 'plan_sha256': sha256_file(paths[label]),
            'enabled_directory': str(enabled), 'ledger_path': str(enabled / 'ledger.json'), 'ledger_sha256': sha256_file(enabled / 'ledger.json'),
            'approval_path': str(enabled / 'approval.json'), 'approval_sha256': sha256_file(enabled / 'approval.json'),
            'authority_path': i['authority_path'], 'authority_sha256': sha256_file(i['authority_path']),
            'evaluation_path': str(evaluation) if evaluation.exists() else None, 'evaluation_sha256': sha256_file(evaluation) if evaluation.exists() else None,
            'attempted': len(ledgers[label]['entries']), 'halt_reason': ledgers[label]['halt_reason'], 'protocol_id': i['protocol_id'], 'code_commit': i['commit']}
    return result


def run_audit(run_root, reports_directory=ROOT / 'reports/server_only/goal007/audit'):
    run_root, reports_directory = Path(run_root).absolute(), Path(reports_directory).absolute()
    audit_dir = run_root / 'audit'
    audit_dir.mkdir(parents=True, exist_ok=True)
    require(not (audit_dir / 'inventory_before.json').exists(), 'new_audit_directory_required')
    paths, initial_plans = sources_from_chain()
    manifest_path = regular(initial_plans['original781']['identity']['manifest_path'])
    manifest = read_json(manifest_path)
    reference = initial_plans['original781']['identity']['checkpoint_sources']
    old160_proof = read_json(ROOT / reference['verification_report'])
    trees = [p.parent for p in paths.values()] + [Path(c['directory']) for c in CHECKPOINTS.values()]
    trees += [Path(old160_proof['download_directory']), Path(old160_proof['restore_directory'])]
    refs = set(map(str, paths.values())) | {str(manifest_path), str(ROOT / reference['verification_report'])}
    for p in initial_plans.values():
        _collect_references(p, refs)
        refs.add(p['identity']['authority_path'])
    _collect_references(manifest, refs)
    for row in manifest['identity']['source_rows']:
        for item in row['source_files']:
            path = ROOT / item['path']; refs.add(str(path))
            require(sha256_file(regular(path)) == item['sha256'], 'immutable_dataset_source_changed')
    for descriptor in manifest['identity']['gt_by_scene'].values():
        refs.update(str(Path(descriptor['building_directory']) / 'gt' / name) for name in ('voxels.npy', 'bbox.json'))
    with source_locks(initial_plans) as locks, no_python_network():
        refs.update(locks)
        process_before = process_audit()
        before = inventory(refs, trees)
        write_new(audit_dir / 'inventory_before.json', before)
        try:
            plans, ledgers, original_checks = validate_original_sources(paths)
            require(manifest['identity_hash'] == object_hash(manifest['identity']), 'manifest_identity_changed')
            caches = validate_caches(plans, ledgers, manifest)
            descriptors = _output_descriptors(paths, plans, ledgers)
            rows = attempt_rows(plans, ledgers)
            costs = accounting(plans, ledgers, rows)
            parent_archive = _old160_proof(plans, audit_dir)
            from ...operations.goal006a import archive as archive_a
            from ...operations.goal006c import archive as archive_c
            github = {}
            for label, checkpoint in CHECKPOINTS.items():
                print('AUDIT read_only_github_restore ' + label, flush=True)
                module = archive_a if label == 'original781' else archive_c
                recovered = module.recover(checkpoint['directory'], audit_dir / 'github' / label, release_id=checkpoint['release_id'], allow_create=False)
                github[label] = validate_recovery(label, recovered, descriptors[label])
                write_new(reports_directory / ('github_' + label + '.json'), github[label])
            count = verify_inventory(before)
            process_after = process_audit()
            write_new(audit_dir / 'inventory_after.json', before)
            with (audit_dir / 'attempts_audit.csv').open('x', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            write_new(audit_dir / 'attempts_audit.json', rows)
            write_new(audit_dir / 'cost_validation.json', costs)
            write_new(audit_dir / 'sources.json', descriptors)
            report = {'version': VERSION, 'status': 'PASS', 'verified_at_utc': utc_now(), 'run_root': str(run_root),
                'sources': descriptors, 'manifest_path': str(manifest_path), 'manifest_sha256': sha256_file(manifest_path),
                'original_validators': original_checks, 'existing_evaluation_caches': caches, 'parent160_checkpoint': parent_archive,
                'github_checkpoints': github, 'cost_validation': costs, 'process_before': process_before, 'process_after': process_after,
                'idle_locks_acquired_readonly': locks, 'immutable_file_count': count, 'immutable_inventories_equal': True,
                'inventory_path': str(audit_dir / 'inventory_before.json'), 'inventory_sha256': sha256_file(audit_dir / 'inventory_before.json'),
                'attempts_audit_path': str(audit_dir / 'attempts_audit.csv'), 'attempts_audit_json_path': str(audit_dir / 'attempts_audit.json'),
                'research_api_requests_sent': 0, 'github_mutations': 0, 'quality_scores_computed': 0,
                'audit_code_sha256': sha256_file(__file__)}
            for name in ('attempts_audit.csv', 'attempts_audit.json', 'cost_validation.json', 'sources.json'):
                report.setdefault('output_sha256', {})[str(audit_dir / name)] = sha256_file(audit_dir / name)
            write_new(audit_dir / 'audit_gate.json', report)
            write_new(reports_directory / 'audit_gate.json', report)
            # Compact evidence table is safe for source control; raw stays external.
            for name in ('attempts_audit.csv', 'cost_validation.json', 'sources.json'):
                target = reports_directory / name
                with target.open('xb') as stream:
                    stream.write((audit_dir / name).read_bytes())
            return report
        except Exception as exc:
            try:
                unchanged = verify_inventory(before) == len(before['files'])
            except Exception:
                unchanged = False
            failure = {'status': 'STOP', 'error_type': type(exc).__name__, 'safe_reason': str(exc) if isinstance(exc, AuditError) else getattr(exc, 'report', None),
                       'immutable_inventories_equal': unchanged, 'research_api_requests_sent': 0}
            write_new(audit_dir / 'audit_failure.json', failure)
            raise


def verify_gate(report_or_path):
    report = read_json(report_or_path) if isinstance(report_or_path, (str, Path)) else report_or_path
    require(report.get('version') == VERSION and report.get('status') == 'PASS' and report.get('research_api_requests_sent') == 0, 'audit_gate_not_passed')
    path = regular(report['inventory_path'])
    require(sha256_file(path) == report['inventory_sha256'], 'audit_inventory_proof_changed')
    verify_inventory(read_json(path))
    require(sha256_file(report['manifest_path']) == report['manifest_sha256'], 'manifest_changed_after_audit')
    for path, digest in report['output_sha256'].items():
        require(sha256_file(regular(path)) == digest, 'audit_output_changed')
    require(set(report['sources']) == set(EXPECTED) and set(report['github_checkpoints']) == set(CHECKPOINTS), 'incomplete_audit_sources')
    paths, plans = sources_from_chain()
    for label, source in report['sources'].items():
        identity = plans[label]['identity']
        require(source['plan_path'] == str(paths[label]) and source['plan_identity_hash'] == EXPECTED[label][0]
                and source['plan_sha256'] == sha256_file(paths[label]) and source['enabled_directory'] == identity['enabled_directory'],
                'source_descriptor_changed')
        enabled = Path(identity['enabled_directory'])
        for field, path in [('ledger', enabled / 'ledger.json'), ('approval', enabled / 'approval.json'),
                            ('authority', Path(identity['authority_path']))]:
            require(source[field + '_path'] == str(path) and source[field + '_sha256'] == sha256_file(regular(path)), 'source_control_descriptor_changed')
        evaluation = enabled / 'evaluation.json'
        require(source['evaluation_path'] == (str(evaluation) if evaluation.exists() else None)
                and source['evaluation_sha256'] == (sha256_file(evaluation) if evaluation.exists() else None), 'cache_descriptor_changed')
        require(source['code_commit'] == identity['commit'] and source['attempted'] == EXPECTED[label][1]
                and source['halt_reason'] == read_json(enabled / 'ledger.json')['halt_reason'], 'source_summary_descriptor_changed')
    for label, recovered in report['github_checkpoints'].items():
        validate_recovery(label, recovered, report['sources'][label])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--reports-directory', default=str(ROOT / 'reports/server_only/goal007/audit'))
    args = parser.parse_args(argv)
    # Deliberately no credential inspection, generation import, or pricing probe.
    report = run_audit(args.run_root, args.reports_directory)
    print(json.dumps({'status': report['status'], 'immutable_files': report['immutable_file_count'], 'gate': str(Path(args.run_root) / 'audit/audit_gate.json')}))


if __name__ == '__main__':
    main()
