"""Bounded read-only input preservation for Goal015B; no execution/API imports.

Snapshots contain hashes, source roles, and absent execution controls. They do
not contain prompt text, responses, credentials, or a new paid authority.
"""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

REPO = Path('/root/workspace/Minecraft-view-dataset')
PARENT = Path('/root/minecraft-runs/goal015-20261017T170546Z')
CHILD = Path('/root/minecraft-runs/goal015a-20261019T141842Z-13cfbc79')
EVALUATION = PARENT / 'evaluations/goal015a-parent72-20261019T141842Z-13cfbc79/evaluation_80.json'
PARENT_ID = '5b80fdef342dd2b9f68b3c2bec2d3d83af2731c3b5a9c8940959846ea11c68ce'
PARENT_SHA = '6533090e07090b7279f0383937631c609b17c84c922f8cd8deadcd2b9ea22257'
LEDGER_SHA = '6c2a6d2412492a1cac722de7cfa7bff491dd64afcfe484cb0c8790db4029673a'
EVALUATION_SHA = 'e430c61b0eaa1d6108158636fb4cbff09edf9cc481d6766c2445460ba3ce2420'
ABSENT = tuple(CHILD / name for name in ('enabled', 'approval.json', 'authority.json', 'ledger.json',
    'execution.lock', 'completion.lock', 'enabled/approval.json', 'enabled/authority.json', 'enabled/ledger.json'))


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def safe(path):
    path = Path(path).absolute()
    require('..' not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
            'Non-symlink absolute path required: ' + str(path))
    return path


def descriptor(path):
    path = safe(path); digest = hashlib.sha256(); size = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block); size += len(block)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': size}


def strict_json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, 'Duplicate JSON metadata key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON metadata')))


def write_exclusive(path, data):
    path = safe(path)
    relative = path.relative_to('/root/minecraft-runs')
    require(relative.parts[0].startswith('goal015b-'), 'Only new Goal015B run outputs allowed')
    with path.open('xb') as stream:
        stream.write(canonical(data) + b'\n'); stream.flush(); os.fsync(stream.fileno())


class Inputs:
    def __init__(self):
        self.files = {}; self.roles = {}

    def add(self, path, role, expected=None):
        name = str(safe(path))
        if name not in self.files:
            self.files[name] = descriptor(name)
        actual = self.files[name]
        if expected is not None:
            for key in ('bytes', 'sha256'):
                if key in expected:
                    require(actual[key] == expected[key], 'Bound input changed: ' + name)
        self.roles.setdefault(name, [])
        if role not in self.roles[name]:
            self.roles[name].append(role)
        return actual

    def read(self, path, role, expected=None):
        measured = self.add(path, role, expected)
        raw = safe(path).read_bytes()
        require(len(raw) == measured['bytes'] and hashlib.sha256(raw).hexdigest() == measured['sha256'],
                'Input changed between hash and read: ' + str(path))
        return strict_json(raw)

    def add_ref(self, ref, role):
        return self.add(ref['path'], role, ref)


def walk_refs(value):
    if isinstance(value, dict):
        if {'path', 'sha256', 'bytes'} <= value.keys():
            yield {key: value[key] for key in ('path', 'sha256', 'bytes')}
        else:
            for child in value.values():
                yield from walk_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_refs(child)


@contextmanager
def parent_lock():
    """Cooperative read-only lock; never creates or rewrites the lock file."""
    with safe(PARENT / 'execution.lock').open('rb') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Parent execution lock is busy; stop diagnostics') from None
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def process_comm():
    rows = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            name = (entry / 'comm').read_text().strip()
        except (OSError, UnicodeError):
            continue
        if any(word in name.lower() for word in ('python', 'codex', 'tmux', 'pytest')):
            rows.append({'pid': int(entry.name), 'comm': name})
    return sorted(rows, key=lambda row: row['pid'])


def _snapshot(inputs, absent=()):
    for path in absent:
        require(not os.path.lexists(path), 'Child execution control exists: ' + str(path))
    return {'schema': 'goal015b_protected_inputs_v1', 'created_at_utc': utc(),
            'files': inputs.files, 'roles': inputs.roles, 'required_absent_paths': [str(p) for p in absent],
            'files_count': len(inputs.files), 'unique_file_bytes': sum(ref['bytes'] for ref in inputs.files.values()),
            'inventory_sha256': hashlib.sha256(canonical(inputs.files)).hexdigest(),
            'scope': 'Explicitly read parent/child/source evidence only; no environment, CLIP, or whole-repository inventory.'}


def snapshot_paths(paths, output_path):
    """Supplemental inputs: call before using the selected old replay/source files."""
    inputs = Inputs()
    for path in paths:
        inputs.add(path, 'supplemental_selected_input')
    result = _snapshot(inputs); write_exclusive(output_path, result)
    return result


def capture_snapshot(run_root, extra_paths=()):
    start = time.monotonic(); cpu = time.process_time(); inputs = Inputs(); run_root = safe(run_root)
    with parent_lock():
        plan = inputs.read(PARENT / 'plan.json', 'parent_plan', {'sha256': PARENT_SHA})
        require(plan['plan_id'] == PARENT_ID == hashlib.sha256(canonical({k:v for k,v in plan.items() if k != 'plan_id'})).hexdigest(), 'Parent identity differs')
        require(plan.get('fixture_only') is not True and plan['proposed_budget_usd'] == '5.00', 'Parent scope/cap differs')
        ledger = inputs.read(PARENT / 'enabled/ledger.json', 'parent_ledger', {'sha256': LEDGER_SHA})
        approval = inputs.read(PARENT / 'enabled/approval.json', 'parent_approval')
        authority = inputs.read(PARENT / 'enabled/authority.json', 'parent_authority')
        require(approval['status'] == authority['status'] == 'APPROVED' and approval['approved_budget_usd'] == '5.00', 'Existing parent authority differs')
        require(ledger['plan_id'] == approval['plan_id'] == authority['plan_id'] == PARENT_ID, 'Parent authority/ledger identity differs')
        inputs.add_ref(authority['plan'], 'authority_bound_plan'); inputs.add_ref(authority['approval'], 'authority_bound_approval')
        require(ledger['status'] == 'SAFE_STOP' and ledger['halt']['reason'] == 'next complete paired block exceeds approved cap', 'Parent stop state differs')
        require(len(ledger['calls']) == len(plan['schedule']) == 80, 'Expected original80 rows')
        counts = Counter(row['state'] for row in ledger['calls'])
        require(dict(counts) == {'SETTLED': 72, 'NOT_SENT': 8}, 'Actual parent call states differ')
        cost = sum((Decimal(row['known_cost_usd']) for row in ledger['calls'] if row['usage_known']), Decimal(0))
        require(cost == Decimal('4.43765700'), 'Parent carried cost differs')
        for original, row in zip(plan['schedule'], ledger['calls']):
            require(all(row.get(k) == v for k,v in original.items()), 'Parent schedule mapping differs')
            inputs.add_ref(row['payload'], 'parent_payload')
            if row['state'] == 'SETTLED':
                require(row['usage_known'] is True and row['send_status'] == 'POST_RETURNED' and row['response_received'] is True, 'Settled delivery state differs')
                for key in ('raw', 'response_text_path'):
                    ref = row[key]
                    suffix = '.raw.json' if key == 'raw' else '.text.txt'
                    require(Path(ref['path']) == PARENT/'enabled/responses'/(row['call_id'] + suffix), 'Response escaped canonical call mapping')
                    inputs.add_ref(ref, 'parent_' + key)
            else:
                require(row['raw'] is None and row['response_text_path'] is None and not row['usage_known'], 'Unsent response evidence differs')
        require(len(plan['implementation_hashes']) == 55, 'Expected frozen55 collection')
        for path, expected in plan['implementation_hashes'].items():
            inputs.add(path, 'parent_frozen55', {'sha256': expected})
        for key in ('specification', 'schedule_artifact', 'pricing_evidence'):
            inputs.add_ref(plan[key], 'parent_' + key)
        inputs.add(PARENT/'COST_ESTIMATE.json', 'historical_cost_extrapolation')
        evaluation = inputs.read(EVALUATION, 'parent_evaluation', {'sha256': EVALUATION_SHA})
        require(evaluation['plan_binding']['plan'] == inputs.files[str(PARENT/'plan.json')]
                and evaluation['plan_binding']['ledger'] == inputs.files[str(PARENT/'enabled/ledger.json')], 'Evaluation source identity differs')
        require([r['call_id'] for r in evaluation['rows']] == [r['call_id'] for r in plan['schedule']], 'Evaluation frame differs')
        for ref in walk_refs(evaluation):
            inputs.add_ref(ref, 'parent_evaluation_artifact')
        inputs.add(EVALUATION.parent/'summary.json', 'parent_evaluation_summary')
        child = inputs.read(CHILD/'plan.json', 'disabled_child_plan')
        require(child['api_allowed'] is False and child['budget_approved'] is False and child['parent_plan_id'] == PARENT_ID
                and child['schedule'] == plan['schedule'][72:] and child['proposed_budget_usd'] == '7.00', 'Disabled child scope differs')
        require(child['plan_id'] == hashlib.sha256(canonical({k:v for k,v in child.items() if k != 'plan_id'})).hexdigest(), 'Child identity differs')
        for path, expected in child['implementation_hashes'].items():
            inputs.add(path, 'child_frozen_control', {'sha256': expected})
        inputs.add_ref(child['specification'], 'child_specification')
        for key in ('github_verification', 'checkpoint', 'evaluation_summary', 'execution_lock'):
            inputs.add_ref(child['parent'][key], 'prior_save_or_parent_state')
        pricing = inputs.read(child['pricing_evidence']['path'], 'expired_execution_pricing', child['pricing_evidence'])
        original_pricing = inputs.read(plan['pricing_evidence']['path'], 'original_pricing', plan['pricing_evidence'])
        for p in sorted((PARENT/'enabled/pricing_refresh').glob('*.json')):
            accepted = inputs.read(p, 'accepted_execution_pricing_reference'); inputs.add_ref(accepted['evidence'], 'accepted_pricing_evidence')
        refresh_dir = Path(child['pricing_evidence']['path']).parent
        inputs.add(refresh_dir/'start-reviewed.sh', 'actual_parent_launcher_source')
        for p in sorted(refresh_dir.glob('start-reviewed*.log')):
            inputs.add(p, 'parent_launcher_log')
        required = ['goal015a/README.md', 'goal015a/PARENT_EVALUATION_COUNTS.json', 'goal015a/PARENT_SAVE_AUDIT.json',
                    'goal015a/EXECUTION_BINDING.json', 'goal015a/MANIFEST.json', 'goal015/EXPERIMENT_SPEC.md',
                    'goal015/EXECUTION_BINDING.json', 'goal015/schedule_80.json']
        for relative in required:
            inputs.add(REPO/'reports/server_only'/relative, 'required_report')
        for path in extra_paths:
            inputs.add(path, 'supplemental_selected_input')
        snapshot = _snapshot(inputs, ABSENT)
        now = datetime.now(timezone.utc)
        pricing_states = []
        for label, item in [('original', original_pricing), ('accepted_refresh', pricing)]:
            acquired = datetime.fromisoformat(item['retrieved_at_utc'].replace('Z', '+00:00'))
            expiry = acquired + timedelta(seconds=plan['pricing_contract']['freshness_seconds'])
            pricing_states.append({'evidence': label, 'retrieved_at_utc': acquired.isoformat(), 'expires_at_utc': expiry.isoformat(), 'expired_at_diagnostic': now > expiry})
        state = {'schema': 'goal015b_state_before_v1', 'checked_at_utc': utc(), 'status': 'PASS',
                 'parent_plan_id': PARENT_ID, 'parent_status': ledger['status'], 'calls': dict(counts),
                 'unknown_usage_calls': 0, 'parent_known_cost_usd': str(cost), 'parent_approved_cap_usd': '5.00',
                 'child_plan_id': child['plan_id'], 'child_approved': False, 'child_proposed_combined_cap_usd': '7.00',
                 'child_execution_controls_absent': [str(p) for p in ABSENT], 'parent_lock_read_only_flock_available': True,
                 'process_comm_only': process_comm(), 'process_inspection_limitation': 'Names only; no arguments or environment inspected. Nonblocking parent lock and absent child controls are the execution exclusion evidence.',
                 'pricing_states': pricing_states, 'pricing_refresh_performed': False,
                 'parent_frozen_code_files': 55, 'parent_raw_files': 72, 'parent_text_files': 72,
                 'cwd': str(Path.cwd()), 'head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO).decode().strip(),
                 'branch': subprocess.check_output(['git','branch','--show-current'],cwd=REPO).decode().strip(),
                 'git_status_porcelain': subprocess.check_output(['git','status','--porcelain=v1'],cwd=REPO).decode(),
                 'research_api_requests_sent': 0, 'network_requests': 0, 'old_paths_written': 0,
                 'cpu_seconds': time.process_time()-cpu, 'elapsed_seconds': time.monotonic()-start,
                 'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        require(state['cpu_seconds'] < 120, 'Preservation CPU limit reached')
        write_exclusive(run_root/'PROTECTED_INPUTS_BEFORE.json', snapshot)
        state['snapshot'] = descriptor(run_root/'PROTECTED_INPUTS_BEFORE.json')
        write_exclusive(run_root/'STATE_BEFORE.json', state)
    return snapshot


def finalize_snapshot(before_path, output_path):
    started = time.monotonic(); cpu = time.process_time(); before_ref = descriptor(before_path)
    before = strict_json(safe(before_path).read_bytes())
    require(before['schema'] == 'goal015b_protected_inputs_v1', 'Wrong input snapshot schema')
    require(hashlib.sha256(canonical(before['files'])).hexdigest() == before['inventory_sha256'], 'Snapshot internal hash mismatch')
    failures = []
    with parent_lock():
        for path, expected in before['files'].items():
            try:
                if descriptor(path) != expected:
                    failures.append({'path': path, 'reason': 'descriptor_changed'})
            except (OSError, ValueError):
                failures.append({'path': path, 'reason': 'missing_or_unsafe'})
        for path in before['required_absent_paths']:
            if os.path.lexists(path):
                failures.append({'path': path, 'reason': 'execution_control_appeared'})
    result = {'schema': 'goal015b_protected_inputs_after_v1', 'checked_at_utc': utc(), 'before': before_ref,
              'status': 'PASS' if not failures else 'FAIL', 'checked_files': len(before['files']),
              'checked_bytes': sum(item['bytes'] for item in before['files'].values()),
              'checked_absent_paths': len(before['required_absent_paths']), 'failures': failures,
              'cpu_seconds': time.process_time()-cpu, 'elapsed_seconds': time.monotonic()-started,
              'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'network_requests': 0, 'research_api_requests_sent': 0, 'old_paths_written': 0}
    write_exclusive(output_path, result)
    require(not failures, 'Protected Goal015B inputs changed; see final receipt')
    return result


def transport_provenance(run_root):
    """Inspect saved launch/transport evidence without importing any API adapter."""
    start = time.monotonic(); cpu = time.process_time(); run_root = safe(run_root)
    before_path = run_root/'PROTECTED_INPUTS_BEFORE.json'
    before = strict_json(before_path.read_bytes())
    def read_bound(path):
        path = safe(path); expected = before['files'][str(path)]; raw = path.read_bytes()
        require(len(raw) == expected['bytes'] and hashlib.sha256(raw).hexdigest() == expected['sha256'],
                'Transport review source changed: ' + str(path))
        return raw
    def metadata(path):
        return strict_json(read_bound(path))
    plan = metadata(PARENT/'plan.json'); child = metadata(CHILD/'plan.json')
    ledger = metadata(PARENT/'enabled/ledger.json'); approval = metadata(PARENT/'enabled/approval.json')
    for goal, root, value in [('goal015', PARENT, plan), ('goal015a', CHILD, child)]:
        registry = metadata(REPO/'reports/server_only'/goal/'EXECUTION_BINDING.json')
        require(registry['plan_id'] == value['plan_id'] and registry['canonical_root'] == str(root)
                and registry['plan'] == before['files'][str(root/'plan.json')], 'Prepared registry differs')
    for key in ('plan', 'ledger', 'approval', 'authority'):
        require(child['parent'][key] == before['files'][str(PARENT/('plan.json' if key == 'plan' else 'enabled/'+key+'.json'))],
                'Child bound parent evidence differs')
    code_paths = {'runtime': REPO/'offline_evaluation/experiments/goal015/runtime.py',
                  'cli': REPO/'offline_evaluation/experiments/goal015/cli.py',
                  'adapter': REPO/'offline_evaluation/experiments/goal015/adapter.py',
                  'providers': REPO/'offline_evaluation/server_v2/providers.py'}
    code = {key: read_bound(path).decode() for key,path in code_paths.items()}
    required_fragments = {
        'runtime': ["require(fixture or (adapter is None and confirmation is None and checkpoint_callback is None),'mock hooks forbidden for real plan')",
                    "payload=bound(row['payload']);adapter.validate_payload(row['builder'],payload)",
                    "raw=adapter.send_request(row['builder'],payload,api_allowed=True,permission=True,timeout=plan['timeout_seconds'])"],
        'cli': ["result=runtime.execute(args.plan,resume=args.command=='resume',refresh_evidence=evidence)"],
        'adapter': ['transport=transport or http_transport', 'urllib.request.build_opener(providers._NoRedirect())'],
        'providers': ['data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")', 'method="POST"']}
    source_boundaries = {}
    for name, fragments in required_fragments.items():
        lines = code[name].splitlines(); evidence = []
        for fragment in fragments:
            matches = [number for number,line in enumerate(lines,1) if fragment in line]
            require(len(matches) == 1, 'Frozen transport boundary not found: ' + name)
            evidence.append({'line': matches[0], 'fragment': fragment})
        source_boundaries[name] = {'source': before['files'][str(code_paths[name])], 'evidence': evidence}
    refresh = Path(child['pricing_evidence']['path']).parent
    launcher_path = refresh/'start-reviewed.sh'; launcher = read_bound(launcher_path).decode()
    require(launcher.count('offline_evaluation.experiments.goal015.cli start') == 1, 'Original start command differs')
    logs = []
    for path, roles in before['roles'].items():
        if 'parent_launcher_log' not in roles:
            continue
        lines = read_bound(path).decode().splitlines(); saved_results = []
        for line in lines:
            if line.startswith('{'):
                try:
                    item = strict_json(line)
                except ValueError:
                    continue
                if isinstance(item, dict) and 'status' in item:
                    saved_results.append({key:item[key] for key in ('status','plan_id','calls_settled','calls_not_sent',
                        'calls_uncertain','fixture_only','budget','failure') if key in item})
        logs.append({'artifact': before['files'][path], 'preflight_pass_recorded': any('Read-only preflight passed' in line for line in lines),
                     'configured_start_invocation_recorded': any(line.startswith('Invoking the existing start command once.') for line in lines),
                     'exit_lines': [line for line in lines if line.startswith(('CLI process exit status:', 'Wrapper exit status:'))],
                     'saved_runtime_results': saved_results})
    sent = [row for row in ledger['calls'] if row['state'] == 'SETTLED']
    models = Counter(row['response_model'] for row in sent)
    ids = [row.get('response_id') for row in sent]
    id_prefixes = Counter('resp_' if isinstance(value,str) and value.startswith('resp_') else
                          'msg_' if isinstance(value,str) and value.startswith('msg_') else 'other_or_missing' for value in ids)
    marker_ids = [row['call_id'] for row in sent if any(marker in str(row.get('response_id','')).lower() for marker in ('fixture','mock'))]
    payload_serializations = []
    for row in plan['schedule']:
        payload = metadata(row['payload']['path'])
        serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
        payload_serializations.append({'call_id':row['call_id'], 'payload':row['payload'],
            'requested_model':payload['model'], 'derived_transport_body_sha256':hashlib.sha256(serialized).hexdigest(),
            'derived_transport_body_bytes':len(serialized), 'observed_wire_body_hash':None})
    save = metadata(REPO/'reports/server_only/goal015a/PARENT_SAVE_AUDIT.json')
    require(save['github']['status'] == 'GITHUB_VERIFIED' and save['github']['release_id'] == 417384305
            and save['github']['asset_id'] == 656494745 and save['github']['restoration']['files'] == 297
            and save['parent_evaluation'] == before['files'][str(EVALUATION)], 'Prior verified save identity differs')
    result = {'schema':'goal015b_transport_provenance_v1', 'checked_at_utc':utc(), 'status':'PASS_WITH_DELIVERY_LIMITATION',
        'snapshot':descriptor(before_path), 'source_boundaries':source_boundaries,
        'production_plan_fixture_only':plan.get('fixture_only') is True,
        'approval_method':approval['approval_method'], 'approved_at_utc':approval['approved_at_utc'],
        'runtime_mock_hooks_forbidden_for_real_plan':True, 'cli_exposes_mock_hooks':False,
        'default_transport':'stdlib urllib, one POST, no redirects/retries/model fallback',
        'low_level_transport_injection_exists':True,
        'low_level_injection_limitation':'The adapter/providers expose a transport argument for isolated tests. Frozen CLI/runtime do not supply it for this real plan. Local source and artifacts cannot prove absence of arbitrary external monkeypatching.',
        'launcher':before['files'][str(launcher_path)],
        'launcher_configured_argv':[plan['existing_execution_environment']['python'],'-B','-m',
            'offline_evaluation.experiments.goal015.cli','start','--plan',str(PARENT/'plan.json'),
            '--pricing-evidence',child['pricing_evidence']['path']],
        'launcher_arguments_evidence':'Configured source plus saved wrapper result logs; no independent process-exec audit or remote request-body log.',
        'launcher_logs':logs, 'ledger_delivery_counts':dict(Counter(row['send_status'] for row in ledger['calls'])),
        'settled_http_status_counts':dict(Counter(str(row['http_status']) for row in sent)),
        'settled_response_model_counts':dict(models), 'response_id_prefix_counts':dict(id_prefixes),
        'unique_settled_response_ids':len(set(ids)), 'unique_raw_sha256':len({row['raw']['sha256'] for row in sent}),
        'unique_saved_text_sha256':len({row['response_text_path']['sha256'] for row in sent}),
        'fixture_or_mock_marker_in_response_ids':marker_ids,
        'canonical_call_raw_text_mapping_verified':True,
        'payload_transport_serialization':payload_serializations,
        'delivery_conclusion':'Saved canonical payloads are hash-bound to schedule/ledger; frozen code validates them then constructs a POST from their parsed JSON. Local HTTP200 envelopes and IDs plus logs are consistent with that path. No independent remote delivery attestation is present; exact transmitted body hashes were not recorded.',
        'fixture_contamination_conclusion':'No fixture-plan, runtime injected-hook, noncanonical response-path, or fixture/mock response-ID evidence was found. This is bounded local provenance evidence, not proof based solely on string absence.',
        'archive_restore_proof_reused':{'source':before['files'][str(REPO/'reports/server_only/goal015a/PARENT_SAVE_AUDIT.json')],
            'release_id':417384305,'asset_id':656494745,'archive_sha256':save['archive']['sha256'],
            'previously_restored_files':297,'fresh_download_or_restore_performed':False},
        'network_requests':0,'research_api_requests_sent':0,'old_paths_written':0,
        'environment_or_key_values_inspected':False,'cpu_seconds':time.process_time()-cpu,
        'elapsed_seconds':time.monotonic()-start,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_exclusive(run_root/'TRANSPORT_PROVENANCE.json', result)
    return result


def consolidate_closure(run_root, *, new_code_change_note):
    """One bounded union hash pass; distinguish evolving new diagnostics from old inputs."""
    started = time.monotonic(); cpu = time.process_time(); run_root = safe(run_root)
    old = {}; groups = {}; receipts = {}; new_observations = []; failures = []
    def read_receipt(path, *, lines=False):
        ref = descriptor(path); raw = safe(path).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Receipt changed during closure read')
        receipts[str(path)] = ref
        return [strict_json(line) for line in raw.splitlines() if line] if lines else strict_json(raw)
    def is_new(path):
        return any(Path(path).is_relative_to(root) for root in
                   (run_root, REPO/'offline_evaluation/analysis/goal015b', REPO/'reports/server_only/goal015b'))
    def include(ref, group):
        ref = {key:ref[key] for key in ('path','sha256','bytes')}; name = ref['path']
        if is_new(name):
            new_observations.append({'group':group,'recorded':ref}); return
        if name in old:
            require(old[name] == ref, 'Conflicting recorded old-input hashes: ' + name)
        old[name] = ref; groups.setdefault(group, set()).add(name)
    main = read_receipt(run_root/'PROTECTED_INPUTS_BEFORE.json')
    supplement = read_receipt(run_root/'ROOT_AUDIT_SUPPLEMENTAL_INPUTS_BEFORE.json')
    for group, item in [('main_snapshot',main), ('root_audit_supplement',supplement)]:
        require(item['schema'] == 'goal015b_protected_inputs_v1' and hashlib.sha256(canonical(item['files'])).hexdigest() == item['inventory_sha256'], 'Snapshot integrity failed')
        for ref in item['files'].values(): include(ref, group)
    phase4 = read_receipt(run_root/'phase4/INPUT_SNAPSHOTS.jsonl', lines=True)
    for ref in phase4: include(ref, 'phase4_recorded_inputs')
    coverage = {}
    for relative in ('audit/INPUT_DESCRIPTORS.json','verified/audit/INPUT_DESCRIPTORS.json'):
        refs = read_receipt(run_root/relative); missing = []; mismatches = []
        for name, ref in refs.items():
            if name not in old: missing.append(name)
            elif old[name] != ref: mismatches.append(name)
        coverage[relative] = {'descriptors':len(refs),'missing':missing,'mismatched':mismatches}
        require(not missing and not mismatches, 'Audit read inputs are not covered by preserved old-input union')
    phase2 = read_receipt(run_root/'phase2/IMPORTED_OLD_SOURCES.json')
    phase2_refs = list(walk_refs(phase2))
    require(all(old.get(ref['path']) == ref for ref in phase2_refs), 'Phase2 imported old source is not covered')
    for entry in new_observations:
        entry['current'] = descriptor(entry['recorded']['path'])
        entry['matches_recorded_read_version'] = entry['current'] == entry['recorded']
    changed_new = [entry for entry in new_observations if not entry['matches_recorded_read_version']]
    require(not changed_new or bool(new_code_change_note), 'Changed new-code read versions need an explanation')
    with parent_lock():
        for path, expected in old.items():
            try:
                if descriptor(path) != expected:
                    failures.append({'path':path,'reason':'old_input_changed'})
            except (OSError, ValueError):
                failures.append({'path':path,'reason':'old_input_missing_or_unsafe'})
        for path in main['required_absent_paths']:
            if os.path.lexists(path): failures.append({'path':path,'reason':'execution_control_appeared'})
        plan = strict_json((PARENT/'plan.json').read_bytes()); ledger = strict_json((PARENT/'enabled/ledger.json').read_bytes())
        approval = strict_json((PARENT/'enabled/approval.json').read_bytes()); child = strict_json((CHILD/'plan.json').read_bytes())
        states = Counter(row['state'] for row in ledger['calls'])
        cost = sum((Decimal(row['known_cost_usd']) for row in ledger['calls'] if row['usage_known']), Decimal(0))
        state = {'parent_plan_id':plan['plan_id'], 'parent_status':ledger['status'], 'halt':ledger['halt'],
                 'parent_state_counts':dict(states), 'parent_known_cost_usd':str(cost),
                 'parent_proposed_cap_usd':plan['proposed_budget_usd'], 'parent_approved_cap_usd':approval['approved_budget_usd'],
                 'unknown_usage_calls':sum(row['state'] not in ('SETTLED','NOT_SENT') for row in ledger['calls']),
                 'child_plan_id':child['plan_id'], 'child_api_allowed':child['api_allowed'],
                 'child_budget_approved':child['budget_approved'], 'child_proposed_combined_cap_usd':child['proposed_budget_usd'],
                 'child_execution_controls_absent':not any(os.path.lexists(path) for path in main['required_absent_paths']),
                 'parent_execution_lock_available_opened_rb':True, 'parent_or_child_lock_created':False}
        state['matches_required_state'] = (plan['plan_id'] == PARENT_ID and dict(states) == {'SETTLED':72,'NOT_SENT':8}
            and cost == Decimal('4.43765700') and plan['proposed_budget_usd'] == approval['approved_budget_usd'] == '5.00'
            and ledger['status'] == 'SAFE_STOP' and ledger['halt']['reason'] == 'next complete paired block exceeds approved cap'
            and child['api_allowed'] is False and child['budget_approved'] is False and child['proposed_budget_usd'] == '7.00'
            and state['child_execution_controls_absent'])
        if not state['matches_required_state']: failures.append({'reason':'execution_state_changed'})
    unstaged = subprocess.check_output(['git','diff','--name-only'],cwd=REPO).decode().splitlines()
    staged = subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO).decode().splitlines()
    old_git_diffs = [name for name in set(unstaged+staged) if not is_new(str(REPO/name))]
    if old_git_diffs: failures.append({'reason':'old_git_diff_paths','paths':sorted(old_git_diffs)})
    result = {'schema':'goal015b_consolidated_integrity_after_v1', 'checked_at_utc':utc(),
        'status':'PASS' if not failures else 'FAIL', 'old_input_files':len(old),
        'old_input_bytes_hashed_once':sum(ref['bytes'] for ref in old.values()),
        'old_input_union_sha256':hashlib.sha256(canonical(old)).hexdigest(),
        'old_input_sources':{group:len(paths) for group,paths in groups.items()},
        'phase4_descriptor_observations':len(phase4), 'phase2_imported_old_sources_covered':len(phase2_refs),
        'audit_input_coverage':coverage, 'receipt_bindings':receipts,
        'all_old_recorded_input_hashes_unchanged':not failures,
        'new_goal015b_read_versions_separate_from_old_input_scope':new_observations,
        'changed_new_file_read_observations':len(changed_new),
        'new_code_change_explanation_from_root':new_code_change_note,
        'new_code_change_explanation_limitation':'Intermediate new-code hashes are retained as observed; this does not reconstruct intermediate file contents or independently prove function-level equivalence.',
        'state':state, 'process_comm_only':process_comm(),
        'process_inspection_limitation':'No process arguments or environment values inspected; read-only parent flock and absent child execution controls checked.',
        'git':{'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO).decode().strip(),
               'branch':subprocess.check_output(['git','branch','--show-current'],cwd=REPO).decode().strip(),
               'unstaged_diff_names':unstaged,'staged_diff_names':staged,'old_diff_names':sorted(old_git_diffs),
               'status_porcelain':subprocess.check_output(['git','status','--porcelain=v1'],cwd=REPO).decode()},
        'failures':failures, 'network_requests':0, 'research_api_requests_sent':0,
        'old_paths_written':0,'inference_calls':0,'archives_redownloaded':0,
        'cpu_seconds':time.process_time()-cpu,'elapsed_seconds':time.monotonic()-started,
        'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'scope_limitation':'Only the explicitly recorded old inputs were rehashed; no old environment or whole-tree scan.'}
    write_exclusive(run_root/'INTEGRITY_AFTER.json', result)
    require(not failures, 'Closure integrity failed; see INTEGRITY_AFTER.json')
    return result


def main():
    parser = argparse.ArgumentParser(description='Goal015B bounded preservation, no execution or network')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('capture'); p.add_argument('--run-root', type=Path, required=True)
    p = sub.add_parser('finalize'); p.add_argument('--before', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('transport'); p.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    result = (capture_snapshot(args.run_root) if args.command == 'capture' else transport_provenance(args.run_root)
              if args.command == 'transport' else finalize_snapshot(args.before, args.output))
    print(json.dumps({key: result[key] for key in ('status', 'files_count', 'checked_files', 'unique_file_bytes', 'cpu_seconds') if key in result}))


if __name__ == '__main__':
    main()
