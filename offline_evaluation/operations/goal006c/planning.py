"""Disabled first-attempt suffix17, preserving both halted predecessor ledgers."""
from __future__ import annotations

from contextlib import ExitStack
from decimal import Decimal
import fcntl
import os
from pathlib import Path
import re
import subprocess

from offline_evaluation.operations.goal006b import planning as previous_plan, runner as previous_runner
from offline_evaluation.server_v2.data import sha256_file
from offline_evaluation.server_v2.generation import GenerationError, object_hash, read_json, write_json, json_bytes, utc_now
from offline_evaluation.server_v2.io_utils import ROOT

SOURCE_PLAN = Path('/root/minecraft-runs/goal006b-ttyfix20-20261015T201419244530Z/plan.json')
SOURCE_IDENTITY = '206dfda6468a90b8d67700e3799571ff0de5aabe1263c15754efe242887f8922'
CHECKPOINT_ARCHIVE_SHA256 = '85498f3b9492d4fc37d9778c1e2f7261bf7bddf0ca6d6e20bc5aeb2410592afe'
CANONICAL_ROOT = Path('/root/minecraft-runs')
PROTOCOL_ID = 'goal006c_unattempted17_v1'
PROVENANCE = 'new_implementation'
SCOPE = 'first_attempt_global943_to959_only_preserve_401_and_520_reservations_usd30_cumulative'
SEMANTIC_FIELDS = ('prompt', 'prompt_sha256', 'schema', 'schema_sha256', 'limits', 'timeout_seconds',
                   'manifest_path', 'manifest_sha256', 'manifest_identity_hash', 'semantic_protocol_id', 'evaluation_scene_ids')
safe_path = previous_plan.safe_path
original_plan = previous_plan.original_plan
base = previous_runner.base


def implementation_hashes():
    old = previous_plan.implementation_hashes()
    own = {p.relative_to(ROOT).as_posix(): sha256_file(p)
           for p in Path(__file__).parent.rglob('*.py') if 'tests' not in p.parts}
    return dict(sorted({**old, **own}.items()))


def verify_code_commit(commit):
    """The operator verifies the remote push; here pin every local code byte to that commit."""
    if not isinstance(commit, str) or not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise GenerationError('goal006c_explicit_code_commit_required')
    from hashlib import sha256
    try:
        for path, expected in implementation_hashes().items():
            data = subprocess.run(['git', 'show', f'{commit}:{path}'], cwd=ROOT,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
            if sha256(data).hexdigest() != expected:
                raise GenerationError('goal006c_code_bytes_differ_from_commit')
    except (OSError, subprocess.SubprocessError):
        raise GenerationError('goal006c_code_commit_unavailable') from None


def current_pricing(identity, *, fresh=False):
    return original_plan.current_pricing(identity, fresh=fresh)


def _source_documents(source_plan_path):
    path = safe_path(source_plan_path)
    if path != SOURCE_PLAN:
        raise GenerationError('goal006c_original_canonical_supplement_required')
    source = previous_plan.load_plan(path)
    if source['identity_hash'] != SOURCE_IDENTITY:
        raise GenerationError('goal006c_source_identity_mismatch')
    identity = source['identity']
    ledger = read_json(safe_path(Path(identity['enabled_directory']) / 'ledger.json'))
    entries = ledger.get('entries', [])
    if (len(entries) != 3 or ledger.get('halt_reason') != 'unknown_usage:http_failure'
            or [row['schedule_index'] for row in identity['schedule']] != list(range(940, 960))
            or [e.get('schedule_index') for e in entries] != [940, 941, 942]
            or [e.get('intent') for e in entries] != ['manual_auth_retry', 'first_attempt', 'first_attempt']
            or [e.get('retry') for e in entries] != [True, False, False]
            or any(e.get('usage') is None or e.get('status') not in {'success', 'failed'} for e in entries[:2])):
        raise GenerationError('goal006c_exact_halted_three_attempt_source_required')
    last, record = entries[-1], identity['schedule'][2]
    if (last.get('status') != 'failed' or last.get('failure_kind') != 'http_failure'
            or last.get('http_status_code') != 520 or last.get('usage') is not None
            or last.get('billing') != 'worst_case_reservation' or last.get('artifacts') != {}
            or last.get('api_response_received') is not True
            or (record.get('provider'), record.get('model'), record.get('scene_id'), record.get('condition'))
            != ('openai', 'gpt-5-mini', 'buildings_100_v4_building_085', 'J')
            or last.get('request_payload_hash') != object_hash(base._payload(identity, record))
            or tuple(last.get(k) for k in ('charged_input_tokens', 'charged_output_tokens', 'charged_cost_usd'))
            != base._reservation(identity, record)):
        raise GenerationError('goal006c_http520_attempt_and_reservation_mismatch')
    pairs = [(r['provider'], r['input_id']) for r in identity['schedule']]
    if len(set(pairs)) != len(pairs) or any((e.get('provider'), e.get('input_id')) != pairs[n] for n, e in enumerate(entries)):
        raise GenerationError('goal006c_duplicate_or_off_schedule_source')
    return source, ledger


def _usage(source, ledger):
    summary = previous_runner.summary(source, ledger)
    original = source['identity']['source_accounting']
    old_unknown = original['unknown_usage_held_reservation']
    unknown = [e for e in ledger['entries'] if e.get('usage') is None]
    known = [e for e in ledger['entries'] if e.get('usage') is not None]
    amount = lambda rows: float(sum((Decimal(str(e['charged_cost_usd'])) for e in rows), Decimal(0)))
    usage = {'calls': summary['combined_calls'], 'retries': summary['combined_retries'],
             'tokens': summary['combined_budget_tokens'], 'cost_usd_upper_bound': summary['combined_budget_cost_usd'],
             'unknown_usage_attempts': source['identity']['parent']['usage']['unknown_usage_attempts'] + len(unknown)}
    holds = [{'source': 'original_http401', 'attempt_id': source['identity']['source_snapshot']['unknown_original_attempt_id'],
              'calls': old_unknown['calls'], 'tokens': old_unknown['tokens'], 'cost_usd_upper_bound': old_unknown['cost_usd_upper_bound']},
             {'source': 'supplement_http520', 'attempt_id': unknown[0]['attempt_id'], 'calls': len(unknown),
              'tokens': sum(e['charged_input_tokens'] + e['charged_output_tokens'] for e in unknown),
              'cost_usd_upper_bound': amount(unknown)}]
    if usage['unknown_usage_attempts'] != 2 or sum(h['calls'] for h in holds) != 2:
        raise GenerationError('goal006c_both_unknown_reservations_required')
    return usage, {'historical_accounting': source['identity']['source_accounting'],
                   'new_source_known_usage_cost_usd': amount(known), 'unknown_usage_holds': holds,
                   'unknown_usage_is_zero': False, 'provider_invoice_claimed': False,
                   'historical_retries': usage['retries']}


def _locks(source):
    identity = source['identity']
    return list(dict.fromkeys([*identity['source_snapshot']['lock_paths'],
        str(Path(identity['enabled_directory']) / 'run.lock'), str(Path(identity['authority_path']).with_suffix('.json.lock'))]))


def _paths(source):
    identity = source['identity']
    tree = sorted(safe_path(p) for p in SOURCE_PLAN.parent.rglob('*') if p.is_file())
    paths = sorted(set(tree) | {safe_path(r['path']) for r in identity['source_snapshot']['files']}
                   | {safe_path(identity['authority_path'])} | {safe_path(p) for p in _locks(source)})
    return tree, paths


def _capture_source(source, ledger):
    directory, checked = previous_runner.load_enabled(source)
    if checked != ledger:
        raise GenerationError('goal006c_source_changed_during_validation')
    tree, paths = _paths(source)
    return {'plan_path': str(SOURCE_PLAN), 'plan_identity_hash': source['identity_hash'],
        'plan_sha256': sha256_file(SOURCE_PLAN), 'ledger_sha256': sha256_file(directory / 'ledger.json'),
        'approval_sha256': sha256_file(directory / 'approval.json'),
        'authority_sha256': sha256_file(source['identity']['authority_path']),
        'halt_reason': ledger['halt_reason'], 'attempted_inputs': len(ledger['entries']),
        'unknown_original_attempt_id': ledger['entries'][-1]['attempt_id'],
        'lock_paths': _locks(source), 'canonical_tree_files': [str(p) for p in tree],
        'files': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256_file(p)} for p in paths],
        'initial_original_validator': 'goal006b.runner.load_enabled'}


def verify_source(identity):
    snapshot = identity['source_snapshot']
    for row in snapshot['files']:
        path = safe_path(row['path'])
        if not path.is_file() or path.stat().st_size != row['bytes'] or sha256_file(path) != row['sha256']:
            raise GenerationError('goal006c_immutable_source_file_changed')
    source, ledger = _source_documents(snapshot['plan_path'])
    tree, paths = _paths(source)
    listed = [r['path'] for r in snapshot['files']]
    if (listed != [str(p) for p in paths] or snapshot['canonical_tree_files'] != [str(p) for p in tree]
            or snapshot['lock_paths'] != _locks(source) or snapshot['attempted_inputs'] != len(ledger['entries'])
            or snapshot['halt_reason'] != ledger['halt_reason']
            or snapshot['unknown_original_attempt_id'] != ledger['entries'][-1]['attempt_id']
            or snapshot['plan_identity_hash'] != source['identity_hash']
            or snapshot['plan_sha256'] != sha256_file(SOURCE_PLAN)
            or snapshot['ledger_sha256'] != sha256_file(Path(source['identity']['enabled_directory']) / 'ledger.json')
            or snapshot['approval_sha256'] != sha256_file(Path(source['identity']['enabled_directory']) / 'approval.json')
            or snapshot['authority_sha256'] != sha256_file(source['identity']['authority_path'])
            or identity['original_source_snapshot'] != source['identity']['source_snapshot']
            or identity['parent']['usage'] != _usage(source, ledger)[0]):
        raise GenerationError('goal006c_immutable_source_inventory_or_accounting_changed')
    return source, ledger


def _verify_checkpoint(path, source, ledger):
    path = safe_path(path); proof = read_json(path)
    assets = proof.get('assets', [])
    names = {'supplement-point-in-time.tar.gz', 'FILES.json', 'snapshot.json', 'SOURCE_REFERENCE.json', 'RESTORE.md', 'SHA256SUMS'}
    asset = next((a for a in assets if a.get('name') == 'supplement-point-in-time.tar.gz'), {})
    if (proof.get('status') != 'GITHUB_VERIFIED' or proof.get('draft') is not True
            or type(proof.get('release_id')) is not int or proof['release_id'] <= 0
            or proof.get('plan_identity_hash') != source['identity_hash'] or proof.get('attempted_inputs') != 3
            or proof.get('verifiedledger_sha256') != sha256_file(Path(source['identity']['enabled_directory']) / 'ledger.json')
            or proof.get('restoration', {}).get('status') != 'VERIFIED'
            or len(assets) != 6 or {a.get('name') for a in assets} != names
            or any(type(a.get('asset_id')) is not int or a['asset_id'] <= 0
                   or a.get('github_digest') != 'sha256:' + a.get('sha256', '') for a in assets)
            or len({a['asset_id'] for a in assets}) != 6 or asset.get('sha256') != CHECKPOINT_ARCHIVE_SHA256):
        raise GenerationError('goal006c_restored_github_three_attempt_checkpoint_required')
    return {'path': str(path), 'sha256': sha256_file(path), 'release_id': proof['release_id'],
        'checkpoint_directory': proof['checkpoint_directory'], 'archive_asset_id': asset['asset_id'],
        'archive_sha256': asset['sha256'], 'ledger_sha256': proof['verifiedledger_sha256'], 'attempted_inputs': 3}


def budgets(source, ledger):
    usage, _ = _usage(source, ledger)
    reservations = [base._reservation(source['identity'], row) for row in source['identity']['schedule'][3:]]
    total = Decimal(str(usage['cost_usd_upper_bound'])) + sum((Decimal(str(r[2])) for r in reservations), Decimal(0))
    limit = source['identity']['budgets']['max_cost_usd']
    if total > Decimal(str(limit)):
        raise GenerationError('goal006c_full_suffix_exceeds_cumulative_cost_limit')
    return {'max_new_calls': 17, 'max_total_calls': source['identity']['budgets']['max_total_calls'],
        'max_new_retries': 0, 'max_automatic_retries': 0, 'max_manual_auth_retries': 0,
        'max_total_retries': usage['retries'], 'max_attempts_per_input': 1, 'concurrency': 1,
        'max_tokens': sum(r[0] + r[1] for r in reservations),
        'max_total_tokens': source['identity']['budgets']['max_total_tokens'],
        'max_cost_usd': limit, 'max_output_tokens': 32768}


def _intents(schedule):
    return [{'local_index': n, 'global_schedule_index': row['schedule_index'], 'kind': 'first_attempt',
             'source_attempt_id': None} for n, row in enumerate(schedule)]


def _pricing(document, previous, *, fresh):
    records = original_plan._pricing_records(document)
    original_plan._unchanged_pricing(previous, records)
    original_plan.validate_pricing(records, fresh=fresh)
    if document['verified_at_utc'] > utc_now():
        raise GenerationError('goal006c_pricing_retrieval_in_future')
    return records


def prepare(destination, *, source_plan_path=SOURCE_PLAN, checkpoint_verification_path, pricing_file, commit):
    destination = safe_path(destination)
    if destination.parent != CANONICAL_ROOT or not destination.name.startswith('goal006c-') or destination.exists():
        raise GenerationError('goal006c_new_canonical_directory_required')
    verify_code_commit(commit)
    source, ledger = _source_documents(source_plan_path)
    with ExitStack() as stack:
        for path in _locks(source):
            stream = stack.enter_context(safe_path(path).open('rb'))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise GenerationError('goal006c_source_run_active') from None
        source, ledger = _source_documents(source_plan_path)
        snapshot = _capture_source(source, ledger)
        checkpoint = _verify_checkpoint(checkpoint_verification_path, source, ledger)
        si = source['identity']; usage, accounting = _usage(source, ledger)
        document = read_json(safe_path(pricing_file)); pricing = _pricing(document, si['pricing'], fresh=True)
        schedule = si['schedule'][3:]
        if [r['schedule_index'] for r in schedule] != list(range(943, 960)) or usage['calls'] + len(schedule) != si['budgets']['max_total_calls']:
            raise GenerationError('goal006c_exact_remaining_seventeen_and_total_calls_required')
        identity = {name: si[name] for name in SEMANTIC_FIELDS}
        identity.update(protocol_id=PROTOCOL_ID, scope=SCOPE, provenance=PROVENANCE,
            new_scope_approved=False, plan_path=str(destination / 'plan.json'), enabled_directory=str(destination / 'enabled'),
            authority_path=str(CANONICAL_ROOT / ('.goal006c-' + SOURCE_IDENTITY + '.json')),
            commit=commit, created_at_utc=utc_now(), source_snapshot=snapshot,
            original_source_snapshot=si['source_snapshot'], parent={'usage': usage, 'lock_paths': snapshot['lock_paths']},
            source_accounting=accounting, source_checkpoint_verification=checkpoint, checkpoint_reference=checkpoint,
            original_checkpoint_reference=si['checkpoint_reference'], schedule=schedule, intents=_intents(schedule),
            batch_scene_ids=list(dict.fromkeys(r['scene_id'] for r in schedule)),
            gt_by_scene={r['scene_id']: si['gt_by_scene'][r['scene_id']] for r in schedule},
            pricing=pricing, pricing_document=document, pricing_document_hash=object_hash(document),
            budgets=budgets(source, ledger), implementation_hashes=implementation_hashes())
        plan = {'version': 'goal006c_disabled_remaining17_v1', 'identity': identity,
                'identity_hash': object_hash(identity), 'api_allowed': False, 'status': 'UNAPPROVED_REMAINING17'}
        destination.mkdir(parents=True, exist_ok=False)
        write_json(destination / 'plan.json', plan)
    return load_plan(destination / 'plan.json')


def load_plan(path):
    path = safe_path(path)
    if path.name != 'plan.json' or path.parent.parent != CANONICAL_ROOT or not path.parent.name.startswith('goal006c-'):
        raise GenerationError('goal006c_canonical_plan_path_required')
    plan = read_json(path); identity = plan['identity']
    if (plan.get('api_allowed') is not False or plan.get('identity_hash') != object_hash(identity)
            or plan.get('version') != 'goal006c_disabled_remaining17_v1' or plan.get('status') != 'UNAPPROVED_REMAINING17'):
        raise GenerationError('goal006c_disabled_identity_mismatch')
    if (identity.get('plan_path') != str(path) or identity.get('enabled_directory') != str(path.parent / 'enabled')
            or identity.get('authority_path') != str(CANONICAL_ROOT / ('.goal006c-' + SOURCE_IDENTITY + '.json'))
            or identity.get('protocol_id') != PROTOCOL_ID or identity.get('scope') != SCOPE
            or identity.get('provenance') != PROVENANCE or identity.get('new_scope_approved') is not False):
        raise GenerationError('goal006c_canonical_scope_or_authority_mismatch')
    if identity.get('implementation_hashes') != implementation_hashes():
        raise GenerationError('goal006c_implementation_hash_mismatch')
    verify_code_commit(identity.get('commit'))
    source, ledger = verify_source(identity); si = source['identity']; schedule = si['schedule'][3:]
    if (identity.get('schedule') != schedule or identity.get('intents') != _intents(schedule)
            or identity.get('budgets') != budgets(source, ledger)
            or identity.get('source_accounting') != _usage(source, ledger)[1]
            or identity['parent'].get('lock_paths') != identity['source_snapshot']['lock_paths']
            or identity.get('batch_scene_ids') != list(dict.fromkeys(r['scene_id'] for r in schedule))
            or identity.get('original_checkpoint_reference') != si['checkpoint_reference']):
        raise GenerationError('goal006c_schedule_budget_or_predecessor_binding_mismatch')
    if any(identity.get(k) != si[k] for k in SEMANTIC_FIELDS):
        raise GenerationError('goal006c_frozen_generation_semantics_changed')
    if identity.get('gt_by_scene') != {r['scene_id']: si['gt_by_scene'][r['scene_id']] for r in schedule}:
        raise GenerationError('goal006c_GT_changed')
    document = identity['pricing_document']
    if identity['pricing_document_hash'] != object_hash(document) or _pricing(document, si['pricing'], fresh=False) != identity['pricing']:
        raise GenerationError('goal006c_pricing_document_mismatch')
    proof = identity['source_checkpoint_verification']
    if (identity['checkpoint_reference'] != proof or sha256_file(safe_path(proof['path'])) != proof['sha256']
            or _verify_checkpoint(proof['path'], source, ledger) != proof):
        raise GenerationError('goal006c_checkpoint_proof_changed')
    for row in schedule:
        if json_bytes(base._payload(identity, row)) != json_bytes(base._payload(si, row)):
            raise GenerationError('goal006c_payload_bytes_changed')
    current_pricing(identity)
    return plan


def refresh_pricing(plan_path, *, pricing_file):
    """Append current same-terms evidence to C only; both predecessors remain immutable."""
    plan = load_plan(plan_path); identity = plan['identity']; path = safe_path(pricing_file)
    document = read_json(path); _pricing(document, identity['pricing'], fresh=True)
    with base._authority_locked(identity):
        current_pricing(identity)
        directory = safe_path(Path(identity['plan_path']).parent / 'pricing-revisions'); directory.mkdir(exist_ok=True)
        previous = sorted(directory.iterdir())
        last = read_json(previous[-1])['revision_hash'] if previous else identity['pricing_document_hash']
        core = {'version': 'official_pricing_reverification_v1', 'plan_identity_hash': plan['identity_hash'],
                'previous_hash': last, 'document': document, 'source_file_sha256': sha256_file(path), 'recorded_at_utc': utc_now()}
        revision = {**core, 'revision_hash': object_hash(core)}
        target = safe_path(directory / f'{len(previous) + 1:04d}.json')
        with target.open('xb') as stream:
            stream.write(json_bytes(revision)); stream.flush(); os.fsync(stream.fileno())
        current_pricing(identity, fresh=True)
    return {'revision_path': str(target), 'revision_hash': revision['revision_hash'], 'plan_identity_hash': plan['identity_hash'],
            'both_predecessors_unchanged': True, 'research_api_sends': 0}
