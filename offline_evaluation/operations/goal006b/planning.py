"""A separate, disabled twenty-send supplement bound to the halted originals.

No original control, attempt, reservation or halt is edited. The original HTTP
401 remains an unknown-usage, fully held reservation, even if the newly approved
manual retry eventually yields a measured outcome for the same fixed slot.
"""
from __future__ import annotations

from contextlib import ExitStack
from decimal import Decimal
import fcntl
import os
from pathlib import Path
import re

from offline_evaluation.continuation_v1 import plan as original_plan, runner as original_runner
from offline_evaluation.server_v2.data import sha256_file
from offline_evaluation.server_v2.generation import GenerationError, object_hash, read_json, write_json, json_bytes, utc_now
from offline_evaluation.server_v2.io_utils import ROOT

SOURCE_PLAN = Path('/root/minecraft-runs/goal006-remaining800-20261014T101309Z-v2/plan.json')
SOURCE_IDENTITY = 'db5533bffb583d0b1efcbcdb7c1563edf2ac87606334e6ce6cd766c650075843'
CANONICAL_ROOT = Path('/root/minecraft-runs')
PROTOCOL_ID = 'goal006b_explicit_auth_retry_and_unattempted19_v1'
PROVENANCE = 'new_implementation'
SCOPE = 'explicit_single_auth_retry_global940_then_unattempted941_to959_usd30_cumulative'
CHECKPOINT_RELEASE = 414157245
CHECKPOINT_ARCHIVE_ASSET = 644830729
CHECKPOINT_ARCHIVE_SHA256 = 'c7897e5f121df0c08cc2a2a7e993f02fae3ac15fa5ca6f248e1e16cd14ca36a5'
safe_path = original_plan.safe_path


def implementation_hashes():
    hashes = original_plan.implementation_hashes()
    paths = [p for p in Path(__file__).parent.rglob('*.py') if 'tests' not in p.parts]
    # The operations archive primitives are reused by the supplement saver.
    paths.append(ROOT / 'offline_evaluation/operations/goal006a/archive.py')
    return dict(sorted({**hashes, **{p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}}.items()))


def current_pricing(identity, *, fresh=False):
    return original_plan.current_pricing(identity, fresh=fresh)


def _latest_document(source):
    identity = source['identity']
    current_pricing(identity, fresh=True)
    revisions = sorted((Path(identity['plan_path']).parent / 'pricing-revisions').glob('*.json'))
    return read_json(revisions[-1])['document'] if revisions else identity['pricing_document']


def _source_documents(source_plan_path):
    path = safe_path(source_plan_path)
    if path != SOURCE_PLAN:
        raise GenerationError('supplement_requires_original_canonical_source')
    source = original_plan.load_plan(path)
    if source['identity_hash'] != SOURCE_IDENTITY:
        raise GenerationError('supplement_source_identity_mismatch')
    identity = source['identity']
    ledger = read_json(safe_path(Path(identity['enabled_directory']) / 'ledger.json'))
    entries = ledger['entries']
    if (len(entries) != 781 or ledger.get('halt_reason') != 'unknown_usage:auth_failure'
            or len(identity['schedule']) != 800
            or sum(entry.get('usage') is None for entry in entries) != 1):
        raise GenerationError('supplement_requires_exact_halted781_with_one_unknown_auth_attempt')
    last, record = entries[-1], identity['schedule'][780]
    if (last.get('local_index') != 780 or last.get('schedule_index') != 940
            or last.get('status') != 'failed' or last.get('failure_kind') != 'auth_failure'
            or last.get('http_status_code') != 401 or last.get('usage') is not None
            or last.get('billing') != 'worst_case_reservation' or last.get('artifacts') != {}
            or last.get('api_response_received') is not True
            or (record.get('provider'), record.get('scene_id'), record.get('condition'), record.get('description_model'))
               != ('openai', 'buildings_100_v4_building_085', 'O', 'openai')
            or (last.get('provider'), last.get('input_id')) != (record['provider'], record['input_id'])
            or last.get('request_payload_hash') != object_hash(original_runner._payload(identity, record))):
        raise GenerationError('supplement_manual_auth_retry_target_mismatch')
    if [row['schedule_index'] for row in identity['schedule'][780:]] != list(range(940, 960)):
        raise GenerationError('supplement_exact_twenty_global_indices_required')
    reservation = original_runner._reservation(identity, record)
    if tuple(last.get(k) for k in ('charged_input_tokens', 'charged_output_tokens', 'charged_cost_usd')) != reservation:
        raise GenerationError('supplement_original_unknown_reservation_must_remain_held')
    return source, ledger


def _usage(source, ledger):
    totals = original_runner.budget_totals(source['identity'], ledger)
    known = [entry for entry in ledger['entries'] if entry['usage'] is not None]
    held = [entry for entry in ledger['entries'] if entry['usage'] is None]
    parent = source['identity']['parent']['usage']
    def amounts(entries):
        return {'calls': len(entries),
                'tokens': sum(entry['charged_input_tokens'] + entry['charged_output_tokens'] for entry in entries),
                'cost_usd_upper_bound': float(sum((Decimal(str(entry['charged_cost_usd'])) for entry in entries), Decimal(0)))}
    measured, reserved = amounts(known), amounts(held)
    measured['calls'] += parent['calls']
    measured['tokens'] += parent['tokens']
    measured['cost_usd_upper_bound'] = float(Decimal(str(measured['cost_usd_upper_bound'])) + Decimal(str(parent['cost_usd_upper_bound'])))
    return {**totals['combined'], 'unknown_usage_attempts': len(held)}, {
        'known_usage_conservative': measured, 'unknown_usage_held_reservation': reserved,
        'unknown_usage_is_zero': False, 'provider_invoice_claimed': False,
        'historical_retries': totals['combined']['retries']}


def _files(paths):
    return [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
            for path in sorted({safe_path(p) for p in paths})]


def _capture_source(source, ledger):
    identity = source['identity']
    original_plan.verify_parent(identity['parent'])
    directory, checked = original_runner._load_enabled(source)
    if checked != ledger:
        raise GenerationError('source_changed_during_initial_validation')
    tree = sorted(p for p in Path(identity['plan_path']).parent.rglob('*') if p.is_file())
    locks = list(dict.fromkeys([*identity['parent']['lock_paths'], str(directory / 'run.lock'),
                               str(Path(identity['authority_path']).with_suffix('.json.lock'))]))
    paths = [*tree, Path(identity['authority_path']), *map(Path, locks)]
    paths += list(map(Path, identity['parent'].get('original_files_sha256', {})))
    return {'plan_path': identity['plan_path'], 'plan_identity_hash': source['identity_hash'],
            'plan_sha256': sha256_file(identity['plan_path']),
            'ledger_sha256': sha256_file(directory / 'ledger.json'),
            'approval_sha256': sha256_file(directory / 'approval.json'),
            'authority_sha256': sha256_file(identity['authority_path']),
            'halt_reason': ledger['halt_reason'], 'attempted_inputs': len(ledger['entries']),
            'canonical_tree_files': [str(path) for path in tree], 'files': _files(paths),
            'lock_paths': locks, 'initial_original_validator': 'continuation_v1.runner._load_enabled',
            'unknown_original_attempt_id': ledger['entries'][-1]['attempt_id']}


def _verify_checkpoint(path, source, ledger):
    path = safe_path(path)
    proof = read_json(path)
    archive = next((asset for asset in proof.get('assets', [])
                    if asset.get('name') == 'continuation-point-in-time.tar.gz'), None)
    if (proof.get('status') != 'GITHUB_VERIFIED' or proof.get('release_id') != CHECKPOINT_RELEASE
            or proof.get('draft') is not True or proof.get('attempted_inputs') != len(ledger['entries'])
            or proof.get('plan_identity_hash') != source['identity_hash']
            or proof.get('verifiedledger_sha256') != sha256_file(Path(source['identity']['enabled_directory']) / 'ledger.json')
            or archive is None or archive.get('asset_id') != CHECKPOINT_ARCHIVE_ASSET
            or archive.get('sha256') != CHECKPOINT_ARCHIVE_SHA256
            or archive.get('github_digest') != 'sha256:' + CHECKPOINT_ARCHIVE_SHA256
            or proof.get('restoration', {}).get('status') != 'VERIFIED'):
        raise GenerationError('verified_original781_checkpoint_required')
    return {'path': str(path), 'sha256': sha256_file(path), 'release_id': CHECKPOINT_RELEASE,
            'checkpoint_directory': proof['checkpoint_directory'],
            'archive_asset_id': CHECKPOINT_ARCHIVE_ASSET, 'archive_sha256': CHECKPOINT_ARCHIVE_SHA256,
            'ledger_sha256': proof['verifiedledger_sha256'], 'attempted_inputs': len(ledger['entries'])}


def budgets(source, ledger):
    identity = source['identity']
    usage, _ = _usage(source, ledger)
    reservations = [original_runner._reservation(identity, row) for row in identity['schedule'][780:]]
    return {'max_new_calls': 20, 'max_total_calls': usage['calls'] + 20,
            'max_new_retries': 0, 'max_automatic_retries': 0, 'max_manual_auth_retries': 1,
            'max_total_retries': usage['retries'] + 1, 'max_attempts_per_input': 1, 'concurrency': 1,
            'max_tokens': sum(input_tokens + output_tokens for input_tokens, output_tokens, _ in reservations),
            'max_total_tokens': identity['budgets']['max_total_tokens'],
            'max_cost_usd': identity['budgets']['max_cost_usd'], 'max_output_tokens': 32768}


def prepare(destination, *, source_plan_path=SOURCE_PLAN, checkpoint_verification_path, commit):
    destination = safe_path(destination)
    if (destination.parent != CANONICAL_ROOT or not destination.name.startswith('goal006b-')
            or destination.exists() or not re.fullmatch(r'[0-9a-f]{40}', commit)):
        raise GenerationError('new_canonical_goal006b_directory_and_commit_required')
    source, ledger = _source_documents(source_plan_path)
    original_identity = source['identity']
    locks = list(dict.fromkeys([*original_identity['parent']['lock_paths'],
        str(Path(original_identity['enabled_directory']) / 'run.lock'),
        str(Path(original_identity['authority_path']).with_suffix('.json.lock'))]))
    with ExitStack() as stack:
        for path in locks:
            stream = stack.enter_context(safe_path(path).open('rb'))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise GenerationError('original_generation_active; supplement_preparation_stopped') from None
            stack.callback(fcntl.flock, stream, fcntl.LOCK_UN)
        source, ledger = _source_documents(source_plan_path)
        snapshot = _capture_source(source, ledger)
        checkpoint = _verify_checkpoint(checkpoint_verification_path, source, ledger)
        usage, accounting = _usage(source, ledger)
        document = _latest_document(source)
        schedule = source['identity']['schedule'][780:]
        intents = [{'local_index': index, 'global_schedule_index': row['schedule_index'],
                    'kind': 'manual_auth_retry' if index == 0 else 'first_attempt',
                    'source_attempt_id': ledger['entries'][-1]['attempt_id'] if index == 0 else None}
                   for index, row in enumerate(schedule)]
        identity = {name: source['identity'][name] for name in ('prompt', 'prompt_sha256', 'schema', 'schema_sha256',
                    'limits', 'timeout_seconds', 'manifest_path', 'manifest_sha256', 'manifest_identity_hash',
                    'semantic_protocol_id', 'evaluation_scene_ids')}
        identity.update(protocol_id=PROTOCOL_ID, scope=SCOPE, provenance=PROVENANCE,
            new_scope_approved=False, plan_path=str(destination / 'plan.json'),
            enabled_directory=str(destination / 'enabled'),
            authority_path=str(CANONICAL_ROOT / ('.goal006b-' + source['identity_hash'] + '.json')),
            commit=commit, created_at_utc=utc_now(), source_snapshot=snapshot,
            parent={'usage': usage, 'lock_paths': snapshot['lock_paths']}, source_accounting=accounting,
            source_checkpoint_verification=checkpoint, checkpoint_reference=checkpoint, schedule=schedule, intents=intents,
            batch_scene_ids=list(dict.fromkeys(row['scene_id'] for row in schedule)),
            gt_by_scene={row['scene_id']: source['identity']['gt_by_scene'][row['scene_id']] for row in schedule},
            pricing=original_plan._pricing_records(document), pricing_document=document,
            pricing_document_hash=object_hash(document), budgets=budgets(source, ledger),
            implementation_hashes=implementation_hashes())
        plan = {'version': 'goal006b_disabled_supplement_v1', 'identity': identity,
                'identity_hash': object_hash(identity), 'api_allowed': False, 'status': 'UNAPPROVED_SUPPLEMENT20'}
        destination.mkdir(parents=True, exist_ok=False)
        write_json(destination / 'plan.json', plan)
    return load_plan(destination / 'plan.json')


def verify_source(identity):
    """Reuse the initial full validator only while every source byte still agrees."""
    snapshot = identity['source_snapshot']
    for descriptor in snapshot['files']:
        path = safe_path(descriptor['path'])
        if not path.is_file() or path.stat().st_size != descriptor['bytes'] or sha256_file(path) != descriptor['sha256']:
            raise GenerationError('immutable_halted_source_file_changed')
    actual = sorted(str(path) for path in Path(snapshot['plan_path']).parent.rglob('*') if path.is_file())
    if actual != snapshot['canonical_tree_files']:
        raise GenerationError('immutable_halted_source_inventory_changed')
    source, ledger = _source_documents(snapshot['plan_path'])
    source_identity = source['identity']
    expected_locks = list(dict.fromkeys([*source_identity['parent']['lock_paths'],
        str(Path(source_identity['enabled_directory']) / 'run.lock'),
        str(Path(source_identity['authority_path']).with_suffix('.json.lock'))]))
    expected_paths = set(actual) | {source_identity['authority_path'], *expected_locks,
                                   *source_identity['parent'].get('original_files_sha256', {})}
    listed_paths = [descriptor['path'] for descriptor in snapshot['files']]
    if (set(listed_paths) != expected_paths or len(listed_paths) != len(set(listed_paths))
            or snapshot.get('lock_paths') != expected_locks
            or snapshot.get('attempted_inputs') != len(ledger['entries'])
            or snapshot.get('halt_reason') != ledger['halt_reason']
            or snapshot.get('unknown_original_attempt_id') != ledger['entries'][-1]['attempt_id']
            or snapshot.get('approval_sha256') != sha256_file(Path(source_identity['enabled_directory']) / 'approval.json')
            or snapshot.get('authority_sha256') != sha256_file(source_identity['authority_path'])):
        raise GenerationError('immutable_source_snapshot_inventory_or_controls_mismatch')
    if (sha256_file(snapshot['plan_path']) != snapshot['plan_sha256']
            or source['identity_hash'] != snapshot['plan_identity_hash']
            or sha256_file(Path(source['identity']['enabled_directory']) / 'ledger.json') != snapshot['ledger_sha256']
            or _usage(source, ledger)[0] != identity['parent']['usage']):
        raise GenerationError('immutable_source_accounting_or_identity_changed')
    return source, ledger


def load_plan(path):
    path = safe_path(path)
    if path.name != 'plan.json' or path.parent.parent != CANONICAL_ROOT or not path.parent.name.startswith('goal006b-'):
        raise GenerationError('supplement_canonical_plan_path_required')
    plan = read_json(path); identity = plan['identity']
    if (plan.get('api_allowed') is not False or plan.get('identity_hash') != object_hash(identity)
            or plan.get('version') != 'goal006b_disabled_supplement_v1'):
        raise GenerationError('supplement_identity_hash_mismatch')
    if (identity.get('plan_path') != str(path) or identity.get('enabled_directory') != str(path.parent / 'enabled')
            or identity.get('authority_path') != str(CANONICAL_ROOT / ('.goal006b-' + SOURCE_IDENTITY + '.json'))
            or identity.get('protocol_id') != PROTOCOL_ID or identity.get('scope') != SCOPE
            or identity.get('provenance') != PROVENANCE or identity.get('new_scope_approved') is not False):
        raise GenerationError('supplement_canonical_scope_or_authority_mismatch')
    if identity.get('implementation_hashes') != implementation_hashes():
        raise GenerationError('supplement_implementation_hash_mismatch')
    source, ledger = verify_source(identity)
    schedule = source['identity']['schedule'][780:]
    expected_intents = [{'local_index': index, 'global_schedule_index': record['schedule_index'],
                        'kind': 'manual_auth_retry' if index == 0 else 'first_attempt',
                        'source_attempt_id': ledger['entries'][-1]['attempt_id'] if index == 0 else None}
                       for index, record in enumerate(schedule)]
    if (identity.get('schedule') != schedule or identity.get('intents') != expected_intents
            or identity.get('budgets') != budgets(source, ledger)
            or identity.get('source_accounting') != _usage(source, ledger)[1]
            or identity.get('parent', {}).get('lock_paths') != identity['source_snapshot']['lock_paths']):
        raise GenerationError('supplement_fixed_schedule_intents_or_accounting_mismatch')
    for name in ('prompt', 'prompt_sha256', 'schema', 'schema_sha256', 'limits', 'timeout_seconds',
                 'manifest_path', 'manifest_sha256', 'manifest_identity_hash', 'semantic_protocol_id', 'evaluation_scene_ids'):
        if identity.get(name) != source['identity'][name]:
            raise GenerationError('supplement_frozen_generation_semantics_changed')
    expected_gt = {row['scene_id']: source['identity']['gt_by_scene'][row['scene_id']] for row in schedule}
    if identity.get('gt_by_scene') != expected_gt:
        raise GenerationError('supplement_fixed_GT_changed')
    document = identity['pricing_document']
    if (identity['pricing_document_hash'] != object_hash(document)
            or identity['pricing'] != original_plan._pricing_records(document)):
        raise GenerationError('supplement_pricing_document_changed')
    original_plan._unchanged_pricing(source['identity']['pricing'], identity['pricing'])
    proof = identity['source_checkpoint_verification']
    if (identity.get('checkpoint_reference') != proof or sha256_file(safe_path(proof['path'])) != proof['sha256']
            or _verify_checkpoint(proof['path'], source, ledger) != proof):
        raise GenerationError('supplement_original_checkpoint_proof_changed')
    for record in schedule:
        if original_runner._payload(identity, record) != original_runner._payload(source['identity'], record):
            raise GenerationError('supplement_production_payload_changed')
    current_pricing(identity)
    return plan


def refresh_pricing(plan_path, *, pricing_file):
    """Append same-terms current evidence to the child only, with no activation."""
    plan = load_plan(plan_path); identity = plan['identity']
    pricing_file = safe_path(pricing_file)
    document = read_json(pricing_file)
    pricing = original_plan._pricing_records(document)
    original_plan.validate_pricing(pricing, fresh=True)
    if document['verified_at_utc'] > utc_now():
        raise GenerationError('pricing_retrieval_cannot_be_in_the_future')
    original_plan._unchanged_pricing(identity['pricing'], pricing)
    with original_runner._authority_locked(identity):
        current_pricing(identity)
        directory = safe_path(Path(identity['plan_path']).parent / 'pricing-revisions')
        directory.mkdir(exist_ok=True)
        previous = sorted(directory.iterdir())
        last_hash = read_json(previous[-1])['revision_hash'] if previous else identity['pricing_document_hash']
        core = {'version': 'official_pricing_reverification_v1', 'plan_identity_hash': plan['identity_hash'],
                'previous_hash': last_hash, 'document': document,
                'source_file_sha256': sha256_file(pricing_file), 'recorded_at_utc': utc_now()}
        revision = {**core, 'revision_hash': object_hash(core)}
        target = safe_path(directory / f'{len(previous) + 1:04d}.json')
        with target.open('xb') as stream:
            stream.write(json_bytes(revision)); stream.flush(); os.fsync(stream.fileno())
        current_pricing(identity, fresh=True)
    return {'revision_path': str(target), 'revision_hash': revision['revision_hash'],
            'plan_identity_hash': plan['identity_hash'], 'original_plan_approval_ledger_halt_unchanged': True,
            'supplement_approval_ledger_unchanged': True, 'research_api_sends': 0}
