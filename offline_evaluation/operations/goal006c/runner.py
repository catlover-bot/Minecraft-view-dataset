"""Seventeen new attempts only; both halted source ledgers remain immutable."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from decimal import Decimal
import fcntl
import os
from pathlib import Path
import re
import shlex
import sys
import uuid

from offline_evaluation.batch_v1 import runner as base
from offline_evaluation.resmoke_v2 import adapter, schema
from offline_evaluation.server_v2 import providers
from offline_evaluation.server_v2.core import Limits, ValidationError, execute, grid_to_payload, strict_json, validate_actions
from offline_evaluation.server_v2.generation import GenerationError, json_bytes, object_hash, utc_now
from offline_evaluation.operations.goal006a.archive import RecoveryError
from .archive import RecoveryError as SupplementalRecoveryError
from . import planning

APPROVAL_SCOPE = 'goal006c_unattempted17_global943_to959_cumulative_usd30_calls991_keep_both_unknown_holds'
read_json, write_json = base.read_json, base.write_json


@contextmanager
def source_locked(identity):
    with ExitStack() as stack:
        for name in identity['source_snapshot']['lock_paths']:
            stream = stack.enter_context(base._checked_path(name).open('rb'))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise GenerationError('original_run_active') from None
        planning.verify_source(identity)
        yield


def _approval(plan, approval, *, fresh):
    fields = {'manual_confirmed', 'plan_identity_hash', 'enabled_directory', 'approved_at_utc',
              'confirmation_channel', 'scope', 'acknowledges_both_unknown_holds'}
    if (not isinstance(approval, dict) or set(approval) != fields
            or approval['manual_confirmed'] is not True
            or approval['plan_identity_hash'] != plan['identity_hash']
            or approval['enabled_directory'] != plan['identity']['enabled_directory']
            or approval['confirmation_channel'] != '/dev/tty'
            or approval['scope'] != APPROVAL_SCOPE
            or approval['acknowledges_both_unknown_holds'] is not True
            or not isinstance(approval['approved_at_utc'], str)
            or (fresh and approval['approved_at_utc'][:10] != utc_now()[:10])):
        raise GenerationError('exact_goal006c_scope_and_both_unknown_holds_tty_approval_required')


def _authority(plan, approval):
    return {'version': 'goal006c_authority_v1', 'plan_identity_hash': plan['identity_hash'],
            'enabled_directory': plan['identity']['enabled_directory'],
            'source_ledger_sha256': plan['identity']['source_snapshot']['ledger_sha256'],
            'original_ledger_sha256': plan['identity']['original_source_snapshot']['ledger_sha256'],
            'approval_hash': object_hash(approval)}


def initialize_enabled(plan_path, *, approval):
    plan = planning.load_plan(plan_path)
    _approval(plan, approval, fresh=True)
    identity = plan['identity']
    planning.current_pricing(identity, fresh=True)
    directory = base._checked_path(identity['enabled_directory'])
    with source_locked(identity), base._authority_locked(identity) as authority:
        if authority.exists() or directory.exists():
            raise GenerationError('supplemental_lineage_already_claimed_never_reset_or_reactivate')
        write_json(authority, _authority(plan, approval))
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / 'approval.json', approval)
        write_json(directory / 'enabled.json', {'api_allowed': True, 'plan_identity_hash': plan['identity_hash'],
                   'approval_hash': object_hash(approval), 'enabled_directory': str(directory)})
        write_json(directory / 'ledger.json', {'version': 'goal006c_ledger_v1',
                   'plan_identity_hash': plan['identity_hash'], 'entries': [], 'halt_reason': None})
    return directory


def _claim(plan, entry):
    names = ('attempt_id', 'local_index', 'schedule_index', 'input_id', 'provider',
             'request_payload_hash', 'retry', 'intent', 'source_attempt_id')
    return {'plan_identity_hash': plan['identity_hash'], **{name: entry[name] for name in names}}


def load_enabled(plan):
    identity = plan['identity']
    directory = base._checked_path(identity['enabled_directory'])
    approval = read_json(directory / 'approval.json')
    _approval(plan, approval, fresh=False)
    if read_json(directory / 'enabled.json') != {'api_allowed': True, 'plan_identity_hash': plan['identity_hash'],
            'approval_hash': object_hash(approval), 'enabled_directory': str(directory)}:
        raise GenerationError('supplemental_enabled_binding_mismatch')
    if read_json(Path(identity['authority_path'])) != _authority(plan, approval):
        raise GenerationError('supplemental_authority_mismatch')
    ledger = read_json(directory / 'ledger.json')
    if (ledger.get('version') != 'goal006c_ledger_v1' or ledger.get('plan_identity_hash') != plan['identity_hash']
            or not isinstance(ledger.get('entries'), list) or len(ledger['entries']) > len(identity['schedule'])):
        raise GenerationError('supplemental_ledger_scope_mismatch')
    claims = list(base._checked_path(directory / 'reservations').glob('*.json'))
    if len(claims) != len(ledger['entries']):
        raise GenerationError('supplemental_reservation_ledger_mismatch_never_resend')
    seen = set()
    for index, entry in enumerate(ledger['entries']):
        record, intent = identity['schedule'][index], identity['intents'][index]
        retry = False
        if (intent.get('kind') != 'first_attempt' or intent.get('source_attempt_id') is not None
                or intent.get('local_index') != index or intent.get('global_schedule_index') != record['schedule_index']):
            raise GenerationError('goal006c_only_previously_unattempted_inputs_allowed')
        if (entry.get('local_index') != index or entry.get('schedule_index') != record['schedule_index']
                or entry.get('input_id') != record['input_id'] or entry.get('provider') != record['provider']
                or entry.get('retry') is not retry or entry.get('intent') != intent['kind']
                or entry.get('source_attempt_id') != intent.get('source_attempt_id')
                or entry.get('request_payload_hash') != object_hash(base._payload(identity, record))
                or not re.fullmatch(r'[0-9a-f]{32}', entry.get('attempt_id', ''))
                or entry['attempt_id'] in seen or entry.get('status') not in {'reserved', 'success', 'failed'}):
            raise GenerationError('supplemental_attempt_scope_mismatch')
        seen.add(entry['attempt_id'])
        if read_json(directory / 'reservations' / (entry['attempt_id'] + '.json')) != _claim(plan, entry):
            raise GenerationError('supplemental_immutable_claim_mismatch')
        expected = base._reservation(identity, record) if entry.get('usage') is None else providers.usage_upper_cost(
            record['provider'], entry['usage'], identity['pricing'][record['provider']])
        if tuple(entry.get(k) for k in ('charged_input_tokens', 'charged_output_tokens', 'charged_cost_usd')) != expected:
            raise GenerationError('supplemental_accounting_mismatch')
        if entry.get('billing') != ('worst_case_reservation' if entry.get('usage') is None else 'known_usage_conservative_cost'):
            raise GenerationError('supplemental_unknown_reservation_mislabeled')
        for name, descriptor in entry['artifacts'].items():
            if name not in {'raw.json', 'parsed.json', 'voxels.json'} or descriptor['path'] != f"attempts/{entry['attempt_id']}/{name}":
                raise GenerationError('supplemental_artifact_scope_mismatch')
            base._artifact(directory, descriptor)
        if entry['status'] == 'success' and (not {'raw.json', 'parsed.json', 'voxels.json'} <= entry['artifacts'].keys()
                or not base._completed_response(identity, record, entry)):
            raise GenerationError('supplemental_success_response_mismatch')
        base._validate_response_artifacts(directory, identity, record, entry)
        if entry.get('failure_kind') in base.CRITICAL_FAILURES and not ledger.get('halt_reason'):
            raise GenerationError('supplemental_critical_halt_missing')
        if index < len(ledger['entries']) - 1 and (entry['status'] == 'reserved' or entry.get('usage') is None):
            raise GenerationError('supplemental_continued_after_unknown_usage')
    return directory, ledger


def summary(plan, ledger):
    identity, entries = plan['identity'], ledger['entries']
    known = [e for e in entries if e.get('usage') is not None]
    unknown = [e for e in entries if e.get('usage') is None]
    cost = lambda rows: sum((Decimal(str(e['charged_cost_usd'])) for e in rows), Decimal(0))
    prior = identity['parent']['usage']
    retries = sum(e['retry'] for e in entries)
    complete = len(entries) == len(identity['schedule']) and not unknown and not ledger['halt_reason']
    return {'status': 'SUPPLEMENT_ATTEMPTED' if complete else 'PARTIAL' if entries else 'NOT_RUN',
            'plan_identity_hash': plan['identity_hash'], 'supplemental_attempted': len(entries),
            'supplemental_not_run': len(identity['schedule']) - len(entries),
            'original_attempts_preserved': identity['original_source_snapshot']['attempted_inputs'],
            'source_supplemental_attempts_preserved': identity['source_snapshot']['attempted_inputs'],
            'manual_auth_retry_attempted': retries,
            'previously_unattempted_attempted': sum(e['intent'] == 'first_attempt' for e in entries),
            'automatic_retries': 0, 'halt_reason': ledger['halt_reason'],
            'source_halt_reason_unchanged': identity['source_snapshot']['halt_reason'],
            'original_halt_reason_unchanged': identity['original_source_snapshot']['halt_reason'],
            'historical_unknown_usage_attempts_preserved': prior['unknown_usage_attempts'],
            'parent_budget_accounting': prior,
            'supplemental_known_usage_cost_estimate_usd': float(cost(known)),
            'supplemental_unknown_usage_reservation_usd': float(cost(unknown)),
            'supplemental_unknown_usage_attempts': len(unknown),
            'combined_budget_cost_usd': float(Decimal(str(prior['cost_usd_upper_bound'])) + cost(entries)),
            'combined_calls': prior['calls'] + len(entries), 'combined_retries': prior['retries'] + retries,
            'combined_budget_tokens': prior['tokens'] + sum(e['charged_input_tokens'] + e['charged_output_tokens'] for e in entries),
            'actual_billed_usd': None, 'actual_billed_status': 'NOT_VERIFIED_FROM_PROVIDER_INVOICE',
            'historical_unknown_usage_reservation_reconciled': False,
            'http520_retried': False,
            'result_interpretation': 'first_attempts_global943_to959_only_both_unknown_holds_and_http520_missing_observation_preserved'}


def _verify_checkpoints(plan, override=None):
    from offline_evaluation.operations.goal006a import archive as original_archive
    from . import archive as supplement_archive
    identity = plan['identity']
    scopes = (
        ('original781', 'original_checkpoint_reference', 'original_source_snapshot',
         'continuation-point-in-time.tar.gz', original_archive),
        ('supplement3', 'checkpoint_reference', 'source_snapshot',
         'supplement-point-in-time.tar.gz', supplement_archive),
    )
    reports = override(plan) if override else {}
    for name, reference, snapshot, archive_name, saver in scopes:
        ref, source = identity[reference], identity[snapshot]
        if not override:
            target = Path('/root/minecraft-checkpoints/goal006c-operations') / (name + '-' + uuid.uuid4().hex)
            reports[name] = saver.recover(ref['checkpoint_directory'], target,
                                         release_id=ref['release_id'], allow_create=False)
        result = reports.get(name, {})
        asset = next((a for a in result.get('assets', []) if a.get('name') == archive_name), {})
        if (result.get('status') != 'GITHUB_VERIFIED' or result.get('draft') is not True
                or result.get('release_id') != ref['release_id']
                or result.get('plan_identity_hash') != source['plan_identity_hash']
                or result.get('verifiedledger_sha256') != source['ledger_sha256']
                or result.get('attempted_inputs') != source['attempted_inputs']
                or asset.get('asset_id') != ref['archive_asset_id']
                or asset.get('sha256') != ref['archive_sha256']):
            raise GenerationError(name + '_checkpoint_verification_mismatch')
    return reports


def _save(plan, directory, ledger, reason, callback):
    local = None
    try:
        if callback:
            result = callback(plan, directory, ledger, reason)
        else:
            from .checkpoint import save_local
            from . import archive
            local = save_local(plan, directory, ledger, reason)
            if local.get('status') != 'LOCAL_VERIFIED':
                raise GenerationError('supplemental_local_checkpoint_not_verified')
            target = Path('/root/minecraft-checkpoints/goal006c-operations') / ('save-' + uuid.uuid4().hex)
            result = archive.recover(local['checkpoint_directory'], target, allow_create=True)
        if result.get('status') != 'GITHUB_VERIFIED':
            raise GenerationError('supplemental_checkpoint_not_verified')
        return result
    except Exception as exc:
        failure = {'status': 'SAVE_FAILED', 'stage': 'supplemental_checkpoint', 'code': type(exc).__name__,
                   'generation_ledger_preserved': True}
        if isinstance(exc, (RecoveryError, SupplementalRecoveryError)):
            failure['github_error'] = exc.report
        if local:
            failure['local_status'] = local.get('status')
            failure['checkpoint_directory'] = local.get('checkpoint_directory')
            failure['recovery_action'] = 'upload_only_recover_this_checkpoint_no_generation'
            if local.get('checkpoint_directory'):
                failure['upload_only_retry_command'] = shlex.join([
                    sys.executable, '-B', str(Path(__file__).with_name('cli.py')),
                    '--save-checkpoint', str(local['checkpoint_directory']), '--allow-create'])
        return failure


def run(plan_path, *, permission=False, transports=None, checkpoint_verifier=None, checkpoint_callback=None):
    if permission is not True:
        raise GenerationError('explicit_paid_supplemental_permission_required')
    plan = planning.load_plan(plan_path)
    identity = plan['identity']
    injected = transports is not None or checkpoint_verifier is not None or checkpoint_callback is not None
    if injected and (identity['provenance'] != 'fixture_only' or transports is None or set(transports) != {'openai', 'anthropic'}):
        raise GenerationError('injections_require_fixture_only_plan_and_both_transports')
    directory = base._checked_path(identity['enabled_directory'])
    if not directory.is_dir():
        raise GenerationError('supplement_not_enabled_new_tty_approval_required')
    with source_locked(identity), base.run_lock(directory), base._authority_locked(identity):
        directory, ledger = load_enabled(plan)
        if any(e['status'] == 'reserved' or e.get('usage') is None for e in ledger['entries']):
            if not ledger['halt_reason']:
                ledger['halt_reason'] = 'unreconciled_supplemental_reservation'
                write_json(directory / 'ledger.json', ledger)
        if ledger['halt_reason'] or len(ledger['entries']) == len(identity['schedule']):
            saved = _save(plan, directory, ledger, 'existing_terminal_state', checkpoint_callback)
            return {**summary(plan, ledger), 'checkpoint': saved,
                    'operational_stop': 'checkpoint_save_failed' if saved['status'] == 'SAVE_FAILED' else None}
        if not all(bool(os.environ.get(k)) for k in providers.KEY_NAMES.values()):
            raise GenerationError('both_provider_keys_required_no_key_probe_performed')
        planning.current_pricing(identity, fresh=True)
        verified = _verify_checkpoints(plan, checkpoint_verifier)
        if ledger['entries']:
            # A previous pricing stop may have failed to save. Verify the entire
            # current prefix before reserving any further input on every resume.
            saved_prefix = _save(plan, directory, ledger, 'before_partial_resume', checkpoint_callback)
            if saved_prefix['status'] != 'GITHUB_VERIFIED':
                return {**summary(plan, ledger), 'checkpoint': saved_prefix,
                        'operational_stop': 'checkpoint_save_failed',
                        'source_checkpoint_verification': verified}
        limits = Limits(**identity['limits'])
        for index in range(len(ledger['entries']), len(identity['schedule'])):
            try:
                planning.current_pricing(identity, fresh=True)
            except GenerationError:
                return {**summary(plan, ledger), 'operational_stop': 'pricing_freshness_stop',
                        'checkpoint': _save(plan, directory, ledger, 'pricing_freshness_stop', checkpoint_callback)}
            record, intent = identity['schedule'][index], identity['intents'][index]
            provider, reservation = record['provider'], base._reservation(identity, record)
            if not base._can_reserve(identity, ledger, reservation):
                ledger['halt_reason'] = 'supplemental_budget_insufficient'
                write_json(directory / 'ledger.json', ledger)
                break
            payload = base._payload(identity, record)
            entry = {'attempt_id': uuid.uuid4().hex, 'local_index': index, 'schedule_index': record['schedule_index'],
                'input_id': record['input_id'], 'provider': provider, 'retry': False,
                'intent': intent['kind'], 'source_attempt_id': intent.get('source_attempt_id'),
                'request_payload_hash': object_hash(payload), 'started_at_utc': utc_now(),
                'status': 'reserved', 'failure_kind': None, 'failure_stage': None,
                'usage': None, 'response_id': None, 'response_model': None, 'api_response_received': False,
                'response_parsed': False, 'http_status_code': None, 'natural_stop': None,
                'json_success': None, 'schema_success': None, 'executor_success': None, 'operation_count': None,
                'final_voxel_count': None, 'empty_output': None, 'charged_input_tokens': reservation[0],
                'charged_output_tokens': reservation[1], 'charged_cost_usd': reservation[2],
                'billing': 'worst_case_reservation', 'artifacts': {}}
            claim = base._checked_path(directory / 'reservations' / (entry['attempt_id'] + '.json'))
            claim.parent.mkdir(parents=True, exist_ok=True)
            with claim.open('xb') as stream:
                stream.write(json_bytes(_claim(plan, entry))); stream.flush(); os.fsync(stream.fileno())
            ledger['entries'].append(entry)
            write_json(directory / 'ledger.json', ledger)
            try:
                raw = adapter.send_request(provider, payload, api_allowed=True, permission=True,
                      timeout=identity['timeout_seconds'], transport=(transports or {}).get(provider))
                if not isinstance(raw, bytes):
                    raise providers.ProviderError('response_parse_failure')
                entry['api_response_received'] = True
                base._save_artifact(directory, entry, 'raw.json', raw)
                response = adapter.parse_response(provider, raw)
                entry['response_parsed'] = True
                entry['response_id'], entry['response_model'] = response.response_id, response.model
                entry['natural_stop'] = response.complete
                charge = providers.usage_upper_cost(provider, response.usage, identity['pricing'][provider])
                entry['usage'] = response.usage
                entry['charged_input_tokens'], entry['charged_output_tokens'], entry['charged_cost_usd'] = charge
                entry['billing'] = 'known_usage_conservative_cost'
                if any(a > b for a, b in zip(charge, reservation)):
                    raise providers.ProviderError('budget_violation')
                if not base._completed_response(identity, record, entry):
                    raise providers.ProviderError('response_identity_failure')
                if response.failure_kind:
                    raise providers.ProviderError(response.failure_kind)
                if not response.complete:
                    raise providers.ProviderError('incomplete_response')
                try:
                    document = strict_json(response.text, limits)
                    entry['json_success'] = True
                except (ValidationError, TypeError):
                    entry['json_success'], entry['failure_stage'] = False, 'json'
                    raise providers.ProviderError('parse_failure') from None
                try:
                    actions = schema.validate_document(document)
                    entry['schema_success'] = True
                except ValidationError:
                    entry['schema_success'], entry['failure_stage'] = False, 'schema'
                    raise providers.ProviderError('validation_failure') from None
                try:
                    actions = validate_actions(actions, limits)
                except ValidationError:
                    entry['failure_stage'] = 'local_validator'
                    raise providers.ProviderError('validation_failure') from None
                entry['operation_count'] = len(actions)
                base._save_artifact(directory, entry, 'parsed.json', json_bytes({'operations': actions}))
                try:
                    grid = execute(actions, limits)
                except (ValidationError, ValueError):
                    entry['executor_success'], entry['failure_stage'] = False, 'executor'
                    raise providers.ProviderError('execution_failure') from None
                entry['executor_success'], entry['final_voxel_count'] = True, len(grid.voxels)
                entry['empty_output'] = not bool(grid.voxels)
                base._save_artifact(directory, entry, 'voxels.json', json_bytes(grid_to_payload(grid)))
                entry['status'] = 'success'
            except providers.ProviderError as exc:
                entry['http_status_code'] = exc.status_code
                entry['api_response_received'] = entry['api_response_received'] or exc.status_code is not None
                entry['status'], entry['failure_kind'] = 'failed', exc.kind
                if entry['usage'] is None:
                    ledger['halt_reason'] = 'unknown_usage:' + exc.kind
                elif exc.kind in base.CRITICAL_FAILURES:
                    ledger['halt_reason'] = exc.kind
            entry['finished_at_utc'] = utc_now()
            write_json(directory / 'ledger.json', ledger)
            write_json(directory / 'summary.json', summary(plan, ledger))
            if ledger['halt_reason']:
                break
        planning.verify_source(identity)
        saved = _save(plan, directory, ledger, 'safe_halt' if ledger['halt_reason'] else 'supplement_finished', checkpoint_callback)
        return {**summary(plan, ledger), 'checkpoint': saved,
                'operational_stop': 'checkpoint_save_failed' if saved['status'] == 'SAVE_FAILED' else None,
                'source_checkpoint_verification': verified}


def status(plan_path):
    plan = planning.load_plan(plan_path)
    identity = plan['identity']
    with source_locked(identity):
        directory = base._checked_path(identity['enabled_directory'])
        if not directory.exists():
            if base._checked_path(identity['authority_path']).exists():
                raise GenerationError('supplemental_partial_activation_never_reset')
            ledger, approved = {'entries': [], 'halt_reason': None}, False
        else:
            lock = directory / 'run.lock'
            if not lock.exists():
                lock = Path(identity['authority_path'] + '.lock')
            with base._checked_path(lock).open('rb') as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise GenerationError('supplemental_run_active') from None
                _, ledger = load_enabled(plan)
            approved = True
        return {**summary(plan, ledger), 'approved': approved, 'api_allowed': False,
                'preview_only': True, 'research_api_requests_sent_by_preview': 0}
