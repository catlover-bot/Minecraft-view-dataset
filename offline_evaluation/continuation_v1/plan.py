"""Disabled suffix800 plan, immutable completed-parent proof and pricing history."""
from __future__ import annotations

import os
from pathlib import Path
import re

from ..batch_v1 import plan as batch_plan, runner as batch_runner
from ..server_v2.data import sha256_file
from ..server_v2.generation import GenerationError, object_hash, read_json, utc_now, write_json, json_bytes
from ..server_v2.io_utils import ROOT
from .inputs import suffix

MODELS = batch_plan.MODELS
OUTPUT_CAP = batch_plan.OUTPUT_CAP
PROMPT_PATH = batch_plan.PROMPT_PATH
PARENT_PLAN_PATH = Path('/root/minecraft-runs/goal005-batch1-20261013T213420Z/plan.json')
PARENT_IDENTITY = 'f54ebfaa041a18cbbfb60d2d27f9e9c4a50d436545c7ed1175c60751be58816f'
PARENT_FILE_SHA256 = '231b6daffb671511cd3adc22847d9d72c26a37db14b5c6793520a6f987001b13'
MANIFEST_SHA256 = 'ad8f5d130351a1027ea9b3929bec357d3c2fd7ba6f5810993bcbaa0b901b4116'
PARENT_USAGE = {'calls': 190, 'retries': 4, 'tokens': 1828481,
                'cost_usd_upper_bound': 4.048922, 'unknown_usage_attempts': 0}
PARENT_PROOF_PATH = ROOT / 'reports/server_only/goal006/results/verification.json'
PARENT_PROOF_SHA256 = 'a378835df416a8cd52048ecb114719d5ab23c143126ee5fe0b2748ada590493e'
PROTOCOL_ID = 'goal006_fixed_evaluation_continuation800_v1'
SCOPE = 'fixed_evaluation60_full_schedule_suffix160_to959_only_800_sends_no_retries'
AUTHORITY_PATH = Path('/root/minecraft-runs/.continuation-v1-' + PARENT_IDENTITY + '-remaining800.json')
CANONICAL_ROOT = Path('/root/minecraft-runs')
safe_path = batch_plan.safe_path
validate_pricing = batch_plan.validate_pricing
_pricing_records = batch_plan._pricing_records
_unchanged_pricing = batch_plan._unchanged_pricing


def current_pricing(identity, *, fresh=False):
    pricing = batch_plan.current_pricing(identity, fresh=False)
    validate_pricing(pricing, fresh=fresh)
    return pricing


def implementation_hashes():
    old = batch_plan.implementation_hashes()
    own = {p.relative_to(ROOT).as_posix(): sha256_file(p)
           for p in Path(__file__).parent.rglob('*.py') if 'tests' not in p.parts}
    return dict(sorted({**old, **own}.items()))


def _parent_documents():
    path = safe_path(PARENT_PLAN_PATH)
    if sha256_file(path) != PARENT_FILE_SHA256:
        raise GenerationError('completed_batch_parent_plan_changed')
    plan = read_json(path)
    if plan['identity_hash'] != PARENT_IDENTITY or object_hash(plan['identity']) != PARENT_IDENTITY:
        raise GenerationError('completed_batch_parent_identity_changed')
    identity = plan['identity']
    if identity['implementation_hashes'] != batch_plan.implementation_hashes():
        raise GenerationError('completed_batch_bound_code_changed')
    manifest_path = safe_path(identity['manifest_path'])
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise GenerationError('original_full_manifest_bytes_changed')
    manifest = read_json(manifest_path)
    if manifest['identity_hash'] != identity['manifest_identity_hash']:
        raise GenerationError('original_full_manifest_identity_changed')
    suffix(manifest, plan)
    return plan, manifest


def parent_snapshot():
    """Reuse the full original-validator proof only while every proved byte matches.

    Goal006's initial audit already executed all154 successful raw→parsed→voxel
    chains. A pinned proof plus its complete exact file inventory avoids replaying
    those154 operations on every status/preview; changed bytes fail closed.
    """
    parent_plan, _ = _parent_documents()
    if sha256_file(safe_path(PARENT_PROOF_PATH)) != PARENT_PROOF_SHA256:
        raise GenerationError('completed_parent_validation_proof_changed')
    proof = read_json(PARENT_PROOF_PATH)
    if (proof['status'] != 'VERIFIED_COMPLETE_BATCH160'
            or proof['plan_sha256'] != PARENT_FILE_SHA256
            or proof['successful_raw_parsed_voxel_replays'] != 154
            or proof['pending_reservations'] != 0 or proof['unknown_usage_attempts'] != 0
            or proof['existing_validators'] != ['planning.load_plan', 'planning.verify_parent', 'runner._load_enabled']
            or proof['implementation_hashes'] != batch_plan.implementation_hashes()):
        raise GenerationError('completed_parent_validation_proof_invalid')
    hashes = {}
    for row in proof['files']:
        path = safe_path(row['path'])
        if path.stat().st_size != row['bytes'] or sha256_file(path) != row['sha256']:
            raise GenerationError('completed_parent_proved_artifact_changed')
        hashes[str(path)] = row['sha256']
    identity = parent_plan['identity']
    directory = safe_path(identity['enabled_directory'])
    expected_tree = {p for p in hashes if Path(p).is_relative_to(PARENT_PLAN_PATH.parent)}
    actual_tree = {str(safe_path(p)) for p in PARENT_PLAN_PATH.parent.rglob('*') if p.is_file()}
    if actual_tree != expected_tree:
        raise GenerationError('completed_parent_file_inventory_changed')
    # The original14 campaign validator checks stored controls and artifact hashes,
    # without executing generated geometry or recomputing metrics.
    batch_plan.original_plan.verify_parent(identity['parent']['old_campaign'])
    for name, expected in identity['parent']['original_files_sha256'].items():
        if sha256_file(safe_path(name)) != expected:
            raise GenerationError('completed_old30_lineage_changed')
    ledger = read_json(directory / 'ledger.json')
    summary = batch_runner.execution_summary(parent_plan, ledger)
    usage = {**summary['used']['combined'], 'unknown_usage_attempts': summary['used']['new']['unknown_usage_attempts']}
    if summary != proof['execution'] or not summary['experiment_complete'] or usage != PARENT_USAGE:
        raise GenerationError('completed_parent_usage_or_terminal_state_changed')
    return {'plan_path': str(PARENT_PLAN_PATH), 'plan_identity_hash': PARENT_IDENTITY,
            'plan_sha256': PARENT_FILE_SHA256, 'enabled_directory': str(directory),
            'usage': usage, 'ledger_sha256': sha256_file(directory / 'ledger.json'),
            'original_files_sha256': hashes, 'lock_paths': proof['idle_locks_acquired_readonly'],
            'verification_proof_path': str(PARENT_PROOF_PATH),
            'verification_proof_sha256': PARENT_PROOF_SHA256,
            'successful_execution_replays_reused': 154}


def verify_parent(parent):
    if parent_snapshot() != parent:
        raise GenerationError('parent_changed; never_rebind_or_reset_previous_ledger')


def budgets(pricing):
    maximum = sum(400 * (pricing[p]['context_window_tokens'] + OUTPUT_CAP) for p in MODELS)
    return {'max_new_calls': 800, 'max_total_calls': 990, 'max_new_retries': 0,
            'max_total_retries': 4, 'max_attempts_per_input': 1, 'concurrency': 1,
            'max_cost_usd': 30.0, 'max_tokens': maximum,
            'max_total_tokens': PARENT_USAGE['tokens'] + maximum, 'max_output_tokens': OUTPUT_CAP}


def prepare(destination, *, pricing_file, commit, checkpoint_sources):
    destination = safe_path(destination)
    if destination.exists() or not destination.is_relative_to(CANONICAL_ROOT):
        raise GenerationError('new_unique_canonical_preparation_directory_required')
    if not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit):
        raise GenerationError('explicit_forty_digit_commit_required')
    if AUTHORITY_PATH.exists():
        raise GenerationError('continuation_lineage_already_activated_resume_original_ledger')
    parent = parent_snapshot()
    parent_plan, manifest = _parent_documents()
    source = parent_plan['identity']
    schedule, scenes, segments = suffix(manifest, parent_plan)
    document = read_json(safe_path(pricing_file))
    pricing = _pricing_records(document)
    _unchanged_pricing(source['pricing'], pricing)
    source_path = safe_path(checkpoint_sources)
    refs = read_json(source_path)
    if refs.get('completed160_verified') is not True:
        raise GenerationError('completed160_verified_github_checkpoint_required_before_continuation')
    identity = {'protocol_id': PROTOCOL_ID, 'semantic_protocol_id': manifest['identity']['protocol_id'],
        'source_run_protocol_id': source['protocol_id'], 'provenance': 'new_implementation', 'commit': commit,
        'plan_path': str(destination / 'plan.json'), 'enabled_directory': str(destination / 'enabled'),
        'authority_path': str(AUTHORITY_PATH), 'parent': parent,
        'manifest_path': source['manifest_path'], 'manifest_sha256': MANIFEST_SHA256,
        'manifest_identity_hash': manifest['identity_hash'], 'schedule': schedule,
        'batch_scene_ids': scenes, 'evaluation_scene_ids': manifest['identity']['evaluation_scene_ids'],
        'segments': segments, 'scope': SCOPE, 'global_schedule_offset': 160,
        **{k: source[k] for k in ('prompt', 'prompt_sha256', 'schema', 'schema_sha256', 'timeout_seconds', 'limits')},
        'pricing': pricing, 'pricing_document': document, 'pricing_document_hash': object_hash(document),
        'pricing_source_file_path': str(safe_path(pricing_file)), 'pricing_source_file_sha256': sha256_file(pricing_file),
        'budgets': budgets(pricing), 'implementation_hashes': implementation_hashes(),
        'gt_by_scene': {s: manifest['identity']['gt_by_scene'][s] for s in scenes},
        'checkpoint_sources': refs, 'checkpoint_sources_path': str(source_path),
        'checkpoint_sources_sha256': sha256_file(source_path),
        'reservation_policy': 'full documented context input plus output cap; cumulative USD30 includes all old consumption; not completion cost prediction',
        'new_scope_approved': False, 'account_structured_output_acceptance': 'historical completed160 only; no new probe'}
    plan = {'identity': identity, 'identity_hash': object_hash(identity), 'created_at_utc': utc_now(),
            'api_allowed': False, 'research_api_additional_sends': 0,
            'status': 'CONTINUATION800_READY_AWAITING_EXPLICIT_APPROVAL'}
    destination.mkdir(parents=True)
    write_json(destination / 'plan.json', plan)
    return load_plan(destination / 'plan.json')


def load_plan(path):
    path = safe_path(path)
    if path.is_dir():
        path /= 'plan.json'
    if not path.is_relative_to(CANONICAL_ROOT):
        raise GenerationError('canonical_production_plan_must_remain_in_minecraft_runs')
    plan = read_json(path)
    identity = plan['identity']
    if plan.get('api_allowed') is not False or plan.get('identity_hash') != object_hash(identity):
        raise GenerationError('disabled_plan_identity_mismatch')
    if identity.get('plan_path') != str(path):
        raise GenerationError('canonical_plan_path_mismatch_copied_plans_cannot_activate')
    if (identity.get('enabled_directory') != str(path.parent / 'enabled')
            or identity.get('authority_path') != str(AUTHORITY_PATH)
            or identity.get('protocol_id') != PROTOCOL_ID or identity.get('scope') != SCOPE
            or identity.get('provenance') != 'new_implementation'
            or identity.get('new_scope_approved') is not False or identity.get('global_schedule_offset') != 160):
        raise GenerationError('canonical_destination_or_continuation_scope_mismatch')
    if identity.get('implementation_hashes') != implementation_hashes():
        raise GenerationError('implementation_hash_mismatch')
    parent_plan, manifest = _parent_documents()
    source = parent_plan['identity']
    schedule, scenes, segments = suffix(manifest, parent_plan)
    if (identity.get('manifest_path') != source['manifest_path']
            or identity.get('manifest_sha256') != MANIFEST_SHA256
            or identity.get('manifest_identity_hash') != manifest['identity_hash']
            or identity.get('schedule') != schedule or identity.get('batch_scene_ids') != scenes
            or identity.get('segments') != segments
            or identity.get('semantic_protocol_id') != manifest['identity']['protocol_id']
            or identity.get('source_run_protocol_id') != source['protocol_id']
            or identity.get('evaluation_scene_ids') != manifest['identity']['evaluation_scene_ids']
            or identity.get('gt_by_scene') != {s: manifest['identity']['gt_by_scene'][s] for s in scenes}):
        raise GenerationError('fixed_suffix800_schedule_or_semantic_protocol_mismatch')
    if any(identity.get(k) != source[k] for k in ('prompt', 'prompt_sha256', 'schema', 'schema_sha256', 'timeout_seconds', 'limits')):
        raise GenerationError('generation_semantics_mismatch')
    document = identity['pricing_document']
    if (sha256_file(safe_path(identity['pricing_source_file_path'])) != identity['pricing_source_file_sha256']
            or read_json(safe_path(identity['pricing_source_file_path'])) != document
            or object_hash(document) != identity['pricing_document_hash']
            or _pricing_records(document) != identity['pricing']
            or identity['budgets'] != budgets(identity['pricing'])):
        raise GenerationError('frozen_pricing_or_budget_mismatch')
    _unchanged_pricing(source['pricing'], identity['pricing'])
    parent = identity['parent']
    if (parent.get('plan_path') != str(PARENT_PLAN_PATH) or parent.get('plan_identity_hash') != PARENT_IDENTITY
            or parent.get('plan_sha256') != PARENT_FILE_SHA256 or parent.get('usage') != PARENT_USAGE
            or parent.get('verification_proof_sha256') != PARENT_PROOF_SHA256):
        raise GenerationError('parent_scope_mismatch')
    if (sha256_file(safe_path(identity['checkpoint_sources_path'])) != identity['checkpoint_sources_sha256']
            or read_json(safe_path(identity['checkpoint_sources_path'])) != identity['checkpoint_sources']):
        raise GenerationError('checkpoint_source_reference_changed')
    current_pricing(identity)
    return plan


def refresh_pricing(plan_path, *, pricing_file):
    """Append actual same-terms official evidence without reapproval or new ledger."""
    plan = load_plan(plan_path)
    identity = plan['identity']
    document = read_json(safe_path(pricing_file))
    pricing = _pricing_records(document)
    validate_pricing(pricing, fresh=True)
    if document['verified_at_utc'] > utc_now():
        raise GenerationError('pricing_retrieval_cannot_be_in_the_future')
    _unchanged_pricing(identity['pricing'], pricing)
    from .runner import _authority_locked
    with _authority_locked(identity):
        current_pricing(identity)
        directory = safe_path(Path(identity['plan_path']).parent / 'pricing-revisions')
        directory.mkdir(exist_ok=True)
        prior = sorted(directory.iterdir())
        previous = read_json(prior[-1])['revision_hash'] if prior else identity['pricing_document_hash']
        core = {'version': 'official_pricing_reverification_v1', 'plan_identity_hash': plan['identity_hash'],
                'previous_hash': previous, 'document': document,
                'source_file_sha256': sha256_file(pricing_file), 'recorded_at_utc': utc_now()}
        revision = {**core, 'revision_hash': object_hash(core)}
        target = safe_path(directory / f'{len(prior) + 1:04d}.json')
        with target.open('xb') as stream:
            stream.write(json_bytes(revision)); stream.flush(); os.fsync(stream.fileno())
        current_pricing(identity, fresh=True)
        return {'revision_path': str(target), 'revision_hash': revision['revision_hash'],
                'plan_identity_hash': plan['identity_hash'], 'ledger_and_approval_unchanged': True}
