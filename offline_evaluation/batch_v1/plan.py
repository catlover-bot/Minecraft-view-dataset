"""Disabled first-batch plan and additive official-pricing refresh records.

Historical parent validation never requires today's prices. Freshness is checked
only for activation and each future send. A refresh cannot change prices, model
IDs, context limits or existing approvals/ledgers; those changes require a new
explicit scope review and are deliberately not implemented here.
"""
from __future__ import annotations

import dataclasses
import datetime
import os
from pathlib import Path
import re

from ..server_v2.core import Limits
from ..server_v2.data import sha256_file
from ..server_v2.generation import GenerationError, digest, object_hash, read_json, utc_now, write_json, json_bytes
from ..server_v2.io_utils import ROOT
from ..resmoke_v2 import plan as original_plan, runner as original_runner, schema

MODELS = original_plan.MODELS
OUTPUT_CAP = original_plan.OUTPUT_CAP
PROMPT_PATH = original_plan.PROMPT_PATH
PARENT_PLAN_PATH = Path('/root/minecraft-runs/goal004a-ttyfix-20261013T194600Z/plan.json')
PARENT_IDENTITY = 'b522f5bbc1d17526b4366cd27a8f1968379f007a73f38c2a471b840bb6a549f2'
PARENT_FILE_SHA256 = 'c1fbd060b0d7459ad26d74b27994f9f2b891264a21d4835bf81bec32679c1030'
PARENT_USAGE = {'calls': 30, 'retries': 4, 'tokens': 354088,
                'cost_usd_upper_bound': 0.68679325, 'unknown_usage_attempts': 0}
PROTOCOL_ID = 'goal005_fixed_v4_evaluation_batch1_160_v1'
AUTHORITY_PATH = Path('/root/minecraft-runs/.batch-v1-' + PARENT_IDENTITY + '-batch1.json')
PRICING_PATH = ROOT / 'reports/server_only/goal005/official_pricing.json'
CAP_REPORT_PATH = ROOT / 'reports/server_only/goal005/planning/lengths.json'
SCOPE = 'first_fixed10_of_evaluation60_only_160_sends_no_retries'


def safe_path(path):
    return original_runner._checked_path(path)


def implementation_hashes():
    own = [p for p in Path(__file__).parent.rglob('*.py') if 'tests' not in p.parts]
    resmoke = original_plan.implementation_hashes()
    for name in ('cli.py', 'runner.py', 'plan.py'):
        p = ROOT / 'offline_evaluation/resmoke_v2' / name
        resmoke[p.relative_to(ROOT).as_posix()] = sha256_file(p)
    return dict(sorted({**resmoke, **{p.relative_to(ROOT).as_posix(): sha256_file(p) for p in own}}.items()))


def parent_snapshot():
    """Read completed immutable old lineage using its original validators."""
    parent_plan = original_plan.load_plan(PARENT_PLAN_PATH)
    if (parent_plan['identity_hash'] != PARENT_IDENTITY
            or sha256_file(PARENT_PLAN_PATH) != PARENT_FILE_SHA256):
        raise GenerationError('completed_resmoke_parent_identity_changed')
    original_plan.verify_parent(parent_plan['identity']['parent'])
    directory, ledger = original_runner._load_enabled(parent_plan)
    summary = original_runner.execution_summary(parent_plan, ledger)
    usage = {**summary['used']['combined'], 'unknown_usage_attempts': summary['used']['new']['unknown_usage_attempts']}
    if not summary['experiment_complete'] or usage != PARENT_USAGE:
        raise GenerationError('parent_not_complete_or_usage_changed')
    old = parent_plan['identity']['parent']
    old_dir = Path(old['campaign_directory'])
    authority = Path(parent_plan['identity']['authority_path'])
    locks = [old_dir / 'run.lock', old_dir / 'budget.lock']
    locks += [old_dir / 'runs' / p / 'run.lock' for p in sorted(MODELS)]
    locks += [directory / 'run.lock', authority.with_suffix(authority.suffix + '.lock')]
    # Include raw, parsed, voxels, claims, controls, historical CPU results and locks.
    files = [PARENT_PLAN_PATH, authority, authority.with_suffix(authority.suffix + '.lock')]
    files += sorted(p for p in directory.rglob('*') if p.is_file())
    hashes = {str(safe_path(p)): sha256_file(safe_path(p)) for p in files}
    return {'plan_path': str(PARENT_PLAN_PATH), 'plan_identity_hash': PARENT_IDENTITY,
            'plan_sha256': PARENT_FILE_SHA256, 'enabled_directory': str(directory),
            'old_campaign': old, 'usage': usage, 'ledger_sha256': sha256_file(directory / 'ledger.json'),
            'original_files_sha256': hashes, 'lock_paths': [str(p) for p in locks]}


def verify_parent(parent):
    if parent_snapshot() != parent:
        raise GenerationError('parent_changed; never_rebind_or_reset_previous_ledger')


def budgets(pricing):
    maximum = sum(80 * (pricing[p]['context_window_tokens'] + OUTPUT_CAP) for p in MODELS)
    return {'max_new_calls': 160, 'max_total_calls': 190, 'max_new_retries': 0,
            'max_total_retries': 4, 'max_attempts_per_input': 1, 'concurrency': 1,
            'max_cost_usd': 10.0, 'max_tokens': maximum,
            'max_total_tokens': PARENT_USAGE['tokens'] + maximum, 'max_output_tokens': OUTPUT_CAP}


def validate_pricing(pricing, *, fresh=False):
    # The existing fixed-model semantic validator includes supported output cap.
    original_plan.validate_pricing(pricing, fresh=fresh)


def _pricing_records(document):
    if (document.get('research_api_requests_sent') != 0
            or not isinstance(document.get('verified_at_utc'), str)
            or not document.get('retrieval')):
        raise GenerationError('official_pricing_retrieval_evidence_required')
    try:
        stamp = datetime.datetime.fromisoformat(document['verified_at_utc'].replace('Z', '+00:00'))
        if stamp.utcoffset() != datetime.timedelta(0):
            raise ValueError('UTC required')
    except (TypeError, ValueError):
        raise GenerationError('official_pricing_retrieval_utc_required') from None
    pricing = {p: document[p] for p in MODELS}
    validate_pricing(pricing)
    if (document.get('verified_on') != stamp.date().isoformat()
            or any(r['verified_on'] != document['verified_on'] for r in pricing.values())):
        raise GenerationError('pricing_retrieval_date_mismatch')
    return pricing


def _unchanged_pricing(original, refreshed):
    # Only evidence date and textual lifecycle status may change. This keeps
    # rates, context, cap, accepted IDs, billing route and all reservation fields.
    mutable_evidence = {'verified_on', 'documentation_status'}
    if any({k: v for k, v in original[p].items() if k not in mutable_evidence}
           != {k: v for k, v in refreshed[p].items() if k not in mutable_evidence}
           for p in MODELS):
        raise GenerationError('pricing_or_model_limits_changed_explicit_scope_reapproval_required')


def current_pricing(identity, *, fresh=False):
    pricing = identity['pricing']
    previous = identity['pricing_document_hash']
    directory = safe_path(Path(identity['plan_path']).parent / 'pricing-revisions')
    if directory.exists():
        if not directory.is_dir():
            raise GenerationError('invalid_pricing_revision_directory')
        paths = sorted(directory.iterdir())
        for index, path in enumerate(paths, 1):
            if path.name != f'{index:04d}.json':
                raise GenerationError('pricing_revision_sequence_mismatch')
            revision = read_json(safe_path(path))
            core = {k: v for k, v in revision.items() if k != 'revision_hash'}
            if (revision.get('revision_hash') != object_hash(core)
                    or revision.get('previous_hash') != previous
                    or revision.get('plan_identity_hash') != object_hash(identity)):
                raise GenerationError('pricing_revision_identity_mismatch')
            pricing = _pricing_records(revision['document'])
            _unchanged_pricing(identity['pricing'], pricing)
            previous = revision['revision_hash']
    validate_pricing(pricing, fresh=fresh)
    return pricing


def refresh_pricing(plan_path, *, pricing_file):
    """Store actual reverified official evidence; never mutate ledger/approval.

The caller must first fetch the official documents, save the complete evidence
record with actual UTC retrieval time, and retain those sources. This operation
never fetches research/model/token APIs and is not a substitute for that review.
"""
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
        # Exclusive immutable creation. A interrupted partial write fails closed.
        with target.open('xb') as stream:
            stream.write(json_bytes(revision)); stream.flush(); os.fsync(stream.fileno())
        current_pricing(identity, fresh=True)
        return {'revision_path': str(target), 'revision_hash': revision['revision_hash'],
                'plan_identity_hash': plan['identity_hash'], 'ledger_and_approval_unchanged': True}


def prepare(destination, *, pricing_file, commit):
    """Create only a unique canonical disabled plan; no authority or approval."""
    from .inputs import make_manifest
    destination = safe_path(destination)
    if destination.exists() or not destination.is_relative_to(Path('/root/minecraft-runs')):
        raise GenerationError('new_unique_canonical_preparation_directory_required')
    if not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit):
        raise GenerationError('explicit_forty_digit_commit_required')
    if AUTHORITY_PATH.exists():
        raise GenerationError('batch1_lineage_already_activated_resume_original_ledger')
    if safe_path(pricing_file) != PRICING_PATH:
        raise GenerationError('initial_plan_requires_frozen_goal005_official_pricing_record')
    document = read_json(safe_path(pricing_file))
    pricing = _pricing_records(document)  # Disabled planning never requires today's date.
    parent = parent_snapshot()
    manifest = make_manifest()
    mi = manifest['identity']
    cap_report = read_json(CAP_REPORT_PATH)
    if cap_report['manifest_identity_hash'] != manifest['identity_hash'] or cap_report['output_cap'] != OUTPUT_CAP:
        raise GenerationError('cpu_cap_evidence_scope_mismatch')
    cap_assessment = {'evidence_path': str(CAP_REPORT_PATH), 'evidence_sha256': sha256_file(CAP_REPORT_PATH),
                      'fixed60_reference_exceeds_cap_count': cap_report['summary']['reference_exceeds_cap_count'],
                      'batch_reference_exceeds_cap_scenes': [row['scene_id'] for row in cap_report['rows']
                           if row['batch1'] and not row['reference_body_within_output_cap_local_openai']],
                      'targets_dropped_or_truncated': 0, 'output_cap': OUTPUT_CAP,
                      'actual_model_output_and_anthropic_tokens_guaranteed': False}
    identity = {'protocol_id': PROTOCOL_ID, 'provenance': 'new_implementation', 'commit': commit,
        'plan_path': str(destination / 'plan.json'), 'enabled_directory': str(destination / 'enabled'),
        'authority_path': str(AUTHORITY_PATH), 'parent': parent,
        'manifest_path': str(destination / 'full_manifest.json'), 'manifest_identity_hash': manifest['identity_hash'],
        'schedule': mi['batch_schedule'], 'batch_scene_ids': mi['batch_scene_ids'],
        'evaluation_scene_ids': mi['evaluation_scene_ids'], 'scope': SCOPE,
        'prompt': PROMPT_PATH.read_text(), 'prompt_sha256': digest(PROMPT_PATH.read_bytes()),
        'schema': schema.action_schema(), 'schema_sha256': schema.schema_sha256(),
        'pricing': pricing, 'pricing_document': document, 'pricing_document_hash': object_hash(document),
        'pricing_source_file_sha256': sha256_file(pricing_file), 'budgets': budgets(pricing),
        'output_cap_cpu_assessment': cap_assessment,
        'timeout_seconds': 600.0, 'limits': dataclasses.asdict(Limits()),
        'implementation_hashes': implementation_hashes(),
        'gt_by_scene': {s: mi['gt_by_scene'][s] for s in mi['batch_scene_ids']},
        'reservation_policy': 'full documented context input plus output cap; not measured or guaranteed completion cost',
        'remaining_800_authorized': False, 'account_structured_output_acceptance': 'NOT_RUN'}
    plan = {'identity': identity, 'identity_hash': object_hash(identity), 'created_at_utc': utc_now(),
            'api_allowed': False, 'research_api_additional_sends': 0,
            'status': 'BATCH160_READY_AWAITING_EXPLICIT_APPROVAL'}
    destination.mkdir(parents=True)
    write_json(destination / 'full_manifest.json', manifest)
    write_json(destination / 'plan.json', plan)
    return load_plan(destination / 'plan.json')


def load_plan(path):
    """Reject relocated/rehashed copies, edited sources, scope or frozen code."""
    from .inputs import make_manifest
    path = safe_path(path)
    if path.is_dir():
        path /= 'plan.json'
    if not path.is_relative_to(Path('/root/minecraft-runs')):
        raise GenerationError('canonical_production_plan_must_remain_in_minecraft_runs')
    plan = read_json(path)
    identity = plan['identity']
    if plan.get('api_allowed') is not False or plan.get('identity_hash') != object_hash(identity):
        raise GenerationError('disabled_plan_identity_mismatch')
    if identity.get('plan_path') != str(path):
        raise GenerationError('canonical_plan_path_mismatch_copied_plans_cannot_activate')
    if (identity.get('enabled_directory') != str(path.parent / 'enabled')
            or identity.get('authority_path') != str(AUTHORITY_PATH)
            or identity.get('manifest_path') != str(path.parent / 'full_manifest.json')
            or identity.get('protocol_id') != PROTOCOL_ID or identity.get('scope') != SCOPE
            or identity.get('provenance') != 'new_implementation'
            or identity.get('remaining_800_authorized') is not False):
        raise GenerationError('canonical_destination_or_batch_scope_mismatch')
    if identity.get('implementation_hashes') != implementation_hashes():
        raise GenerationError('implementation_hash_mismatch')
    if (identity.get('prompt') != PROMPT_PATH.read_text()
            or identity.get('prompt_sha256') != digest(PROMPT_PATH.read_bytes())
            or identity.get('schema') != schema.action_schema()
            or identity.get('schema_sha256') != schema.schema_sha256()
            or identity.get('limits') != dataclasses.asdict(Limits())
            or identity.get('timeout_seconds') != 600.0):
        raise GenerationError('generation_semantics_mismatch')
    document = identity['pricing_document']
    if (document != read_json(PRICING_PATH)
            or identity['pricing_source_file_sha256'] != sha256_file(PRICING_PATH)
            or object_hash(document) != identity['pricing_document_hash']
            or _pricing_records(document) != identity['pricing']
            or identity['budgets'] != budgets(identity['pricing'])):
        raise GenerationError('frozen_pricing_or_budget_mismatch')
    cap_assessment = identity['output_cap_cpu_assessment']
    if (cap_assessment.get('evidence_path') != str(CAP_REPORT_PATH)
            or cap_assessment.get('evidence_sha256') != sha256_file(CAP_REPORT_PATH)):
        raise GenerationError('cpu_cap_evidence_changed')
    parent = identity['parent']
    if (parent.get('plan_path') != str(PARENT_PLAN_PATH) or parent.get('plan_identity_hash') != PARENT_IDENTITY
            or parent.get('plan_sha256') != PARENT_FILE_SHA256 or parent.get('usage') != PARENT_USAGE):
        raise GenerationError('parent_scope_mismatch')
    manifest = read_json(safe_path(identity['manifest_path']))
    if (manifest.get('identity_hash') != identity['manifest_identity_hash']
            or manifest.get('identity_hash') != object_hash(manifest['identity'])
            or manifest.get('api_allowed') is not False):
        raise GenerationError('full_manifest_hash_mismatch')
    expected = make_manifest()
    if manifest['identity'] != expected['identity']:
        raise GenerationError('manifest_differs_from_frozen_sources_or_fixed_selection')
    mi = manifest['identity']
    if (identity['schedule'] != mi['batch_schedule'] or len(identity['schedule']) != 160
            or identity['batch_scene_ids'] != mi['batch_scene_ids'] or len(mi['batch_scene_ids']) != 10
            or identity['evaluation_scene_ids'] != mi['evaluation_scene_ids'] or len(mi['evaluation_scene_ids']) != 60
            or identity['gt_by_scene'] != {s: mi['gt_by_scene'][s] for s in mi['batch_scene_ids']}):
        raise GenerationError('fixed160_schedule_or_gt_mismatch')
    current_pricing(identity)
    return plan
