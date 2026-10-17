"""Fail-closed, read-only join to immutable Goal007 selected O outcomes.

No old outcome bytes are read until source/scorer/token gates and frozen score
artifact bytes have been verified. This module never reevaluates geometry.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

from ..goal013.source_binding import digest, safe, strict_json
from .associations import BUILDERS, PROVIDERS, SCORE_ID, finite, require

ROOT = Path(__file__).resolve().parents[3]
GOAL007_MANIFEST = ROOT / 'reports/server_only/goal007/MANIFEST.json'
GOAL007_MANIFEST_SHA256 = 'ef05c2ed93247802e9a68a00d700d39a8934d6231c7787315ab37b01d2467714'
GOAL007_SLOTS_PRODUCT = 'data/final_slots_960.json'
GOAL007_SLOTS_SHA256 = 'f0960ef78f21bd42c78cfeb709cf9505bbf34d27f60a4986824ea763cc061b2d'
IDENTITY_FIELDS = ('scene_id', 'description_provider', 'O_input_id', 'O_input_sha256',
                   'description_source_sha256', 'summary_sha256')
KNOWN_MODEL_OUTPUT_FAILURES = frozenset(('parse_failure', 'validation_failure', 'execution_failure', 'truncation'))
MAX_JSON_BYTES = 32 * 1024**2


def _read_bound(path, expected_sha256, expected_bytes=None):
    path = safe(path)
    require(re.fullmatch(r'[0-9a-f]{64}', expected_sha256 or '') is not None, 'actual SHA256 required')
    require(path.is_file() and path.stat().st_size <= MAX_JSON_BYTES, 'bound JSON missing or too large')
    raw = path.read_bytes()
    require(digest(raw) == expected_sha256, 'bound artifact SHA256 differs: ' + str(path))
    require(expected_bytes is None or len(raw) == expected_bytes, 'bound artifact bytes differ')
    return strict_json(raw), {'path': str(path), 'sha256': digest(raw), 'bytes': len(raw)}


def _artifact(descriptor):
    require(isinstance(descriptor, dict) and type(descriptor.get('bytes')) is int
            and descriptor['bytes'] >= 0, 'explicit artifact descriptor required')
    return _read_bound(Path(descriptor['path']), descriptor['sha256'], descriptor['bytes'])


def _rows(value):
    rows = value if isinstance(value, list) else value.get('rows') if isinstance(value, dict) else None
    require(isinstance(rows, list) and all(isinstance(r, dict) for r in rows), 'row-list artifact required')
    return rows


def _index(rows, label):
    result = {(r['scene_id'], r['description_provider']): r for r in rows}
    require(len(result) == len(rows), label + ' duplicate identity')
    return result


def _ordered_images(row):
    images = row.get('images_ordered')
    require(isinstance(images, list) and len(images) == 6 and all(isinstance(image, dict) for image in images),
            'six ordered image identities required')
    identities = [(image['path'], image['sha256'], image['bytes']) for image in images]
    require(len({identity[0] for identity in identities}) == 6, 'six distinct requested image paths required')
    return identities


def validate_score_binding(score_rows, source_rows, *, fixture=False):
    """No outcome access; check all planned score/source identities, including nulls."""
    source = _index(source_rows, 'source')
    scores = _index(score_rows, 'score')
    scenes = sorted({key[0] for key in source})
    require(fixture or len(scenes) == 60, 'production source requires 60 planned buildings')
    require(scenes and set(source) == set(scores) == {(s, p) for s in scenes for p in PROVIDERS}, 'planned source/score frame differs')
    for key, row in scores.items():
        origin = source[key]
        require(origin.get('binding_status') == 'PASS', 'source binding not passed')
        require(all(field in row and row[field] == origin[field] for field in IDENTITY_FIELDS), 'score/source identity differs')
        require(_ordered_images(row) == _ordered_images(origin), 'score/source ordered image identities differ')
        require(origin['O_input_sha256'] == origin['description_source_sha256'], 'original O bytes differ from description')
        require(row.get('score_id') == SCORE_ID, 'only fixed Goal013A scorer allowed')
        if row.get('score_status') == 'COMPUTATION_VALID':
            require(finite(row.get('score')), 'valid score must be finite')
        else:
            require(row.get('score') is None, 'invalid score must remain null')
        if row.get('token_coverage_status') == 'VERIFIED':
            require(type(row.get('summary_token_count_untruncated')) is int
                    and row['summary_token_count_untruncated'] > 0, 'verified positive token count required')
        if origin.get('summary_status') != 'VALID_ORIGINAL_SUMMARY':
            require(row.get('score') is None and row.get('score_status') != 'COMPUTATION_VALID', 'missing original summary cannot be scored')
    return scenes


def _slot_outcome(slot):
    status = slot['status']
    require(type(slot.get('usage_known')) is bool and type(slot.get('completed_request_outcome')) is bool,
            'known-usage/completed-outcome flags required')
    require(slot.get('attempted') is True and isinstance(slot.get('attempt_id'), str)
            and slot['attempt_id'] and slot.get('selected_attempt_id') == slot['attempt_id']
            and slot.get('selected_final_observation') is True, 'selected observation identity required')
    if 'http_status' in slot and 'http_status_code' in slot:
        require(slot['http_status'] == slot['http_status_code'], 'HTTP aliases disagree')
    http_status = slot.get('http_status', slot.get('http_status_code'))
    metrics, value = slot.get('metrics'), None
    if status == 'success':
        require(slot['usage_known'] and slot['completed_request_outcome'], 'success requires a known completed outcome')
        if metrics is not None:
            require(isinstance(metrics, dict) and metrics.get('metric_version') == 'server_v2.2_strict_translation', 'frozen metric version required')
            aligned = metrics.get('aligned')
            require(aligned is None or isinstance(aligned, dict), 'invalid aligned metrics')
            value = aligned.get('occupancy_iou') if aligned is not None else None
            require(value is None or (finite(value) and 0 <= value <= 1), 'invalid aligned occupancy IoU')
    else:
        require(metrics is None, 'non-success outcome cannot contain metrics')
    eligible = (status == 'failed' and slot['usage_known'] and slot['completed_request_outcome']
                and slot.get('failure_kind') in KNOWN_MODEL_OUTPUT_FAILURES and http_status not in (401, 520))
    return {
        'status': status, 'aligned_occupancy_iou': value,
        'known_failure0_eligible': eligible, 'failure_kind': slot.get('failure_kind'),
        'usage_known': slot['usage_known'], 'completed_request_outcome': slot['completed_request_outcome'],
        'global_schedule_index': slot['global_schedule_index'], 'selected_attempt_id': slot['selected_attempt_id'],
        'original_attempt_id': slot.get('original_attempt_id'), 'source_plan_identity': slot.get('source_plan_identity'),
        'source_entry_hash': slot.get('source_entry_hash'), 'input_id': slot['input_id'], 'input_sha256': slot['input_sha256'],
    }


def bind_selected_outcomes(score_rows, source_rows, slots, *, fixture=False):
    """Pure join for already-gated production input or self-contained fixtures."""
    scenes = validate_score_binding(score_rows, source_rows, fixture=fixture)
    indices = [r['global_schedule_index'] for r in slots]
    require(len(set(indices)) == len(indices), 'attempts must not duplicate selected slots')
    if not fixture:
        require(len(slots) == 960 and indices == list(range(960)), 'canonical 960 selected slots in global order required')
        require(Counter(r['status'] for r in slots) == {'success': 934, 'failed': 25, 'operational_unmeasurable': 1}, 'frozen selected-slot status scope differs')
    originals = [r for r in slots if r['condition'] == 'O']
    by_key = {(r['scene_id'], r['description_model'], r['provider']): r for r in originals}
    require(len(by_key) == len(originals) and set(by_key) == {(s, p, b) for s in scenes for p in PROVIDERS for b in BUILDERS}, 'selected O scope differs')
    result = []
    for row in sorted(score_rows, key=lambda r: (r['scene_id'], r['description_provider'])):
        outcomes = {}
        for builder in BUILDERS:
            slot = by_key[(row['scene_id'], row['description_provider'], builder)]
            require(slot['input_id'] == row['O_input_id'] and slot['input_sha256'] == row['O_input_sha256']
                    == row['description_source_sha256'], 'selected O input/source identity differs')
            outcomes[builder] = _slot_outcome(slot)
        # Deliberate allowlist: no raw summary, GT, voxel, artifact or usage payload.
        result.append({field: row[field] for field in IDENTITY_FIELDS} | {
            'score': row['score'], 'score_status': row['score_status'], 'score_id': row['score_id'],
            'token_coverage_status': row.get('token_coverage_status'),
            'summary_token_count_untruncated': row.get('summary_token_count_untruncated'), 'outcomes': outcomes})
    return result, scenes


def load_frozen_outcomes(freeze_path, *, expected_freeze_sha256):
    """Read exact source/score/gate artifacts, then the manifest-bound old slots.

    The expected freeze SHA must be recorded externally before this call; the
    function does not manufacture or update freeze receipts or any old file.
    """
    freeze, freeze_descriptor = _read_bound(Path(freeze_path), expected_freeze_sha256)
    require(freeze.get('status') == 'PASS' and freeze.get('source_binding_status') == 'PASS'
            and freeze.get('scorer_validation_status') == 'PASS', 'source/scorer/freeze gates must pass')
    require(freeze.get('score_id') == SCORE_ID, 'fixed scorer ID required')
    for field in ('code_commit', 'spec_commit'):
        require(isinstance(freeze.get(field), str) and re.fullmatch(r'[0-9a-f]{40}', freeze[field]) is not None,
                'committed code/spec identity required')
    require(freeze['code_commit'] != freeze['spec_commit'], 'code and selected specification require separate commits')
    gate, gate_descriptor = _artifact(freeze['representativeness_artifact'])
    require(gate.get('status') == 'PASS' and gate.get('overall', {}).get('status') == 'PASS'
            and set(gate.get('providers', {})) == set(PROVIDERS)
            and all(gate['providers'][p].get('status') == 'PASS' for p in PROVIDERS), 'overall and each provider representativeness gate must pass')
    scores, score_descriptor = _artifact(freeze['score_artifact'])
    sources, source_descriptor = _artifact(freeze['source_binding_artifact'])
    scores, sources = _rows(scores), _rows(sources)
    validate_score_binding(scores, sources)

    # The first old-outcome access is below every preceding gate and hash check.
    manifest, manifest_descriptor = _read_bound(GOAL007_MANIFEST, GOAL007_MANIFEST_SHA256)
    canonical_root = safe(Path(manifest['output_root']))
    canonical, _ = _read_bound(canonical_root / 'MANIFEST.json', GOAL007_MANIFEST_SHA256)
    require(manifest == canonical and manifest.get('source_inventories_before_after_equal') is True,
            'canonical Goal007 manifest/protection differs')
    descriptor = manifest['products'][GOAL007_SLOTS_PRODUCT]
    require(descriptor['sha256'] == GOAL007_SLOTS_SHA256, 'fixed selected-slot artifact differs')
    slots, slots_descriptor = _read_bound(canonical_root / GOAL007_SLOTS_PRODUCT, descriptor['sha256'], descriptor['bytes'])
    rows, scenes = bind_selected_outcomes(scores, sources, slots)
    receipt = {
        'schema': 'goal013a_outcome_binding_v1', 'status': 'PASS', 'score_freeze_verified': True,
        'outcome_binding_status': 'PASS', 'representativeness_status': 'PASS',
        'freeze_receipt': freeze_descriptor, 'score_artifact': score_descriptor,
        'source_binding_artifact': source_descriptor, 'representativeness_artifact': gate_descriptor,
        'goal007_manifest': manifest_descriptor, 'selected_slots_artifact': slots_descriptor,
        'code_commit': freeze['code_commit'], 'spec_commit': freeze['spec_commit'],
        'planned_buildings': len(scenes), 'planned_descriptions': len(rows), 'bound_selected_O_slots': len(rows) * 2,
        'outcome_metric': 'saved aligned occupancy IoU', 'outcomes_recomputed': False,
        'attempts_counted_as_additional_observations': False, 'GT_files_opened': 0,
        'source_writes': 0, 'research_api_requests_sent': 0,
    }
    return rows, scenes, receipt
