"""Explicit offline source-validation boundary; production has no fixture CLI."""
from copy import deepcopy
from pathlib import Path
import socket

import pytest

from offline_evaluation.operations.goal006b import planning
from offline_evaluation.continuation_v1 import runner as original_runner
from offline_evaluation.resmoke_v2 import adapter
from offline_evaluation.server_v2 import providers
from offline_evaluation.server_v2.core import Grid, grid_to_payload
from offline_evaluation.server_v2.generation import object_hash, read_json, write_json, utc_now, digest
from offline_evaluation.server_v2.io_utils import ROOT


@pytest.fixture
def fixture_plan(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('live network/provider transport forbidden in fixture')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(adapter, 'http_transport', forbidden)
    monkeypatch.setattr(providers, 'http_transport', forbidden)
    for name in providers.KEY_NAMES.values():
        monkeypatch.setenv(name, 'fixture_only_never_live')
    root = tmp_path / 'minecraft-runs'; root.mkdir()
    source_root = root / 'original'; source_root.mkdir()
    directory = source_root / 'enabled'; directory.mkdir()
    source_path = source_root / 'plan.json'
    authority = root / 'original-authority.json'
    lock = authority.with_suffix('.json.lock'); lock.write_bytes(b'')
    (directory / 'run.lock').write_bytes(b'')
    parent = root / 'previous'; parent.mkdir(); (parent / 'run.lock').write_bytes(b'')
    document = read_json(ROOT / 'reports/server_only/goal006a/pricing_refresh_20261015T111329Z/official_pricing.json')
    document['verified_on'], document['verified_at_utc'] = utc_now()[:10], utc_now()
    document['retrieval'] = 'OFFLINE_FIXTURE_ONLY_NO_DOCUMENTATION_FETCH'
    for provider in adapter.MODELS:
        document[provider]['verified_on'] = document['verified_on']
    prices = {provider: document[provider] for provider in adapter.MODELS}
    rows = []
    for index in range(800):
        provider = 'openai' if index % 2 == 0 else 'anthropic'
        scene = 'buildings_100_v4_building_085' if index < 784 else 'buildings_100_v4_building_091'
        text = f'fixture building input {index}'
        rows.append({'provider': provider, 'model': adapter.MODELS[provider],
            'scene_id': scene, 'condition': 'O', 'description_model': 'openai',
            'input_id': f'fixture_input_{index:04d}', 'input_text': text, 'input_sha256': digest(text.encode()),
            'schedule_index': index + 160, 'protocol_id': 'fixed_semantic_fixture',
            'prompt_sha256': 'fixture_prompt_sha', 'schema_sha256': 'fixture_schema_sha'})
    grid = Grid({(0, 0, 0): 'stone'})
    gt = {scene: {'grid_sha256': object_hash(grid_to_payload(grid))} for scene in {r['scene_id'] for r in rows}}
    identity = {'plan_path': str(source_path), 'enabled_directory': str(directory), 'authority_path': str(authority),
        'protocol_id': 'source_fixture', 'semantic_protocol_id': 'fixed_semantic_fixture',
        'schedule': rows, 'prompt': 'fixture common prompt', 'prompt_sha256': 'fixture_prompt_sha',
        'schema': {}, 'schema_sha256': 'fixture_schema_sha', 'limits': {}, 'timeout_seconds': 1.0,
        'manifest_path': str(source_root / 'manifest.json'), 'manifest_sha256': 'fixture_manifest_sha',
        'manifest_identity_hash': 'fixture_manifest_identity', 'evaluation_scene_ids': list(gt), 'gt_by_scene': gt,
        'pricing': prices, 'pricing_document': document, 'pricing_document_hash': object_hash(document),
        'parent': {'usage': {'calls': 190, 'retries': 4, 'tokens': 1828481, 'cost_usd_upper_bound': 4.048922,
                            'unknown_usage_attempts': 0}, 'lock_paths': [str(parent / 'run.lock')],
                   'original_files_sha256': {}},
        'budgets': {'max_output_tokens': 32768, 'max_total_tokens': 268042881, 'max_cost_usd': 30.0}}
    source_plan = {'identity': identity, 'identity_hash': object_hash(identity)}
    write_json(source_path, source_plan)
    write_json(source_root / 'manifest.json', {'fixture_only': True})
    entries = []
    for index, record in enumerate(rows[:781]):
        reserve = original_runner._reservation(identity, record)
        charge = providers.usage_upper_cost(record['provider'], {'input_tokens': 10, 'output_tokens': 20}, prices[record['provider']])
        entries.append({'attempt_id': f'{index:032x}', 'local_index': index, 'schedule_index': index + 160,
            'input_id': record['input_id'], 'provider': record['provider'], 'retry': False,
            'request_payload_hash': object_hash(original_runner._payload(identity, record)),
            'status': 'success' if index < 780 else 'failed', 'failure_kind': None if index < 780 else 'auth_failure',
            'http_status_code': None if index < 780 else 401, 'api_response_received': True,
            'usage': {'input_tokens': 10, 'output_tokens': 20} if index < 780 else None,
            'billing': 'known_usage_conservative_cost' if index < 780 else 'worst_case_reservation',
            'charged_input_tokens': charge[0] if index < 780 else reserve[0],
            'charged_output_tokens': charge[1] if index < 780 else reserve[1],
            'charged_cost_usd': charge[2] if index < 780 else reserve[2], 'artifacts': {}})
    ledger = {'version': 'continuation_v1_ledger_v1', 'plan_identity_hash': source_plan['identity_hash'],
              'halt_reason': 'unknown_usage:auth_failure', 'entries': entries}
    write_json(directory / 'ledger.json', ledger)
    write_json(directory / 'approval.json', {'fixture_original_approval': True})
    write_json(directory / 'enabled.json', {'fixture_original_enabled': True})
    write_json(authority, {'fixture_original_authority': True})
    proof_path = tmp_path / 'checkpoint-proof.json'
    from offline_evaluation.server_v2.data import sha256_file
    write_json(proof_path, {'status': 'GITHUB_VERIFIED', 'release_id': planning.CHECKPOINT_RELEASE,
        'draft': True, 'attempted_inputs': 781, 'plan_identity_hash': source_plan['identity_hash'],
        'verifiedledger_sha256': sha256_file(directory / 'ledger.json'),
        'checkpoint_directory': str(tmp_path / 'source-checkpoint'),
        'assets': [{'name': 'continuation-point-in-time.tar.gz', 'asset_id': planning.CHECKPOINT_ARCHIVE_ASSET,
                    'sha256': planning.CHECKPOINT_ARCHIVE_SHA256, 'github_digest': 'sha256:' + planning.CHECKPOINT_ARCHIVE_SHA256}],
        'restoration': {'status': 'VERIFIED'}})
    monkeypatch.setattr(planning, 'CANONICAL_ROOT', root)
    monkeypatch.setattr(planning, 'SOURCE_PLAN', source_path)
    monkeypatch.setattr(planning, 'SOURCE_IDENTITY', source_plan['identity_hash'])
    monkeypatch.setattr(planning, 'PROVENANCE', 'fixture_only')
    # Only the historical original validator is replaced; new child planning,
    # source hashes, accounting, payload comparison and canonical checks execute.
    monkeypatch.setattr(planning.original_plan, 'load_plan', lambda path: read_json(Path(path)))
    monkeypatch.setattr(planning.original_plan, 'verify_parent', lambda parent: None)
    monkeypatch.setattr(original_runner, '_load_enabled', lambda source: (directory, read_json(directory / 'ledger.json')))
    # Other agents may still finish unrelated operational code during fixtures.
    # Stable fixture code hashes do not bypass production's implementation_hashes.
    monkeypatch.setattr(planning, 'implementation_hashes', lambda: {'fixture_only.py': 'fixture_hash'})
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    path = root / 'goal006b-fixture' / 'plan.json'
    child = planning.prepare(path.parent, source_plan_path=source_path,
                             checkpoint_verification_path=proof_path, commit='a' * 40)
    return {'path': path, 'plan': child, 'source_plan': source_plan,
            'source_directory': directory, 'source_before': before, 'source_ledger': ledger,
            'source_path': source_path, 'proof_path': proof_path, 'root': root,
            'grids': {scene: grid for scene in gt}}
