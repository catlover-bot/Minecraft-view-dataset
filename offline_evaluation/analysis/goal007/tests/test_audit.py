"""Offline fixture checks for evidence gate fail-closed boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import pytest

from offline_evaluation.analysis.goal007 import audit


def metric():
    return {'metric_version': 'server_v2.2_strict_translation', 'alignment': {
        'mode': 'strict_translation', 'radius': None, 'globally_optimal_unrestricted': True,
        'search_complete': True, 'status': 'PASS', 'rotation': False, 'scale': False,
        'clipping': False, 'all_aligned_metrics_share_shift': True}}


def test_exact_cached_metric_is_accepted():
    audit.check_metric(metric())


@pytest.mark.parametrize('field,value', [('radius', 8), ('rotation', True), ('scale', True), ('clipping', True),
    ('search_complete', False), ('globally_optimal_unrestricted', False), ('all_aligned_metrics_share_shift', False)])
def test_approximate_or_inconsistent_cached_metrics_rejected(field, value):
    m = metric(); m['alignment'][field] = value
    with pytest.raises(audit.AuditError):
        audit.check_metric(m)


def test_wrong_metric_version_rejected():
    m = metric(); m['metric_version'] = 'legacy'
    with pytest.raises(audit.AuditError):
        audit.check_metric(m)


@pytest.mark.parametrize('mutation', ['content', 'addition', 'deletion', 'symlink'])
def test_inventory_checks_content_and_exact_tree(tmp_path, mutation):
    (tmp_path / 'ledger.json').write_text('{}')
    before = audit.inventory([], [tmp_path])
    assert audit.verify_inventory(before) == 1
    if mutation == 'content':
        (tmp_path / 'ledger.json').write_text('{"altered":true}')
    elif mutation == 'addition':
        (tmp_path / 'new.json').write_text('{}')
    elif mutation == 'deletion':
        (tmp_path / 'ledger.json').unlink()
    else:
        (tmp_path / 'alias').symlink_to(tmp_path / 'ledger.json')
    with pytest.raises(audit.AuditError):
        audit.verify_inventory(before)


def test_existing_audit_output_never_overwritten(tmp_path):
    path = tmp_path / 'proof.json'
    audit.write_new(path, {'old': True})
    with pytest.raises(FileExistsError):
        audit.write_new(path, {'old': False})
    assert json.loads(path.read_text()) == {'old': True}


def test_python_network_disabled_and_restored():
    original = socket.socket.connect
    with audit.no_python_network():
        with pytest.raises(audit.AuditError, match='network_disabled'):
            socket.socket.connect(None, ('127.0.0.1', 1))
    assert socket.socket.connect is original


def recovery_fixture(tmp_path):
    label = 'supplement3'; checkpoint = audit.CHECKPOINTS[label]
    directory = tmp_path / 'redownload'; directory.mkdir()
    assets = []
    for number, name in enumerate(['supplement-point-in-time.tar.gz', 'FILES.json', 'snapshot.json', 'SOURCE_REFERENCE.json', 'RESTORE.md', 'SHA256SUMS'], 1):
        path = directory / name; path.write_bytes(b'x')
        digest = audit.sha256_file(path)
        assets.append({'asset_id': number, 'name': name, 'sha256': digest, 'github_digest': 'sha256:' + digest, 'bytes': 1})
    source = {'plan_identity_hash': audit.EXPECTED[label][0], 'ledger_sha256': 'ledger'}
    report = {'status': 'GITHUB_VERIFIED', 'draft': True, 'release_id': checkpoint['release_id'],
        'plan_identity_hash': source['plan_identity_hash'], 'verifiedledger_sha256': 'ledger',
        'attempted_inputs': 3, 'current_attempted_inputs': 3, 'uploaded_assets': [], 'research_api_requests_sent': 0,
        'assets': assets, 'verification_directory': str(tmp_path), 'archive': assets[0]['name'],
        'copied_plan_load_rejection': 'canonical_path', 'current_ledger_extends_restored_prefix': True}
    return report, source, assets[0]['sha256']


@pytest.mark.parametrize('mutation', ['status', 'digest', 'upload', 'ledger', 'asset_duplicate', 'download_content', 'copied_plan'])
def test_checkpoint_corruption_stops_gate(tmp_path, monkeypatch, mutation):
    report, source, digest = recovery_fixture(tmp_path)
    monkeypatch.setitem(audit.CHECKPOINTS['supplement3'], 'archive_sha256', digest)
    audit.validate_recovery('supplement3', report, source)
    if mutation == 'status': report['status'] = 'LOCAL_VERIFIED'
    elif mutation == 'digest': report['assets'][0]['github_digest'] = 'sha256:wrong'
    elif mutation == 'upload': report['uploaded_assets'] = [{'asset_id': 44}]
    elif mutation == 'ledger': report['verifiedledger_sha256'] = 'other'
    elif mutation == 'asset_duplicate': report['assets'][-1] = report['assets'][0]
    elif mutation == 'download_content': (tmp_path / 'redownload/FILES.json').write_bytes(b'y')
    else: report['copied_plan_load_rejection'] = None
    with pytest.raises(audit.AuditError):
        audit.validate_recovery('supplement3', report, source)


def test_partial_or_failed_gate_rejected():
    with pytest.raises(audit.AuditError, match='not_passed'):
        audit.verify_gate({'version': audit.VERSION, 'status': 'STOP', 'research_api_requests_sent': 0})


@pytest.mark.parametrize('field', ['plan_path', 'plan_identity_hash', 'ledger_sha256', 'approval_path',
                                 'authority_sha256', 'evaluation_sha256', 'code_commit', 'attempted'])
def test_forged_source_descriptor_cannot_replay_valid_inventory(tmp_path, monkeypatch, field):
    source_dir = tmp_path / 'canonical'; source_dir.mkdir()
    enabled = source_dir / 'enabled'; enabled.mkdir()
    identity = {'plan_path': str(source_dir / 'plan.json'), 'enabled_directory': str(enabled),
                'authority_path': str(source_dir / 'authority.json'), 'commit': 'frozen-commit', 'protocol_id': 'fixture'}
    plan = {'identity': identity, 'identity_hash': audit.object_hash(identity)}
    ledger = {'entries': [{}], 'halt_reason': None}
    audit.write_new(source_dir / 'plan.json', plan)
    audit.write_new(source_dir / 'authority.json', {'scope': 'fixed'})
    audit.write_new(enabled / 'ledger.json', ledger)
    audit.write_new(enabled / 'approval.json', {'scope': 'fixed'})
    audit.write_new(enabled / 'evaluation.json', {'rows': []})
    paths, plans, ledgers = {'fixture': source_dir / 'plan.json'}, {'fixture': plan}, {'fixture': ledger}
    monkeypatch.setattr(audit, 'EXPECTED', {'fixture': (plan['identity_hash'], 1)})
    monkeypatch.setattr(audit, 'CHECKPOINTS', {})
    monkeypatch.setattr(audit, 'sources_from_chain', lambda: (paths, plans))
    source = audit._output_descriptors(paths, plans, ledgers)
    saved = audit.inventory([], [source_dir]); proof = tmp_path / 'inventory.json'; audit.write_new(proof, saved)
    report = {'version': audit.VERSION, 'status': 'PASS', 'research_api_requests_sent': 0,
        'inventory_path': str(proof), 'inventory_sha256': audit.sha256_file(proof), 'output_sha256': {},
        'manifest_path': str(source_dir / 'plan.json'), 'manifest_sha256': audit.sha256_file(source_dir / 'plan.json'),
        'sources': source, 'github_checkpoints': {}}
    audit.verify_gate(report)
    report['sources']['fixture'][field] = 'forged'
    with pytest.raises(audit.AuditError):
        audit.verify_gate(report)
