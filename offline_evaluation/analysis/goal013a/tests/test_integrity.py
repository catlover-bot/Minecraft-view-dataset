"""Synthetic protection fixtures; no research sources or old environments used."""
import hashlib
import json
from pathlib import Path

import pytest

from offline_evaluation.analysis.goal013a import integrity as guard


def test_binary_gt_role_is_hashed_without_parsing(tmp_path):
    path = tmp_path/'gt.npy'; raw = b'\xff\x00not JSON or a loadable array'
    path.write_bytes(raw)
    assert guard.descriptor(path) == {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def test_tree_records_absence_empty_directories_and_symlink_targets(tmp_path):
    missing = tmp_path/'missing'
    assert guard.tree_inventory(missing) == {str(missing): {'kind': 'absent'}}
    root = tmp_path/'cache'; root.mkdir(); (root/'empty').mkdir()
    (root/'file').write_bytes(b'x'); (root/'link').symlink_to('file')
    tree = guard.tree_inventory(root)
    assert tree[str(root/'empty')] == {'kind': 'directory'}
    assert tree[str(root/'link')] == {'symlink_target': 'file'}
    assert tree[str(root/'file')]['sha256'] == hashlib.sha256(b'x').hexdigest()


@pytest.mark.parametrize('mutation', ['file_bytes', 'cache_member', 'environment_symlink', 'canonical_member', 'missing_cache_created'])
def test_snapshot_detects_changes(tmp_path, monkeypatch, mutation):
    old = tmp_path/'source'; old.write_bytes(b'original')
    env = tmp_path/'venv'; env.mkdir(); (env/'python').symlink_to('/usr/bin/python3')
    cache = tmp_path/'cache'; cache.mkdir(); missing = tmp_path/'absent_cache'
    canonical = tmp_path/'canonical'; canonical.mkdir()
    before = {'schema': 'goal013a_protected_snapshot_v1', 'inventory': {str(old): guard.descriptor(old)},
              'tracked_paths': ['old.py'], 'main_head': 'fixture-main',
              'environment_roots': [str(env)], 'environment_inventory': guard.tree_inventory(env),
              'cache_roots': [str(cache), str(missing)],
              'cache_inventory': {**guard.tree_inventory(cache), **guard.tree_inventory(missing)},
              'goal013_canonical_root': str(canonical), 'goal013_canonical_tree': guard.tree_inventory(canonical)}
    snapshot = tmp_path/'snapshot.json'; snapshot.write_text(json.dumps(before))
    monkeypatch.setattr(guard, 'SNAPSHOT_SHA256', hashlib.sha256(snapshot.read_bytes()).hexdigest())
    monkeypatch.setattr(guard, 'tracked', lambda: ['old.py'])
    monkeypatch.setattr(guard, 'git', lambda *args: 'fixture-main')
    assert guard.verify_snapshot(snapshot) == before
    if mutation == 'file_bytes': old.write_bytes(b'mutated')
    elif mutation == 'cache_member': (cache/'new').write_text('new')
    elif mutation == 'environment_symlink':
        (env/'python').unlink(); (env/'python').symlink_to('/usr/bin/other')
    elif mutation == 'canonical_member': (canonical/'new').write_text('new')
    elif mutation == 'missing_cache_created': missing.mkdir()
    with pytest.raises(guard.IntegrityError): guard.verify_snapshot(snapshot)


def test_guard_rejects_credential_path_symlink_parent_and_overwrite(tmp_path):
    with pytest.raises(guard.IntegrityError): guard.safe(Path('/root/.aws/credentials'))
    target = tmp_path/'target'; target.mkdir(); link = tmp_path/'link'; link.symlink_to(target, target_is_directory=True)
    with pytest.raises(guard.IntegrityError): guard.safe(link/'file')
    output = tmp_path/'receipt.json'; guard.write_new(output, {'status': 'PASS'})
    with pytest.raises(FileExistsError): guard.write_new(output, {'status': 'PASS'})


def test_protected_context_rechecks_after_body_failure(tmp_path, monkeypatch):
    snapshot = tmp_path/'snapshot.json'; snapshot.write_text('{"inventory":{},"schema":"goal013a_protected_snapshot_v1"}')
    monkeypatch.setattr(guard, 'SNAPSHOT_SHA256', hashlib.sha256(snapshot.read_bytes()).hexdigest())
    calls = []
    monkeypatch.setattr(guard, 'verify_snapshot', lambda path: calls.append(path))
    with pytest.raises(RuntimeError, match='body failed'):
        with guard.protected_sources(snapshot) as locks:
            assert locks == 0
            raise RuntimeError('body failed')
    assert calls == [snapshot, snapshot]


def test_snapshot_set_cannot_be_silently_replaced(tmp_path, monkeypatch):
    snapshot = tmp_path/'snapshot.json'; snapshot.write_text('{"inventory":{},"schema":"goal013a_protected_snapshot_v1"}')
    monkeypatch.setattr(guard, 'SNAPSHOT_SHA256', hashlib.sha256(snapshot.read_bytes()).hexdigest())
    snapshot.write_text('{"inventory":{},"schema":"goal013a_protected_snapshot_v1","replaced":true}')
    with pytest.raises(guard.IntegrityError, match='snapshot itself changed'):
        guard.load_snapshot(snapshot)
