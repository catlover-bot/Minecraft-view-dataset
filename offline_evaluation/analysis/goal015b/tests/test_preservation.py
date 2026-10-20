"""Synthetic byte/absence checks only; no old lock, environment or API use."""
from contextlib import nullcontext
import json

import pytest

from offline_evaluation.analysis.goal015b import preservation as p


def test_changed_input_is_reported_and_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'parent_lock', nullcontext)
    source = tmp_path/'synthetic-input.json'; source.write_text('{}')
    before = tmp_path/'before.json'; p.snapshot_paths([source], before)
    ok = p.finalize_snapshot(before, tmp_path/'unchanged.json')
    assert ok['status'] == 'PASS' and ok['checked_files'] == 1
    source.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='inputs changed'):
        p.finalize_snapshot(before, tmp_path/'changed.json')
    failed = json.loads((tmp_path/'changed.json').read_text())
    assert failed['failures'] == [{'path':str(source),'reason':'descriptor_changed'}]


def test_new_absence_control_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'parent_lock', nullcontext)
    absent = tmp_path/'synthetic-control'; before = tmp_path/'before.json'
    p.write_exclusive(before, p._snapshot(p.Inputs(), [absent]))
    absent.write_text('fixture marker, not execution authority')
    with pytest.raises(ValueError, match='inputs changed'):
        p.finalize_snapshot(before, tmp_path/'after.json')
    assert json.loads((tmp_path/'after.json').read_text())['failures'][0]['reason'] == 'execution_control_appeared'


def test_duplicate_metadata_and_symlink_refused(tmp_path):
    with pytest.raises(ValueError, match='Duplicate'):
        p.strict_json('{"id":1,"id":2}')
    target = tmp_path/'target'; target.write_bytes(b'synthetic')
    link = tmp_path/'link'; link.symlink_to(target)
    with pytest.raises(ValueError, match='Non-symlink'):
        p.descriptor(link)


def test_output_is_exclusive_and_old_root_refused(tmp_path):
    output = tmp_path/'only-new.json'; p.write_exclusive(output, {'fixture':True})
    with pytest.raises(FileExistsError):
        p.write_exclusive(output, {'fixture':'must not overwrite'})
    assert json.loads(output.read_text()) == {'fixture':True}
    with pytest.raises(ValueError, match='Only new Goal015B'):
        p.write_exclusive(p.PARENT/'must-never-be-created-goal015b.json', {})
    assert not (p.PARENT/'must-never-be-created-goal015b.json').exists()
