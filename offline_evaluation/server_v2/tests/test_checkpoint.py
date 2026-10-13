import io
import tarfile
import pytest
from offline_evaluation.server_v2.checkpoint import safe_relative, scan_file, restore_archive, create_checkpoint
from offline_evaluation.server_v2.io_utils import sha256, ROOT

@pytest.mark.parametrize('path', ['../escape', '/absolute', 'ok/../../escape', '.ssh/key', '.env', '.env.local', 'credentials.json', 'x\\..\\z'])
def test_unsafe_paths(path):
    with pytest.raises(ValueError): safe_relative(path)

def test_scan_private_key(tmp_path):
    p=tmp_path/'secret'; p.write_text('-----BEGIN ' + 'PRIVATE KEY-----')
    with pytest.raises(ValueError, match='suppressed'): scan_file(p)

def test_restore_rejects_links_before_writing(tmp_path):
    p=tmp_path/'unsafe.tar.gz'
    with tarfile.open(p,'w:gz') as t:
        entry=tarfile.TarInfo('link'); entry.type=tarfile.SYMTYPE; entry.linkname='/etc/passwd'; t.addfile(entry)
    dest=ROOT/'reports/server_only/unsafe-restore-test'
    with pytest.raises(ValueError): restore_archive(p,dest,sha256(p))
    assert not dest.exists()

def test_snapshot_only_frozen_completed_tasks(tmp_path):
    import json
    run=tmp_path/'run'; run.mkdir()
    task=run/'task.json'; task.write_text('{"done":true}')
    (run/'live.log').write_text('-----BEGIN ' + 'PRIVATE KEY-----')
    (run/'completed.json').write_text(json.dumps({'status':'COMPLETE','active_tasks':0,'files':{'task.json':sha256(task)}}))
    result=create_checkpoint(tmp_path/'checkpoint',run,['README.md'],'a'*40,'fixture-completed-only')
    with tarfile.open(tmp_path/'checkpoint/checkpoint.tar.gz','r:gz') as archive:
        names=archive.getnames()
    assert 'run/task.json' in names and 'run/live.log' not in names
    assert result['completeness']=='PARTIAL'
