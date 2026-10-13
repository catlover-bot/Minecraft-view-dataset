import io
import tarfile
import pytest
from offline_evaluation.server_v2.checkpoint import safe_relative, scan_file, restore_archive
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
