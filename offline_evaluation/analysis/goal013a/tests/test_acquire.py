"""Offline acquisition-boundary fixtures. No HTTP, model or research sources."""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.error
import zipfile

import pytest

from offline_evaluation.analysis.goal013a import acquire as a


class Response(io.BytesIO):
    status = 200

    def __init__(self, raw, url, length=True):
        super().__init__(raw)
        self.url = url
        self.headers = {'Content-Length': str(len(raw))} if length else {}
        self.read_sizes = []

    def geturl(self):
        return self.url

    def read(self, amount=-1):
        assert amount >= 0, 'unbounded body read'
        self.read_sizes.append(amount)
        return super().read(amount)


@pytest.fixture
def fetcher(tmp_path, monkeypatch):
    monkeypatch.setattr(a.shutil, 'disk_usage', lambda path: SimpleNamespace(free=20*1024**3))
    root = tmp_path/'tools'; root.mkdir()
    f = a.Fetcher(root)
    def no_network(*args, **kwargs):
        raise AssertionError('unexpected network operation')
    f.opener = SimpleNamespace(open=no_network)
    return f


def test_fixed_source_and_dynamic_metadata_have_distinct_provenance():
    source = 'https://raw.githubusercontent.com/openai/CLIP/'+a.CLIP+'/clip/clip.py'
    assert a.received_status(source, None) == 'RECEIVED_HTTPS_REVISION_BOUND'
    assert a.received_status('https://pypi.org/pypi/filelock/json', None) == 'RECEIVED_HTTPS_HASHED_METADATA'
    assert a.received_status('https://download.pytorch.org/whl/cpu/torch/', None) == 'RECEIVED_HTTPS_HASHED_METADATA'
    assert a.received_status(a.WEIGHT_URL, a.WEIGHT_SHA) == 'VERIFIED'


def test_redirect_and_final_bodies_counted(fetcher):
    origin = 'https://pypi.org/example'; final = 'https://files.pythonhosted.org/example'
    redirect_body = Response(b'xy', origin)
    final_body = Response(b'abc', final)
    def open_redirect(req, timeout):
        forwarded = fetcher.redirect_handler.redirect_request(req, redirect_body, 302, 'Found', redirect_body.headers, final)
        assert forwarded.full_url == final
        return final_body
    fetcher.opener = SimpleNamespace(open=open_redirect)
    result = fetcher.get(origin, fetcher.root/'artifact', hashlib.sha256(b'abc').hexdigest())
    assert result.read_bytes() == b'abc'
    receipt = json.loads((fetcher.root/'DOWNLOAD_RECEIPT.json').read_text())
    assert receipt['http_requests'] == 2
    assert receipt['response_body_bytes'] == 5
    assert receipt['events'][0]['bytes'] == 3
    assert receipt['events'][0]['redirects'][0]['body_bytes'] == 2


def test_unknown_length_stops_without_budget_overshoot(fetcher, monkeypatch):
    monkeypatch.setattr(a, 'LIMIT', 5)
    response = Response(b'0123456789', 'https://pypi.org/example', length=False)
    fetcher.opener = SimpleNamespace(open=lambda *args, **kwargs: response)
    with pytest.raises(RuntimeError, match='download byte limit'):
        fetcher.get(response.url, fetcher.root/'partial')
    assert fetcher.total == 5
    assert (fetcher.root/'partial').read_bytes() == b'01234'
    assert response.read_sizes == [5]
    assert fetcher.events[0]['status'] == 'FAILED'


def test_declared_body_larger_than_remaining_is_not_read(fetcher, monkeypatch):
    monkeypatch.setattr(a, 'LIMIT', 5)
    response = Response(b'0123456789', 'https://pypi.org/example')
    fetcher.opener = SimpleNamespace(open=lambda *args, **kwargs: response)
    with pytest.raises(RuntimeError, match='would be exceeded'):
        fetcher.get(response.url, fetcher.root/'partial')
    assert fetcher.total == 0 and response.read_sizes == []


def test_http_error_body_is_counted_and_never_success(fetcher):
    url = 'https://pypi.org/example'
    def failed(*args, **kwargs):
        raise urllib.error.HTTPError(url, 404, 'Not Found', {'Content-Length': '4'}, io.BytesIO(b'nope'))
    fetcher.opener = SimpleNamespace(open=failed)
    with pytest.raises(urllib.error.HTTPError): fetcher.get(url, fetcher.root/'missing')
    assert fetcher.total == 4
    assert fetcher.events[0]['http_error_body_bytes'] == 4
    assert fetcher.events[0]['http_error_status'] == 404
    assert fetcher.events[0]['status'] == 'FAILED'


def test_hash_mismatch_keeps_failure_evidence(fetcher):
    response = Response(b'abc', 'https://pypi.org/example')
    fetcher.opener = SimpleNamespace(open=lambda *args, **kwargs: response)
    with pytest.raises(RuntimeError, match='SHA256 mismatch'):
        fetcher.get(response.url, fetcher.root/'bad', '0'*64)
    assert fetcher.events[0]['status'] == 'FAILED'
    assert fetcher.events[0]['measured_sha256'] == hashlib.sha256(b'abc').hexdigest()
    assert (fetcher.root/'bad').read_bytes() == b'abc'


def test_resume_legacy_receipt_reuses_only_verified_identical_bytes(fetcher):
    path = fetcher.root/'old'; path.write_bytes(b'old')
    fetcher.events = [{'url': 'https://pypi.org/example', 'local_path': str(path), 'status': 'RECEIVED_HTTPS_REVISION_BOUND',
                       'bytes': 3, 'measured_sha256': hashlib.sha256(b'old').hexdigest()}]
    assert fetcher.get('https://pypi.org/example', path) == path
    assert fetcher.requests == 0
    path.write_bytes(b'new')
    with pytest.raises(AssertionError): fetcher.get('https://pypi.org/example', path)


def test_root_escape_and_symlinks_rejected_before_network(fetcher, tmp_path):
    with pytest.raises(ValueError, match='escaped'):
        fetcher.get('https://pypi.org/example', tmp_path/'outside')
    (fetcher.root/'linked').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        fetcher.get('https://pypi.org/example', fetcher.root/'linked/artifact')
    root_link = tmp_path/'resume'; root_link.symlink_to(fetcher.root, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'): a.Fetcher(root_link)
    (fetcher.root/'DOWNLOAD_RECEIPT.json').symlink_to(tmp_path/'outside-receipt')
    with pytest.raises(ValueError, match='symlink'): fetcher.save()
    assert not (tmp_path/'outside').exists() and not (tmp_path/'outside-receipt').exists()


@pytest.mark.parametrize('url', ['http://pypi.org/example', 'https://evil.example/x',
                               'https://user:secret@pypi.org/x', 'https://pypi.org/x?token=secret'])
def test_unapproved_url_rejected(url):
    with pytest.raises(ValueError): a.check_url(url)


def test_both_license_spellings_are_hashed(tmp_path):
    path = tmp_path/'synthetic.whl'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('pkg.dist-info/LICENCE', b'MIT and MPL fixture')
        archive.writestr('pkg.dist-info/LICENSE.txt', b'MIT fixture')
        archive.writestr('pkg/module.py', b'not a license')
    with zipfile.ZipFile(path) as archive:
        result = a.license_members(archive)
    assert {r['archive_member'] for r in result} == {'pkg.dist-info/LICENCE', 'pkg.dist-info/LICENSE.txt'}
    assert result[0]['sha256'] == hashlib.sha256(b'MIT and MPL fixture').hexdigest()
