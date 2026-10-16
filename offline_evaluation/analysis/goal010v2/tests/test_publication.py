import gzip
import json
from pathlib import Path
import pytest
from offline_evaluation.analysis.goal010v2.publication import publish as p


def test_small_identity_and_existing_destination(tmp_path):
    source=tmp_path/'source.json'; source.write_text('{"scope":"synthetic"}\n')
    target=tmp_path/'new/sub/copy.json'
    r=p.copy_one(source,target)
    assert r['encoding']=='identity' and source.read_bytes()==target.read_bytes()
    assert r['source_sha256']==r['public_sha256'] and r['decoded_bytes_exactly_match_source']
    with pytest.raises(FileExistsError): p.copy_one(source,target)


def test_large_json_is_lossless_deterministic_gzip(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'MAX_BYTES',200)
    source=tmp_path/'source.json'; source.write_text(json.dumps({'values':[7]*1000}))
    r=p.copy_one(source,tmp_path/'copy.json')
    assert r['encoding']=='gzip' and r['public_bytes']<=200
    assert gzip.decompress(Path(r['public_path']).read_bytes())==source.read_bytes()
    other=p.copy_one(source,tmp_path/'other.json')
    assert other['public_sha256']==r['public_sha256']


def test_large_non_json_kept_canonical(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'MAX_BYTES',10)
    source=tmp_path/'source.csv'; source.write_text('column\n'+'value\n'*10)
    with pytest.raises(ValueError): p.copy_one(source,tmp_path/'copy.csv')
    assert not (tmp_path/'copy.csv').exists()


def test_uncompressible_oversize_not_written(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'MAX_BYTES',10)
    source=tmp_path/'source.json'; source.write_text('{"long":1234567890}')
    with pytest.raises(ValueError): p.copy_one(source,tmp_path/'copy.json')
    assert not (tmp_path/'copy.json.gz').exists()


@pytest.mark.parametrize('location',['source','destination','parent'])
def test_symlinks_rejected(tmp_path,location):
    source=tmp_path/'source.json'; source.write_text('{}')
    target=tmp_path/'copy.json'
    if location=='source':
        link=tmp_path/'link.json'; link.symlink_to(source); source=link
    elif location=='destination': target.symlink_to(source)
    else:
        real=tmp_path/'real'; real.mkdir(); link=tmp_path/'link'; link.symlink_to(real,target_is_directory=True); target=link/'copy.json'
    with pytest.raises((ValueError,p.guard.audit8.AuditError)): p.copy_one(source,target)


def test_allowlist_excludes_large_candidate_pairs_and_raw_predictions():
    assert '3_full/new_metric_values_960.json' in p.ALLOWLIST
    assert '3_full/new_metric_values_960.csv' in p.ALLOWLIST
    assert '4_summary/metric_pairs.json' not in p.ALLOWLIST
    assert not any(name.endswith(('.npy','.actions.json')) or 'raw.json' in name for name in p.ALLOWLIST)
    assert len(p.ALLOWLIST)==len(set(p.ALLOWLIST))
