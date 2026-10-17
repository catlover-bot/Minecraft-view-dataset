"""Local archive and fake GitHub-ID save-only lifecycle; no real GitHub calls."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import subprocess
import sys
import pytest
from offline_evaluation.experiments.goal015 import checkpoint as c, runtime as r
from offline_evaluation.experiments.goal015.tests.test_runtime import fixture_plan,execute,MockAdapter


def package(tmp_path):
    plan=fixture_plan(tmp_path);a=MockAdapter();assert execute(plan,a)['calls_settled']==80
    result=c.create_checkpoint(plan,plan.parent/'enabled/ledger.json',reason='FIXTURE_COMPLETE',target_commit='a'*40)
    return plan,a,Path(result['path']),r.read_json(Path(result['path'])/'checkpoint.json')


def test_import_and_frozen_helpers_without_provider_import_or_github_call():
    script='''import sys
from offline_evaluation.experiments.goal015 import checkpoint as c
calls=[]
def fake_run(args,**kwargs):
    calls.append(args)
    from types import SimpleNamespace
    return SimpleNamespace(returncode=0,stdout=b'[]',stderr=b'')
c.subprocess.run=fake_run
assert c.Github().pages('repos/fixture/releases','fixture')==[]
assert len(calls)==1
assert not any(k.endswith('.providers') or k.endswith('.generation') or k.endswith('goal015.adapter') for k in sys.modules)
'''
    result=subprocess.run([sys.executable,'-B','-c',script],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_checkpoint_contains_evaluation_and_status_reports_bound_counts(tmp_path):
    plan,a,d,p=package(tmp_path)
    evaluation=plan.parent/'evaluations/fixture';evaluation.mkdir(parents=True)
    artifact=evaluation/'evaluation_80.json';artifact.write_text('{"fixture":true}')
    summary={'schema':'goal015_paired_pilot_summary_v1','plan_binding':{'plan_id':r.read_json(plan)['plan_id'],
        'plan':r.descriptor(plan),'ledger':r.descriptor(plan.parent/'enabled/ledger.json')},
        'evaluation_artifact':r.descriptor(artifact),'planned_calls':80,'actual_metric_rows':79,'evaluated_protocol_valid_rows':79}
    (evaluation/'summary.json').write_text(json.dumps(summary))
    before={str(x) for x in plan.parent.rglob('*')}
    status=r.status(plan)
    assert status['latest_evaluation_counts']=={'planned_calls':80,'actual_metric_rows':79,'evaluated_protocol_valid_rows':79}
    assert {str(x) for x in plan.parent.rglob('*')}==before
    saved=c.create_save_only(plan)
    restored=Path(saved['path'])/'verified-restoration'
    assert (restored/'evaluations/fixture/evaluation_80.json').read_bytes()==artifact.read_bytes()
    assert (restored/'evaluations/fixture/summary.json').read_bytes()==(evaluation/'summary.json').read_bytes()
    assert a.calls==80


class FakeGithub:
    def __init__(self):self.releases=[];self.assets={};self.blobs={};self.calls=[];self.fail_download=False
    def call(self,stage,args,output=None):
        self.calls.append((stage,args,output))
        if stage=='commit_ancestry':return {'status':'identical','behind_by':0,'base_commit':{'sha':'a'*40}}
        if stage=='release_create':
            payload=json.loads(Path(args[-1]).read_text());release=dict(payload,id=123,published_at=None)
            self.releases.append(release);return copy.deepcopy(release)
        if stage=='release_lookup':return copy.deepcopy(next(r for r in self.releases if r['id']==int(args[-1].split('/')[-1])))
        if stage=='asset_upload':
            raw=Path(args[-1]).read_bytes();row={'id':456,'name':'goal015-state.tar.gz','size':len(raw),'digest':'sha256:'+hashlib.sha256(raw).hexdigest(),'state':'uploaded'}
            self.assets[123]=[row];self.blobs[456]=raw;return copy.deepcopy(row)
        if stage=='asset_download':
            assert '/releases/assets/456' in args[1]
            if self.fail_download:raise RuntimeError('fixture interruption')
            Path(output).write_bytes(self.blobs[456]);return None
        raise AssertionError(stage)
    def pages(self,endpoint,stage):
        self.calls.append((stage,endpoint,None))
        if stage=='release_list':return copy.deepcopy(self.releases)
        if stage=='asset_list':return copy.deepcopy(self.assets.get(int(endpoint.split('/')[-2]),[]))
        raise AssertionError(stage)


def test_local_verified_preserves_raw_and_no_payload_copy(tmp_path):
    plan,a,d,p=package(tmp_path)
    assert a.calls==80 and p['status']=='LOCAL_VERIFIED' and p['github_status']=='NOT_UPLOADED'
    restored=d/'verified-restoration'
    assert (restored/'enabled/responses/c00.raw.json').is_file()
    assert not (restored/'payload.json').exists()
    assert r.read_json(restored/'plan.json')['canonical_root']==str(plan.parent)
    with pytest.raises(r.RuntimeErrorSafe,match='copied'):r.verify_plan(restored/'plan.json')


def test_new_draft_id_then_reuse_no_generation_or_overwrite(tmp_path):
    plan,a,d,p=package(tmp_path);gh=FakeGithub()
    first=c.upload_checkpoint(d,tmp_path/'v1',allow_create=True,github=gh)
    assert first['status']=='GITHUB_VERIFIED' and first['release_id']==123 and first['asset_id']==456
    second=c.upload_checkpoint(d,tmp_path/'v2',github=gh)
    third=c.upload_checkpoint(d,tmp_path/'v3',release_id=123,github=gh)
    assert not second['uploaded'] and not third['uploaded'] and a.calls==80
    assert sum(stage=='release_create' for stage,_,_ in gh.calls)==1
    assert sum(stage=='asset_upload' for stage,_,_ in gh.calls)==1
    assert not any('/tags/' in str(args) for _,args,_ in gh.calls)


def test_download_failure_retry_saves_only_reuses_draft_and_asset(tmp_path):
    plan,a,d,p=package(tmp_path);gh=FakeGithub();gh.fail_download=True
    with pytest.raises(RuntimeError,match='fixture interruption'):c.upload_checkpoint(d,tmp_path/'fail',allow_create=True,github=gh)
    gh.fail_download=False;out=c.upload_checkpoint(d,tmp_path/'retry',github=gh)
    assert out['status']=='GITHUB_VERIFIED' and not out['uploaded'] and a.calls==80
    assert sum(stage=='release_create' for stage,_,_ in gh.calls)==1


def test_missing_draft_requires_explicit_create(tmp_path):
    _,_,d,_=package(tmp_path);gh=FakeGithub()
    with pytest.raises(r.RuntimeErrorSafe,match='explicit creation'):c.upload_checkpoint(d,tmp_path/'v',github=gh)
    assert not gh.releases


def test_old_draft_numeric_id_rejected_without_mutation(tmp_path):
    _,_,d,_=package(tmp_path);gh=FakeGithub();gh.releases=[{'id':123,'draft':True,'published_at':None,'tag_name':'old-goal006', 'target_commitish':'a'*40,'body':'existing'}]
    with pytest.raises(r.RuntimeErrorSafe,match='mismatch'):c.upload_checkpoint(d,tmp_path/'v',release_id=123,allow_create=True,github=gh)
    assert not any(stage in ('release_create','asset_upload') for stage,_,_ in gh.calls)


@pytest.mark.parametrize('change',[{'published_at':'2020-01-01'}, {'draft':False}, {'body':'other plan'}])
def test_remote_identity_mismatch_never_overwrites(tmp_path,change):
    _,_,d,_=package(tmp_path);gh=FakeGithub();c.upload_checkpoint(d,tmp_path/'first',allow_create=True,github=gh)
    gh.releases[0].update(change)
    with pytest.raises(r.RuntimeErrorSafe,match='mismatch'):c.upload_checkpoint(d,tmp_path/'second',release_id=123,allow_create=True,github=gh)
    assert sum(stage=='asset_upload' for stage,_,_ in gh.calls)==1


def test_corrupt_asset_refused_before_remote(tmp_path):
    _,_,d,_=package(tmp_path);(d/'goal015-state.tar.gz').write_bytes(b'bad');gh=FakeGithub()
    with pytest.raises(r.RuntimeErrorSafe):c.upload_checkpoint(d,tmp_path/'v',allow_create=True,github=gh)
    assert not gh.calls


@pytest.mark.parametrize('name,link',[('../escape',False),('/absolute',False),('link',True)])
def test_archive_traversal_and_links_rejected(tmp_path,name,link):
    path=tmp_path/'bad.tar.gz'
    with tarfile.open(path,'w:gz') as a:
        info=tarfile.TarInfo(name)
        if link:info.type=tarfile.SYMTYPE;info.linkname='/etc/passwd';a.addfile(info)
        else:info.size=1;a.addfile(info,io.BytesIO(b'x'))
        raw=b'{"files":{}}';info=tarfile.TarInfo('FILES.json');info.size=len(raw);a.addfile(info,io.BytesIO(raw))
    with pytest.raises((r.RuntimeErrorSafe,ValueError)):c.verify_archive(path,r.descriptor(path)['sha256'],destination=tmp_path/'restore')


def test_save_only_checkpoint_never_calls_adapter(tmp_path):
    plan,a,d,p=package(tmp_path);out=c.create_save_only(plan)
    assert out['status']=='LOCAL_VERIFIED' and a.calls==80


def test_mock80_real_local_checkpoints_each_building(tmp_path):
    plan=fixture_plan(tmp_path);a=MockAdapter()
    out=r.execute(plan,adapter=a,confirmation=lambda p:True)
    assert out['status']=='COMPLETE' and a.calls==80
    packages=list((plan.parent/'checkpoints').glob('*/checkpoint.json'))
    reasons=[r.read_json(path)['reason'] for path in packages]
    assert reasons.count('BUILDING_COMPLETE')==10 and reasons.count('FINAL')==1
    assert all(r.read_json(path)['status']=='LOCAL_VERIFIED' for path in packages)
    receipt={'status':'PASS','fixture_only':True,'mock_transport_calls':80,'real_research_api_requests':0,
        'planned_calls':80,'settled_calls':80,'building_checkpoints':10,'final_checkpoints':1,'generation_retries':0}
    r._save(tmp_path/'MOCK80_INTEGRATION_RECEIPT.json',receipt,exclusive=True)
