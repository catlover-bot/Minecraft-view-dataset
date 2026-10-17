from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest

from offline_evaluation.experiments.goal015 import pricing,runtime
from offline_evaluation.experiments.goal015.tests.test_runtime import fixture_plan,save_plan


def pricing_fixture(tmp_path,monkeypatch):
    root=tmp_path/'pricing';root.mkdir();old=root/'old.json'
    doc=root/'doc.html';doc.write_text('fixture original docs')
    contract={'rates':'fixture'}
    record=dict(runtime.descriptor(doc),url='https://developers.openai.com/api/docs/models/gpt-5-mini')
    old.write_text(json.dumps({'documents':[record]}))
    plan={'canonical_root':str(root),'plan_id':'fixture-id','pricing_contract':contract,'pricing_evidence':runtime.descriptor(old)}
    monkeypatch.setattr(pricing,'verify_plan',lambda _:plan)
    return root,plan


def test_capture_public_get_is_unreviewed_then_seal_same_contract(tmp_path,monkeypatch):
    root,plan=pricing_fixture(tmp_path,monkeypatch);calls=[]
    class Response:
        url='https://developers.openai.com/api/docs/models/gpt-5-mini';status=200
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,size):return b'fixture fresh official-document bytes'
    def opened(req,timeout):
        assert req.get_method()=='GET' and not req.has_header('Authorization');calls.append(req.full_url);return Response()
    monkeypatch.setattr(pricing.urllib.request,'build_opener',lambda *_:SimpleNamespace(open=opened))
    out=pricing.capture('fixture',root/'fresh');assert len(calls)==1
    assert out['review_status']=='REQUIRES_HUMAN_CONTRACT_REVIEW'
    reviewed=root/'reviewed.json';reviewed.write_text(json.dumps(plan['pricing_contract']))
    result=pricing.seal('fixture',root/'fresh/capture.json',reviewed,'Reviewed complete fixture contract, rates, snapshots and support.',root/'sealed.json')
    sealed=runtime.read_json(result['path']);assert sealed['pricing_contract']==plan['pricing_contract']
    assert sealed['research_api_requests_sent']==0 and sealed['review_status']=='REVIEWED_OFFICIAL_DOCUMENTS'
    reviewed.write_text('{"rates":"changed"}')
    with pytest.raises(runtime.RuntimeErrorSafe,match='changed price'):
        pricing.seal('fixture',root/'fresh/capture.json',reviewed,'This must not silently change the original scope.',root/'bad.json')
    assert not (root/'bad.json').exists()


@pytest.mark.parametrize('kind',['traversal','symlink'])
def test_capture_path_rejected_before_any_write_or_network(tmp_path,monkeypatch,kind):
    root,plan=pricing_fixture(tmp_path,monkeypatch)
    if kind=='traversal':target=root/'..'/'escape'
    else:
        other=tmp_path/'other';other.mkdir();(root/'link').symlink_to(other,target_is_directory=True);target=root/'link'/'escape'
    monkeypatch.setattr(pricing.urllib.request,'build_opener',lambda *_: (_ for _ in ()).throw(AssertionError('network setup before guard')))
    with pytest.raises(runtime.RuntimeErrorSafe):pricing.capture('fixture',target)
    assert not target.exists()


@pytest.mark.parametrize('url',['http://developers.openai.com/a','https://api.openai.com/v1/models','https://api.anthropic.com/v1/messages','https://example.com/docs','https://user:password@developers.openai.com/docs'])
def test_document_get_allowlist_excludes_api_and_credentials(url):
    with pytest.raises(runtime.RuntimeErrorSafe):pricing.official(url)


def test_actual_cli_readonly_and_evaluate_no_provider_import_or_network(tmp_path):
    p=fixture_plan(tmp_path);plan=runtime.read_json(p)
    plan['gt_sources']={f's{i}':{'building_directory':'/NONEXISTENT_GT_NOT_READ','voxels_sha256':'a','bbox_sha256':'b','grid_sha256':'c'} for i in range(10)}
    save_plan(p,plan)
    program='''import builtins,sys,socket
original=builtins.__import__
def guarded(name,globals=None,locals=None,fromlist=(),level=0):
 if name.endswith(('.providers','.generation','.adapter')) or any(x in ('providers','generation','adapter') for x in (fromlist or ())):
  raise AssertionError('research API module imported: '+name)
 return original(name,globals,locals,fromlist,level)
builtins.__import__=guarded
socket.socket=lambda *a,**k: (_ for _ in ()).throw(AssertionError('network socket'))
from offline_evaluation.experiments.goal015.cli import main
main(sys.argv[1:])
'''
    for command in ('preview','status','evaluate'):
        run=subprocess.run([sys.executable,'-B','-c',program,command,'--plan',str(p)],capture_output=True,text=True)
        assert run.returncode==0,run.stderr
        data=json.loads(run.stdout);assert data['research_api_requests_sent']==0
    evaluation=next(p.parent.glob('evaluations/*/evaluation_80.json'))
    run=subprocess.run([sys.executable,'-B','-c',program,'aggregate','--plan',str(p),'--evaluation',str(evaluation)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    for name in ('enabled','authority','execution.lock'):assert not (p.parent/name).exists()
