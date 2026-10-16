"""Synthetic source guards only; no actual research analysis or network I/O."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from offline_evaluation.analysis.goal010 import source_guard as guard


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,ensure_ascii=False,sort_keys=True).encode()
    path.write_bytes(data)
    return {'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}


def descriptor(path):
    return {'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}


@pytest.mark.parametrize('name',['../escape.json','/etc/passwd','a/../../escape'])
def test_safe_child_rejects_absolute_and_parent_paths(tmp_path,name):
    with pytest.raises(ValueError):guard.safe_child(tmp_path,name)


def test_safe_child_rejects_file_and_directory_symlinks(tmp_path):
    real=tmp_path/'real';real.mkdir();put(real/'x.json',{})
    (tmp_path/'link.json').symlink_to(real/'x.json')
    (tmp_path/'linked').symlink_to(real,target_is_directory=True)
    for name in ('link.json','linked/x.json'):
        with pytest.raises(guard.audit8.AuditError):guard.safe_child(tmp_path,name)
    assert guard.safe_child(tmp_path,'real/x.json')==real/'x.json'


@pytest.mark.parametrize('mutation',['hash','size'])
def test_products_reject_hash_or_size_changes(tmp_path,mutation):
    bound=put(tmp_path/'product.json',{'fixture':True})
    guard.verify_products(tmp_path,{'product.json':bound})
    changed=dict(bound);changed['sha256' if mutation=='hash' else 'bytes']='0'*64 if mutation=='hash' else bound['bytes']+1
    with pytest.raises(ValueError,match='hash mismatch'):guard.verify_products(tmp_path,{'product.json':changed})


def goal008_fixture(tmp_path,monkeypatch):
    root=tmp_path/'repository';public=root/'reports';canonical=tmp_path/'canonical'
    root.mkdir();public.mkdir();canonical.mkdir()
    code=root/'new.py';code.write_text('# synthetic\n');old=root/'old.py';old.write_text('# frozen synthetic\n')
    refs={}
    for index in range(59):
        path=canonical/f'{index:02d}.json';desc=put(path,{'fixture_index':index})
        refs[str(index)]={'canonical_source':str(path),**desc}
    names=('phase01_public_copy.json','examples_publication_public_copy.json')
    products={names[0]:put(public/names[0],{'files':dict(list(refs.items())[:30])}),
              names[1]:put(public/names[1],{'files':dict(list(refs.items())[30:])})}
    products['aggregate.json']=put(public/'aggregate.json',{'fixture':True})
    initial_root=tmp_path/'initial_examples';initial_root.mkdir()
    initial={'reproduction':{'argv':['python','--destination',str(initial_root)]},
             'products':{'fixture_figure.json':put(initial_root/'fixture_figure.json',{'synthetic':True})}}
    put(initial_root/'EXAMPLES_MANIFEST.json',initial)
    products['initial_examples_manifest.json']=put(public/'initial_examples_manifest.json',initial)
    manifest={'products':products,'code_hashes':{'new.py':descriptor(code)['sha256']},
              'frozen_old_code_hashes':{'old.py':descriptor(old)['sha256']}}
    digest=put(public/'MANIFEST.json',manifest)['sha256']
    monkeypatch.setattr(guard,'ROOT',root);monkeypatch.setattr(guard,'GOAL008',public)
    monkeypatch.setattr(guard,'GOAL008_MANIFEST_SHA256',digest)
    calls=[]
    monkeypatch.setattr(guard.audit8,'verify_report',lambda path:calls.append(path) or {'status':'PASS'})
    return public,canonical,code,old,calls


def test_frozen_goal008_products_and_59_canonical_refs_fixture_pass(tmp_path,monkeypatch):
    public,canonical,code,old,calls=goal008_fixture(tmp_path,monkeypatch)
    manifest,gate=guard.verify_goal008()
    assert gate['status']=='PASS' and calls==[public/'audit/source_gate.json']


@pytest.mark.parametrize('target',['manifest','product','canonical','new_code','old_code','initial_manifest','initial_product'])
def test_frozen_goal008_hash_failure_precedes_source_gate(tmp_path,monkeypatch,target):
    public,canonical,code,old,calls=goal008_fixture(tmp_path,monkeypatch)
    path={'manifest':public/'MANIFEST.json','product':public/'aggregate.json','canonical':canonical/'58.json',
          'new_code':code,'old_code':old,'initial_manifest':tmp_path/'initial_examples/EXAMPLES_MANIFEST.json',
          'initial_product':tmp_path/'initial_examples/fixture_figure.json'}[target]
    with path.open('ab') as stream:stream.write(b' ')
    with pytest.raises(ValueError):guard.verify_goal008()
    assert calls==[]


@pytest.mark.parametrize('mutation',['short_commit','wrong_blob','unsafe_name'])
def test_commit_gate_rejects_unbound_bytes_and_paths(tmp_path,monkeypatch,mutation):
    path=tmp_path/'code.py';path.write_text('fixture=1\n');monkeypatch.setattr(guard,'ROOT',tmp_path)
    calls=[]
    def run(command,**kwargs):
        calls.append((command,kwargs))
        return SimpleNamespace(returncode=0,stdout=b'wrong' if mutation=='wrong_blob' else path.read_bytes())
    monkeypatch.setattr(guard.subprocess,'run',run)
    with pytest.raises(ValueError):
        guard.verify_commit('abc' if mutation=='short_commit' else 'a'*40,['../code.py'] if mutation=='unsafe_name' else ['code.py'])
    assert len(calls)==(1 if mutation=='wrong_blob' else 0)


def test_commit_gate_pass_uses_read_only_git_show(tmp_path,monkeypatch):
    path=tmp_path/'code.py';path.write_text('fixture=1\n');monkeypatch.setattr(guard,'ROOT',tmp_path);calls=[]
    def run(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0,stdout=path.read_bytes())
    monkeypatch.setattr(guard.subprocess,'run',run)
    result=guard.verify_commit('a'*40,['code.py'])
    assert result=={'code.py':descriptor(path)['sha256']}
    assert calls==[['git','show','a'*40+':code.py']]


@pytest.mark.parametrize('field,value',[('status','PENDING'),('input_kind','unknown'),('phase0_binding','PENDING'),
                                      ('definitions_frozen',False),('specification_sha256','b'*64)])
def test_source_gate_requires_phase0_and_exact_frozen_specification(field,value):
    gate={'status':'PASS','input_kind':'fixture','phase0_binding':'PASS','definitions_frozen':True,'specification_sha256':'a'*64}
    guard.require_gate(gate,'a'*64)
    gate[field]=value
    with pytest.raises(ValueError):guard.require_gate(gate,'a'*64)


def test_locked_sources_uses_existing_locks_and_restores_python_network_hooks(monkeypatch,tmp_path):
    plan=tmp_path/'plan.json';put(plan,{'identity':'fixture'})
    monkeypatch.setattr(guard,'read_json',lambda path: {'sources':{'fixture':{'plan_path':str(plan)}}} if Path(path).name=='MANIFEST.json' else {'identity':'fixture'})
    events=[]
    @contextmanager
    def locks(plans):
        assert plans=={'fixture':{'identity':'fixture'}};events.append('enter')
        try:yield ['existing-readonly-lock']
        finally:events.append('exit')
    monkeypatch.setattr(guard.audit8.old_audit,'source_locks',locks)
    monkeypatch.setattr(guard.audit8.old_audit,'process_audit',lambda:events.append('process'))
    before=(socket.socket.connect,socket.socket.connect_ex)
    with guard.locked_sources() as names:
        assert names==['existing-readonly-lock']
        # Exercise the installed Python methods without requesting a kernel
        # socket; the workspace sandbox itself may deny socket construction.
        with pytest.raises(guard.audit8.AuditError,match='python_network_disabled'):socket.socket.connect(None,('127.0.0.1',1))
        with pytest.raises(guard.audit8.AuditError,match='python_network_disabled'):socket.socket.connect_ex(None,('127.0.0.1',1))
    assert (socket.socket.connect,socket.socket.connect_ex)==before
    assert events==['enter','process','process','exit']


def metadata_fixture(tmp_path):
    source_path=tmp_path/'description.json';bbox_path=tmp_path/'bbox.json';voxels_path=tmp_path/'voxels.npy'
    text={'summary':'日本語の説明','materials':[{'name':'stone'},'   '],'dimensions_estimate':{'width':2,'height':3,'depth':4},
          'elements':['roof','  '],'rebuild_hints':['place blocks'],'uncertainties':[],'metadata':{'not_counted':'x'*20}}
    source_digest=put(source_path,text)['sha256'];bbox_digest=put(bbox_path,{'xmin':-2,'xmax':-1,'ymin':3,'ymax':5,'zmin':-4,'zmax':-1})['sha256']
    voxels_path.write_bytes(b'fixture-no-numpy-read');voxel_digest=descriptor(voxels_path)['sha256']
    identity={'scene_id':'fixture','description_provider':'openai','input_id':'fixture-input','input_sha256':source_digest,
              'source_sha256':source_digest,'gt_bbox_sha256':bbox_digest,'gt_voxels_sha256':voxel_digest}
    source={**identity,'source_path':str(source_path),'gt_bbox_path':str(bbox_path),'gt_voxels_path':str(voxels_path),
            'gt_descriptor':{'bbox_sha256':bbox_digest,'voxels_sha256':voxel_digest}}
    score={**identity,'gt_material_target':{'occupied_voxels':12}}
    return {'cohort':{'scenes':[{'scene_id':'fixture','size_rank_stratum':2}]},'proxy_scores':[score],'descriptions':[source]}


def test_metadata_negative_world_coordinates_inclusive_volume_and_human_chars(tmp_path):
    package=metadata_fixture(tmp_path);before=deepcopy(package);row=guard.metadata_rows(package)[0]
    assert package==before and row['gt_storage_bbox_volume']==24 and row['gt_occupancy_density']==.5
    assert row['description_chars']==len(Path(package['descriptions'][0]['source_path']).read_text())
    assert row['human_text_chars']==len('日本語の説明')+len('stone')+len('roof')+len('place blocks')


def test_metadata_length_preserves_exact_decoded_crlf_bytes(tmp_path):
    package=metadata_fixture(tmp_path);source=package['descriptions'][0];score=package['proxy_scores'][0]
    path=Path(source['source_path']);text=json.dumps(json.loads(path.read_text()),ensure_ascii=False,indent=2).replace('\n','\r\n')
    path.write_bytes(text.encode('utf-8'));digest=descriptor(path)['sha256']
    for row in (source,score):row['source_sha256']=row['input_sha256']=digest
    assert guard.metadata_rows(package)[0]['description_chars']==len(text)


@pytest.mark.parametrize('mutation',['missing_axis','zero_extent','negative_extents','bool_axis','zero_occupied','negative_occupied','excess_occupied','missing_size_stratum'])
def test_metadata_invalid_sizes_fail_closed(tmp_path,mutation):
    package=metadata_fixture(tmp_path);source=package['descriptions'][0];score=package['proxy_scores'][0]
    bbox=json.loads(Path(source['gt_bbox_path']).read_text())
    if mutation=='missing_axis':bbox.pop('xmin')
    elif mutation=='zero_extent':bbox['xmax']=bbox['xmin']-1
    elif mutation=='negative_extents':bbox.update(xmax=bbox['xmin']-3,ymax=bbox['ymin']-3)
    elif mutation=='bool_axis':bbox['xmin']=True
    elif mutation=='zero_occupied':score['gt_material_target']['occupied_voxels']=0
    elif mutation=='negative_occupied':score['gt_material_target']['occupied_voxels']=-1
    elif mutation=='excess_occupied':score['gt_material_target']['occupied_voxels']=25
    else:package['cohort']['scenes'][0].pop('size_rank_stratum')
    h=put(Path(source['gt_bbox_path']),bbox)['sha256'];source['gt_descriptor']['bbox_sha256']=h;source['gt_bbox_sha256']=h;score['gt_bbox_sha256']=h
    with pytest.raises((ValueError,KeyError)):guard.metadata_rows(package)


@pytest.mark.parametrize('mutation',['changed_bbox','score_source_mismatch','score_input_mismatch','score_gt_mismatch'])
def test_metadata_reread_source_and_gt_hashes_remain_bound(tmp_path,mutation):
    package=metadata_fixture(tmp_path)
    if mutation=='changed_bbox':
        with Path(package['descriptions'][0]['gt_bbox_path']).open('a') as stream:stream.write(' ')
    else:
        key={'score_source_mismatch':'source_sha256','score_input_mismatch':'input_sha256','score_gt_mismatch':'gt_bbox_sha256'}[mutation]
        package['proxy_scores'][0][key]='0'*64
    with pytest.raises(ValueError):guard.metadata_rows(package)
