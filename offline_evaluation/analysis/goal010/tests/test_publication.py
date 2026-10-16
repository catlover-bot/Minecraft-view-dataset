"""Synthetic completion-copy tests; no actual Goal010 products are copied."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from offline_evaluation.analysis.goal010 import pipeline as p
from offline_evaluation.analysis.goal010.publication import publish as pub
from offline_evaluation.analysis.goal010.tests.test_pipeline import case, CODE, SPEC

COPIER = 'd'*40


@pytest.fixture
def publication_case(case,monkeypatch,tmp_path):
    root,state,binding,execute=case
    repo=tmp_path/'repository';allowed=repo/'reports/server_only/goal010';allowed.mkdir(parents=True)
    actual_root=pub.guard.ROOT
    code_hashes={name:pub.guard.sha256_file(actual_root/name) for name in pub.OWN_FILES}
    monkeypatch.setattr(pub.guard,'ROOT',repo)
    def committed(commit,files):
        assert commit==COPIER and tuple(files)==pub.OWN_FILES
        return deepcopy(code_hashes)
    monkeypatch.setattr(pub.guard,'verify_commit',committed)
    def build(extra=None):
        def phase(number,package,outroot,gate,bound):
            result=execute(number,package,outroot,gate,bound);out=outroot/p.DIRECTORIES[number]
            if number==0:
                path=out/'source_inventory_receipt.json';value=json.loads(path.read_text());value['status']='PASS'
                path.write_text(json.dumps(value))
            if number==1:
                raw=out/'predictions';raw.mkdir();actions=raw/'synthetic.actions.json';actions.write_text('{"fixture":true}')
                coords=raw/'synthetic.xyz.npy';coords.write_bytes(b'synthetic-array-no-real-prediction')
                artifacts={name:{'path':path.name,'sha256':pub.guard.sha256_file(path),'bytes':path.stat().st_size}
                           for name,path in [('actions',actions),('coordinates',coords)]}
                p.write_json(out/'baseline_predictions_manifest.json',{'predictions':[{'artifacts':artifacts}]})
            if number==2:(out/'synthetic.csv').write_text('fixture,count\nonly,1\n')
            if number==3:(out/'synthetic.md').write_text('# Synthetic fixture\n')
            if number==4:(out/'synthetic.png').write_bytes(b'non-image fixture; never copied')
            if extra:extra(number,out)
            return result
        monkeypatch.setattr(p,'execute_phase',phase)
        for number in range(6):p.run_phase(number,root,CODE,SPEC)
        return root,allowed/'analysis',state,binding
    return build


def run(root,output):
    return pub.publish(root,output,code_commit=COPIER)


def test_completed_copy_is_exact_and_excludes_predictions_start_markers_and_png(publication_case):
    root,output,state,binding=publication_case();before={str(x):x.read_bytes() for x in root.rglob('*') if x.is_file()}
    result=run(root,output);receipt=json.loads((output/'PUBLIC_COPY_RECEIPT.json').read_text())
    assert result['status']=='PUBLIC_COPY_VERIFIED' and receipt['analysis_binding']==binding
    expected={f'PHASE{i}_RECEIPT.json' for i in range(6)}
    expected.update(x.relative_to(root).as_posix() for directory in p.DIRECTORIES for x in (root/directory).iterdir() if x.is_file() and x.suffix in pub.EXTENSIONS)
    assert set(receipt['explicit_allowlist'])==expected==set(receipt['files'])
    assert {x.relative_to(output).as_posix() for x in output.rglob('*') if x.is_file()}==expected|{'PUBLIC_COPY_RECEIPT.json'}
    for name,descriptor in receipt['files'].items():
        assert (output/name).read_bytes()==(root/name).read_bytes()
        assert descriptor['source_sha256']==descriptor['copy_sha256']==pub.guard.sha256_file(root/name)
    assert not any('predictions/' in name or name.endswith(('.npy','.png','.actions.json')) for name in expected)
    assert before=={str(x):x.read_bytes() for x in root.rglob('*') if x.is_file()}
    assert receipt['new_inference_runs']==receipt['github_mutations']==receipt['research_api_requests_sent']==0
    assert receipt['initial_and_before_after_source_packages_identical'] is True


@pytest.mark.parametrize('mutation',['missing_receipt','bad_product','not_unchanged','nonzero_query','stopped_phase'])
def test_incomplete_or_changed_phases_fail_before_destination(publication_case,mutation):
    root,output,state,_=publication_case()
    if mutation=='missing_receipt':(root/'PHASE5_RECEIPT.json').unlink()
    elif mutation=='bad_product':(root/p.DIRECTORIES[2]/'synthetic.json').write_text('{}')
    elif mutation=='stopped_phase':(root/'PHASE3_STOP.json').write_text('{}')
    else:
        path=root/'PHASE5_RECEIPT.json';value=json.loads(path.read_text())
        value['source_before_after_unchanged' if mutation=='not_unchanged' else 'research_api_requests_sent']=False if mutation=='not_unchanged' else 1
        path.write_text(json.dumps(value))
    with pytest.raises(Exception):run(root,output)
    assert not output.exists()


def test_initial_source_package_mismatch_blocks_destination(publication_case):
    root,output,state,_=publication_case();state['value']='changed after Phase0'
    with pytest.raises(ValueError,match='initial Phase0'):run(root,output)
    assert not output.exists()


def test_secret_scanner_and_size_limit_run_before_copy(publication_case,monkeypatch):
    def extra(number,out):
        if number==2:(out/'synthetic_sensitive.json').write_bytes(b'{"synthetic":"'+b'sk-'+b'A'*35+b'"}')
    root,output,*_=publication_case(extra)
    with pytest.raises(ValueError,match='suppressed'):run(root,output)
    assert not output.exists()


def test_individual_size_limit_blocks_before_destination(publication_case,monkeypatch):
    root,output,*_=publication_case();monkeypatch.setattr(pub,'MAX_FILE_BYTES',1)
    with pytest.raises(ValueError,match='10MiB'):run(root,output)
    assert not output.exists()


@pytest.mark.parametrize('kind',['existing','outside','parent_traversal','symlink'])
def test_destination_is_fresh_nonsymlink_and_report_scoped(publication_case,tmp_path,kind):
    root,output,state,_=publication_case();before=state['loaded']
    if kind=='existing':output.mkdir();(output/'keep').write_text('preserve')
    elif kind=='outside':output=tmp_path/'outside'
    elif kind=='parent_traversal':output=output.parent/'x/../analysis'
    else:output.symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises((ValueError,FileExistsError)):run(root,output)
    assert state['loaded']==before
    if kind=='existing':assert (output/'keep').read_text()=='preserve'


def test_uncommitted_copier_fails_before_destination(publication_case,monkeypatch):
    root,output,state,_=publication_case();before=state['loaded']
    def denied(*a,**k):raise ValueError('commit does not bind current file')
    monkeypatch.setattr(pub.guard,'verify_commit',denied)
    with pytest.raises(ValueError,match='commit'):run(root,output)
    assert not output.exists() and state['loaded']==before


@pytest.mark.parametrize('change',['original_package','canonical_product','extra_copy_file'])
def test_mutation_during_copy_never_gets_completion_receipt(publication_case,monkeypatch,change):
    root,output,state,_=publication_case();copy=pub._copy;changed=False
    def altering(source,target,descriptor):
        nonlocal changed
        copy(source,target,descriptor)
        if not changed:
            changed=True
            if change=='original_package':state['value']='changed during copy'
            elif change=='canonical_product':(root/p.DIRECTORIES[2]/'synthetic.json').write_text('{}')
            else:(output/'unexpected.txt').write_text('not allowed')
    monkeypatch.setattr(pub,'_copy',altering)
    with pytest.raises(ValueError):run(root,output)
    assert output.is_dir() and not (output/'PUBLIC_COPY_RECEIPT.json').exists()


def test_exit_audit_failure_preserves_partial_copy_without_pass(publication_case,monkeypatch):
    root,output,state,_=publication_case()
    @contextmanager
    def locks():
        state['active']=True
        with p.guard.audit8.old_audit.no_python_network():yield ['fixture-lock']
        state['active']=False
        raise ValueError('fixture exit audit failed')
    monkeypatch.setattr(pub.guard,'locked_sources',locks)
    with pytest.raises(ValueError,match='exit audit'):run(root,output)
    assert output.is_dir() and not (output/'PUBLIC_COPY_RECEIPT.json').exists()


def test_new_unbound_top_level_file_not_silently_published(publication_case):
    root,output,*_=publication_case();(root/p.DIRECTORIES[5]/'unbound.json').write_text('{}')
    with pytest.raises(ValueError,match='not bound'):run(root,output)
    assert not output.exists()


def test_cli_diagnostic_suppresses_exception_payload(monkeypatch,capsys):
    def denied(*a,**k):raise ValueError('SYNTHETIC_PRIVATE_PAYLOAD_NOT_FOR_STDOUT')
    monkeypatch.setattr(pub,'publish',denied)
    result=pub.main(['--run-root','fixture','--destination','fixture','--code-commit',COPIER])
    output=capsys.readouterr().out
    assert result==5 and 'SYNTHETIC_PRIVATE' not in output and json.loads(output)['status']=='STOP'
