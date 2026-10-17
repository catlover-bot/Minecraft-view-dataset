"""Goal015 local verified checkpoints and a separate Draft-only upload path.

Reuses the corrected Goal006C gh transport, paginated numeric-ID lookup pattern,
and frozen path/secret scanners. No import or call of any research API adapter.
"""
from __future__ import annotations
from datetime import datetime, timezone
import ast
from functools import lru_cache
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
from urllib.parse import quote
import uuid

from ...server_v2.checkpoint import safe_relative, scan_file
from .runtime import descriptor, path_safe, read_json, require, _save, verify_plan

REPOSITORY='catlover-bot/Minecraft-view-dataset'
BRANCH='research/server-only-20261013'
MAX_BYTES=1024**3


def verify_archive(path, expected_sha256, *, destination=None):
    """Reject links/traversal/duplicates; verify bytes before optional safe restore."""
    require(descriptor(path)['sha256']==expected_sha256,'archive SHA differs')
    target=path_safe(destination) if destination is not None else None
    if target is not None:
        require(not target.exists(),'new verification destination required');target.mkdir(parents=True)
    with tarfile.open(path,'r:gz') as archive:
        members=archive.getmembers();names=[m.name for m in members]
        require(len(set(names))==len(names) and 'FILES.json' in names,'archive duplicate/missing manifest')
        total=0
        for member in members:
            safe_relative(member.name);require(member.isfile(),'archive regular files only')
            total+=member.size;require(total<=MAX_BYTES,'archive expanded limit')
        manifest=json.loads(archive.extractfile('FILES.json').read())
        require(manifest.get('schema')=='goal015_checkpoint_files_v1','Goal015 archive manifest required')
        require(set(names)==set(manifest['files'])|{'FILES.json'},'archive manifest membership differs')
        required={'plan.json','enabled/approval.json','enabled/authority.json','enabled/ledger.json'}
        require(required<=set(names),'archive execution controls missing')
        plan=json.loads(archive.extractfile('plan.json').read())
        ledger=json.loads(archive.extractfile('enabled/ledger.json').read())
        from .runtime import identity
        require(plan['plan_id']==identity(plan)==manifest['plan_id']==ledger['plan_id'],'archive plan/ledger identity differs')
        require(len(plan['schedule'])==len(ledger['calls'])==80,'archive schedule scope differs')
        for original,row in zip(plan['schedule'],ledger['calls']):
            require(all(row.get(k)==v for k,v in original.items()),'archive ledger schedule differs')
            for key in ('raw','response_text_path'):
                if row.get(key) is not None:
                    item=row[key];relative=Path(item['path']).relative_to(plan['canonical_root']).as_posix()
                    require(relative in manifest['files'] and manifest['files'][relative]=={k:item[k] for k in ('bytes','sha256')},
                            'archive selected response evidence missing')
        for member in members:
            raw=archive.extractfile(member).read()
            if member.name!='FILES.json':
                require(manifest['files'][member.name]=={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},'archive member hash differs')
            if target is not None:
                out=target/member.name;out.parent.mkdir(parents=True,exist_ok=True)
                with out.open('xb') as f:f.write(raw)
                scan_file(out)
    return {'status':'VERIFIED','archive_sha256':expected_sha256,'files':len(manifest['files']),
            'plan_id':manifest['plan_id'],'geometry_or_generation_reexecuted':False}


def create_checkpoint(plan_path, ledger_path, *, reason, target_commit=None):
    """Called under the canonical execution lock, or via create_save_only()."""
    plan=verify_plan(plan_path);root=Path(plan['canonical_root']);ledger_path=path_safe(ledger_path)
    require(ledger_path==root/'enabled/ledger.json','canonical ledger required')
    ledger=read_json(ledger_path);require(ledger['plan_id']==plan['plan_id'],'ledger plan mismatch')
    from .runtime import _validate_ledger
    _validate_ledger(plan,ledger,read_json(root/'enabled/approval.json'))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    name='goal015-'+plan['plan_id'][:12]+'-'+stamp+'-'+uuid.uuid4().hex[:6]
    directory=root/'checkpoints'/name;directory.mkdir(parents=True,exist_ok=False)
    entries={'plan.json':path_safe(plan_path)}
    for source in (root/'enabled',root/'evaluations'):
        if not source.exists():continue
        path_safe(source)
        for path in sorted(source.rglob('*')):
            require(not path.is_symlink(),'nonregular state in checkpoint')
            if path.is_dir():continue
            require(path.is_file(),'nonregular state in checkpoint')
            if path.name.startswith('.pending-'):continue
            entries[path.relative_to(root).as_posix()]=path
    files={}
    for relative,path in entries.items():
        safe_relative(relative);scan_file(path);desc=descriptor(path);files[relative]={k:desc[k] for k in ('bytes','sha256')}
    manifest={'schema':'goal015_checkpoint_files_v1','plan_id':plan['plan_id'],'files':files}
    archive_path=directory/'goal015-state.tar.gz'
    with archive_path.open('xb') as stream,gzip.GzipFile(fileobj=stream,mode='wb',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w') as archive:
        for relative,path in entries.items():
            raw=path.read_bytes();require(files[relative]=={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},'source changed while checkpointing')
            item=tarfile.TarInfo(relative);item.size=len(raw);item.mode=0o600;archive.addfile(item,io.BytesIO(raw))
        raw=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode();item=tarfile.TarInfo('FILES.json');item.size=len(raw);item.mode=0o600;archive.addfile(item,io.BytesIO(raw))
    arc=descriptor(archive_path);require(arc['bytes']<=MAX_BYTES,'checkpoint asset exceeds1GiB')
    restore=verify_archive(archive_path,arc['sha256'],destination=directory/'verified-restoration')
    if target_commit is None:
        target_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[3]).decode().strip()
    require(len(target_commit)==40 and all(c in '0123456789abcdef' for c in target_commit),'full commit identity required')
    package={'schema':'goal015_checkpoint_v1','status':'LOCAL_VERIFIED','github_status':'NOT_UPLOADED',
        'checkpoint_id':name,'directory':str(directory),'canonical_root':str(root),'plan_id':plan['plan_id'],
        'tag':'checkpoint-'+name,'target_commitish':target_commit,'reason':reason,
        'assets':[{'name':archive_path.name,'bytes':arc['bytes'],'sha256':arc['sha256']}],
        'archive_sha256':arc['sha256'],'restoration':restore,'fixture_only':plan.get('fixture_only') is True,
        'original_payloads_copied':False,'research_requests_sent_by_save':0}
    _save(directory/'checkpoint.json',package,exclusive=True)
    return {'status':'LOCAL_VERIFIED','path':str(directory),'checkpoint':descriptor(directory/'checkpoint.json'),
            'archive_sha256':arc['sha256'],'research_api_requests_sent':0}


def create_save_only(plan_path, *, reason='SAVE_ONLY_RETRY'):
    from .runtime import canonical_lock
    plan=verify_plan(plan_path)
    with canonical_lock(plan['canonical_root']):
        return create_checkpoint(plan_path,Path(plan['canonical_root'])/'enabled/ledger.json',reason=reason)


@lru_cache(maxsize=1)
def _github_helpers():
    """Load exact frozen pure gh helpers without importing generation/providers."""
    source=Path(__file__).resolve().parents[2]/'operations/goal006c/archive.py'
    selected={'RecoveryError','_fail','_gh','_pages'}
    tree=ast.parse(source.read_text(),filename=str(source))
    nodes=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in selected]
    require({node.name for node in nodes}==selected,'frozen GitHub helper definitions missing')
    namespace={'Path':Path,'json':json,'re':re,'subprocess':subprocess}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
    return namespace


class Github:
    """Corrected helper suppresses stderr/secret URLs and uses explicit API IDs."""
    def call(self,stage,args,*,output=None):return _github_helpers()['_gh'](stage,args,output=output)
    def pages(self,endpoint,stage):return _github_helpers()['_pages'](endpoint,stage)


def upload_checkpoint(directory, verification_directory, *, release_id=None, allow_create=False, github=None):
    """Save-only; a retry never executes/resumes generation or changes old Drafts."""
    directory=path_safe(directory);package=read_json(directory/'checkpoint.json')
    require(package['schema']=='goal015_checkpoint_v1' and package['status']=='LOCAL_VERIFIED'
            and package['directory']==str(directory) and package['checkpoint_id'].startswith('goal015-')
            and package['tag']=='checkpoint-'+package['checkpoint_id'],'Goal015 immutable package required')
    require(github is None or package.get('fixture_only') is True,'fake GitHub transport forbidden for real package')
    require(len(package['assets'])==1 and package['assets'][0]['name']=='goal015-state.tar.gz','exact Goal015 asset required')
    asset=package['assets'][0];original=descriptor(directory/asset['name'])
    require({k:original[k] for k in ('bytes','sha256')}=={k:asset[k] for k in ('bytes','sha256')},'local checkpoint asset changed')
    local_proof=verify_archive(directory/asset['name'],asset['sha256'])
    require(local_proof['plan_id']==package['plan_id'],'package/archive plan differs')
    target=path_safe(verification_directory);require(not target.exists(),'new upload verification directory required');target.mkdir(parents=True)
    gh=github or Github();base=f'repos/{REPOSITORY}'
    ancestry=gh.call('commit_ancestry',['api',f"{base}/compare/{package['target_commitish']}...{quote(BRANCH,safe='')}"])
    require(ancestry.get('status') in ('ahead','identical') and ancestry.get('behind_by')==0
            and ancestry.get('base_commit',{}).get('sha')==package['target_commitish'],'checkpoint commit not on dedicated branch')
    body=f"Goal015 immutable save-only checkpoint\nplan_id: {package['plan_id']}\narchive_sha256: {package['archive_sha256']}"
    def validate_release(release):
        require(type(release.get('id')) is int and release.get('draft') is True and release.get('published_at') is None
                and release.get('tag_name')==package['tag'] and release.get('target_commitish')==package['target_commitish']
                and release.get('body')==body,'release ID/tag/target/plan/archive/Draft mismatch')
        return release
    if release_id is not None:
        require(type(release_id) is int and release_id>0,'positive release ID required')
        release=gh.call('release_lookup',['api',f'{base}/releases/{release_id}']);require(release.get('id')==release_id,'release ID mismatch')
    else:
        matches=[r for r in gh.pages(base+'/releases','release_list') if r.get('tag_name')==package['tag']]
        require(len(matches)<=1,'ambiguous Draft identity')
        if matches:release=matches[0]
        else:
            require(allow_create,'Draft absent; explicit creation option required')
            request={'tag_name':package['tag'],'target_commitish':package['target_commitish'],'name':package['checkpoint_id'],'draft':True,'body':body}
            _save(target/'release-request.json',request,exclusive=True)
            release=gh.call('release_create',['api','--method','POST',base+'/releases','--input',str(target/'release-request.json')])
    release=validate_release(release);rid=release['id']
    rows=gh.pages(f'{base}/releases/{rid}/assets','asset_list');require(len(rows)<=1,'unexpected asset; never overwrite')
    uploaded=False
    if not rows:
        require(allow_create,'missing asset; upload not authorized')
        gh.call('asset_upload',['api','--method','POST',f"https://uploads.github.com/{base}/releases/{rid}/assets?name={quote(asset['name'],safe='')}",
            '-H','Content-Type: application/octet-stream','-H',f"Content-Length: {asset['bytes']}",'--input',str(directory/asset['name'])]);uploaded=True
        rows=gh.pages(f'{base}/releases/{rid}/assets','asset_list')
    require(len(rows)==1,'exact asset required');remote=rows[0]
    require(type(remote.get('id')) is int and remote.get('name')==asset['name'] and remote.get('size')==asset['bytes']
            and remote.get('digest')=='sha256:'+asset['sha256'] and remote.get('state')=='uploaded','remote asset identity mismatch')
    downloaded=target/asset['name']
    gh.call('asset_download',['api',f"{base}/releases/assets/{remote['id']}",'-H','Accept: application/octet-stream'],output=downloaded)
    require(descriptor(downloaded)['sha256']==asset['sha256'] and downloaded.stat().st_size==asset['bytes'],'redownload hash/size mismatch')
    restored=verify_archive(downloaded,asset['sha256'],destination=target/'restored')
    validate_release(gh.call('release_lookup',['api',f'{base}/releases/{rid}']))
    final=gh.pages(f'{base}/releases/{rid}/assets','asset_list')
    require(len(final)==1 and all(final[0].get(k)==remote.get(k) for k in ('id','name','size','digest','state')),'remote asset changed')
    require(descriptor(directory/asset['name'])==original,'local original changed during save')
    receipt={'status':'GITHUB_VERIFIED','release_id':rid,'asset_id':remote['id'],'draft':True,'uploaded':uploaded,
        'archive_sha256':asset['sha256'],'restoration':restored,'research_api_requests_sent':0,'generation_reexecuted':False,
        'fixture_only':package.get('fixture_only') is True,'github_transport':'FAKE_FIXTURE' if github is not None else 'GH_LIVE_SAVE_ONLY'}
    _save(target/'verification.json',receipt,exclusive=True);return receipt
