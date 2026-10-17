"""Goal015 future execution lifecycle; preparation and read-only status send nothing.

The immutable plan remains disabled. Only a separate real-TTY approval creates
an authority and a new ledger. Uncertain calls are never eligible for resend.
"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import uuid


REPOSITORY_ROOT=Path(__file__).resolve().parents[3]
EXECUTION_BINDING=REPOSITORY_ROOT/'reports/server_only/goal015/EXECUTION_BINDING.json'


class RuntimeErrorSafe(RuntimeError):
    pass


def require(ok, reason):
    if not ok:
        raise RuntimeErrorSafe(reason)


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def identity(plan):
    return hashlib.sha256(canonical({k:v for k,v in plan.items() if k != 'plan_id'})).hexdigest()


def path_safe(path):
    path=Path(path).absolute()
    require('..' not in path.parts and not any(p.is_symlink() for p in (path,*path.parents)), 'nonsymlink canonical path required')
    return path


def descriptor(path):
    path=path_safe(path);raw=path.read_bytes()
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def read_json(path):
    def pairs(values):
        out={}
        for k,v in values:
            require(k not in out,'duplicate JSON field');out[k]=v
        return out
    return json.loads(path_safe(path).read_text(),object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(RuntimeErrorSafe('nonfinite JSON')))


def bound(desc):
    require(descriptor(desc['path'])==desc,'bound artifact changed')
    return read_json(desc['path'])


def _fsync_directory(path):
    fd=os.open(path_safe(path),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def _save(path, value, *, exclusive=False):
    path=path_safe(path);raw=canonical(value)+b'\n';tmp=path.with_name('.pending-'+uuid.uuid4().hex)
    with tmp.open('xb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    try:
        if exclusive:
            os.link(tmp,path);tmp.unlink()
        else:os.replace(tmp,path)
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        if tmp.exists():tmp.unlink()


def _raw_save(path, raw):
    require(type(raw) is bytes,'raw response bytes required')
    with path_safe(path).open('xb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    fd=os.open(Path(path).parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return descriptor(path)


def _decimal(value):
    value=Decimal(str(value));require(value.is_finite() and value>=0,'nonnegative finite USD required');return value


def _verify_registry(plan,path):
    registry=read_json(EXECUTION_BINDING)
    require(registry['plan_id']==plan['plan_id'] and registry['canonical_root']==plan['canonical_root']
            and registry['plan']==descriptor(path),'prepared execution registry differs; copy/replay refused')


def verify_plan(plan_path):
    path=path_safe(plan_path);plan=read_json(path);root=path_safe(plan['canonical_root'])
    require(path==root/'plan.json','copied plan cannot activate')
    require(plan.get('fixture_only') is True or (root.parent==Path('/root/minecraft-runs') and root.name.startswith('goal015-')),
            'new dedicated Goal015 canonical root required')
    require(plan.get('schema')=='goal015_disabled_plan_v1' and plan.get('api_allowed') is False
            and plan.get('budget_approved') is False,'immutable disabled Goal015 plan required')
    require(plan.get('plan_id')==identity(plan),'plan identity changed')
    if plan.get('fixture_only') is not True:_verify_registry(plan,path)
    require(plan['planned_calls']==80 and plan['planned_buildings']==10 and plan['concurrency']==1
            and plan['max_retries']==0,'fixed pilot scope differs')
    require(_decimal(plan['proposed_budget_usd'])==Decimal('5'),'fixed unapproved USD5 proposal required')
    require(plan['pricing_contract']['freshness_seconds']==86400,'fixed pricing freshness required')
    require(plan.get('timeout_seconds')==600,'fixed600 second request timeout required')
    rows=plan['schedule'];require(len(rows)==80,'80 schedule rows required')
    require(len({r['call_id'] for r in rows})==80 and len({r['scene_id'] for r in rows})==10,'duplicate or missing schedule identities')
    require(all(re.fullmatch('[A-Za-z0-9_-]+',r['call_id']) for r in rows),'safe call IDs required')
    blocks=[]
    for offset in range(0,80,2):
        a,b=rows[offset:offset+2]
        require(a['paired_block_id']==b['paired_block_id'] and (a['scene_id'],a['builder'],a['repeat_index'])
                ==(b['scene_id'],b['builder'],b['repeat_index'])
                and {a['condition'],b['condition']}=={'SUMMARY_ONLY','FULL_RECORD'},'adjacent S/F paired block required')
        require(a['builder'] in ('openai','anthropic') and a['repeat_index'] in (1,2),'fixed Builder/repeat required')
        blocks.append(a['paired_block_id'])
    require(len(set(blocks))==40,'40 unique paired blocks required')
    cells={(r['scene_id'],r['builder'],r['condition'],r['repeat_index']) for r in rows}
    require(len(cells)==80,'duplicate experimental cell')
    for row in rows:
        payload=bound(row['payload']);require(isinstance(payload,dict),'payload object required');_decimal(row['reserve_usd'])
        require(type(row['input_reservation_tokens']) is int and row['input_reservation_tokens']>0,'positive input reservation required')
        if plan.get('fixture_only') is not True:
            require(row['input_reservation_tokens']=={'openai':400000,'anthropic':200000}[row['builder']],
                    'full-context conservative input reservation required')
            rates=plan['pricing_contract']['rates_per_million'][row['builder']]
            ceiling=(_decimal(row['input_reservation_tokens'])*_decimal(rates['input'])+Decimal(32768)*_decimal(rates['output']))/Decimal(1000000)
            require(_decimal(row['reserve_usd'])>=ceiling,'full input plus output-cap reservation required')
    for name,expected in plan['implementation_hashes'].items():
        require(descriptor(name)['sha256']==expected,'bound implementation changed')
    require(descriptor(plan['specification']['path'])==plan['specification'],'bound specification changed')
    bound(plan['pricing_evidence'])
    return plan


def pricing_fresh(plan, evidence=None, *, at=None):
    document=bound(evidence or plan['pricing_evidence'])
    current=at or datetime.now(timezone.utc)
    def timestamp(value):
        stamp=datetime.fromisoformat(value.replace('Z','+00:00'))
        require(stamp.tzinfo is not None and stamp.utcoffset().total_seconds()==0,'UTC evidence timestamp required')
        require(0<=(current-stamp).total_seconds()<=plan['pricing_contract']['freshness_seconds'],
                'official evidence older than24 hours or future timestamp')
        return stamp
    stamp=timestamp(document['retrieved_at_utc'])
    if plan.get('fixture_only') is not True:
        require(document.get('review_status')=='REVIEWED_OFFICIAL_DOCUMENTS'
                and isinstance(document.get('documents'),list) and document['documents'],
                'reviewed official pricing documents required')
        original=bound(plan['pricing_evidence'])
        require(sorted(d['url'] for d in document['documents'])==sorted(d['url'] for d in original['documents']),
                'all original official pricing sources required')
        for item in document['documents']:
            require(descriptor(item['path'])=={k:item[k] for k in ('path','bytes','sha256')},
                    'official pricing source document changed')
            require(isinstance(item.get('url'),str) and item['url'].startswith(('https://developers.openai.com/',
                    'https://platform.claude.com/','https://platform.openai.com/','https://docs.anthropic.com/')),
                    'official source HTTPS URL required')
            require(item.get('http_status')==200,'successful official document retrieval required')
            require(stamp<=timestamp(item['retrieved_at_utc']),'evidence timestamp must cover oldest document')
    require(document['pricing_contract']==plan['pricing_contract'],'pricing/model/contract change requires new approval')
    shutdown=plan['pricing_contract'].get('shutdown_at_utc')
    if shutdown is not None:
        limit=datetime.fromisoformat(shutdown.replace('Z','+00:00'))
        require(limit.tzinfo is not None and current<limit,'fixed model shutdown reached; new contract/approval required')
    return document


def approval_text(plan):
    return f"APPROVE GOAL015 {plan['plan_id']} USD {plan['proposed_budget_usd']} 80 CALLS RETRY 0"


def confirm_tty(plan):
    """Separate /dev/tty read/write handles; never r+ on a nonseekable device."""
    expected=approval_text(plan)
    try:
        with open('/dev/tty','r',encoding='utf-8') as reader, open('/dev/tty','w',encoding='utf-8') as writer:
            require(os.isatty(reader.fileno()) and os.isatty(writer.fileno()),'real controlling TTY required')
            writer.write(f"Goal015 plan {plan['plan_id']}\nSUMMARY_ONLY / FULL_RECORD;80 calls;10 buildings;retry0;USD cap {plan['proposed_budget_usd']}\nType exactly: {expected}\n> ")
            writer.flush();answer=reader.readline(len(expected)+4)
            require(answer!='' and answer.rstrip('\r\n')==expected,'explicit exact approval absent')
    except RuntimeErrorSafe:raise
    except (OSError,UnicodeError,ValueError):raise RuntimeErrorSafe('TTY unavailable or read/write failed') from None
    return True


def preview(plan_path):
    plan=verify_plan(plan_path)
    return {'status':'PILOT80_READY_AWAITING_EXPLICIT_APPROVAL','plan_id':plan['plan_id'],
        'planned_calls':80,'planned_independent_buildings':10,'budget_approved':False,
        'model_account_availability':'NOT_RUN','concurrency':1,'max_retries':0,
        'new_budget_proposal_usd':str(plan['proposed_budget_usd']),'research_api_requests_sent':0,
        'canonical_root':plan['canonical_root'],'approval_text':approval_text(plan)}


def status(plan_path):
    plan=verify_plan(plan_path);path=Path(plan['canonical_root'])/'enabled/ledger.json'
    if not path.exists():result=preview(plan_path)
    else:
        ledger=read_json(path);require(ledger['plan_id']==plan['plan_id'],'ledger identity differs')
        result={'status':ledger['status'],'plan_id':plan['plan_id'],'calls':ledger['calls'],
                'budget':accounting(ledger),'approval_present':(path.parent/'approval.json').is_file()}
    products=[];directory=Path(plan['canonical_root'])/'evaluations'
    if directory.exists():
        path_safe(directory)
        for summary_path in sorted(directory.rglob('summary.json')):
            summary=read_json(summary_path)
            binding=summary.get('plan_binding') or {}
            require(summary.get('schema')=='goal015_paired_pilot_summary_v1' and binding.get('plan_id')==plan['plan_id']
                    and binding.get('plan')==descriptor(plan_path),'saved evaluation plan identity differs')
            artifact=summary['evaluation_artifact']
            require(path_safe(artifact['path']).is_relative_to(directory) and descriptor(artifact['path'])==artifact,
                    'saved evaluation artifact identity differs')
            products.append({'summary':descriptor(summary_path),'evaluation_artifact':artifact,
                'planned_calls':summary['planned_calls'],'actual_metric_rows':summary['actual_metric_rows'],
                'evaluated_protocol_valid_rows':summary['evaluated_protocol_valid_rows'],
                'source_ledger':binding.get('ledger')})
    result['evaluation_products']=products
    result['latest_evaluation_counts']=({k:products[-1][k] for k in ('planned_calls','actual_metric_rows','evaluated_protocol_valid_rows')} if products else None)
    return result


@contextmanager
def canonical_lock(root):
    path=path_safe(Path(root)/'execution.lock')
    with path.open('a+b') as f:
        try:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeErrorSafe('canonical execution already active') from None
        try:yield
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)


def accounting(ledger):
    known=sum((_decimal(r.get('known_cost_usd') or 0) for r in ledger['calls']),Decimal(0))
    held=sum((_decimal(r.get('reservation_usd') or 0) for r in ledger['calls'] if r['state']!='SETTLED'),Decimal(0))
    return {'known_cost_usd':str(known),'unsettled_and_unknown_holds_usd':str(held),'committed_usd':str(known+held)}


def _validate_ledger(plan,ledger,approval,*,for_resume=False):
    require(ledger['plan_id']==approval['plan_id']==plan['plan_id'] and approval['canonical_root']==plan['canonical_root']
            and approval['approved_budget_usd']==plan['proposed_budget_usd'] and approval['scope_calls']==80
            and approval['status']=='APPROVED','same canonical authority required')
    require(len(ledger['calls'])==80,'ledger scope differs')
    if for_resume:
        blockers=any(row.get('failure_kind') in ('response_model_mismatch','unexpected_response_feature','incomplete_response') for row in ledger['calls'])
        require(ledger.get('requires_new_explicit_approval') is not True and not blockers,
                'contract mismatch requires new explicit approval')
    for source,row in zip(plan['schedule'],ledger['calls']):
        require(all(row.get(k)==v for k,v in source.items()),'ledger/schedule identity differs')
        require(row['state'] in ('NOT_SENT','SENT_PENDING','UNKNOWN_USAGE','SETTLED'),'invalid call state')
        require(_decimal(row['reservation_usd']) in (Decimal(0),_decimal(row['reserve_usd'])),'reservation changed')
        if row['state']!='NOT_SENT':require(_decimal(row['reservation_usd'])==_decimal(row['reserve_usd']),'attempt requires whole reservation')
        if row['state']=='NOT_SENT':require(row['raw'] is None and row['send_status']=='NOT_SENT','NOT_SENT requires no transmission evidence')
        if row['state']=='SETTLED':
            require(row['usage_known'] is True and row['raw'] is not None and row['known_cost_usd'] is not None,
                    'settlement without durable raw/known cost')
            _decimal(row['known_cost_usd'])
            if for_resume:require(_decimal(row['known_cost_usd'])<=_decimal(row['reserve_usd']),'settled usage exceeds reservation')
        else:require(row['known_cost_usd'] is None and row['usage_known'] is False,'unsettled call cannot carry known cost')
        if row.get('raw') is not None:require(descriptor(row['raw']['path'])==row['raw'],'saved raw response changed')
        if row.get('response_text_path') is not None:require(descriptor(row['response_text_path']['path'])==row['response_text_path'],'saved response text changed')
    if for_resume:require(_decimal(accounting(ledger)['committed_usd'])<=_decimal(plan['proposed_budget_usd']),'existing committed budget exceeds cap')


def execute(plan_path, *, resume=False, adapter=None, confirmation=None, refresh_evidence=None, checkpoint_callback=None):
    plan=verify_plan(plan_path);root=Path(plan['canonical_root']);fixture=plan.get('fixture_only') is True
    require(fixture or (adapter is None and confirmation is None and checkpoint_callback is None),'mock hooks forbidden for real plan')
    pricing_fresh(plan,refresh_evidence)
    if not fixture:
        try:
            committed=subprocess.check_output(['git','show','HEAD:reports/server_only/goal015/EXECUTION_BINDING.json'],cwd=REPOSITORY_ROOT,stderr=subprocess.DEVNULL)
            branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPOSITORY_ROOT).decode().strip()
        except (OSError,subprocess.CalledProcessError):raise RuntimeErrorSafe('committed execution registry required') from None
        require(committed==EXECUTION_BINDING.read_bytes() and branch=='research/server-only-20261013','committed registry/dedicated branch required')
    enabled=root/'enabled'
    if not resume:
        require(not enabled.exists(),'existing execution cannot be started again; use resume')
        require((confirmation or confirm_tty)(plan) is True,'approval refused')
    with canonical_lock(root):
        require(verify_plan(plan_path)==plan,'plan changed while awaiting approval')
        pricing_fresh(plan,refresh_evidence)
        if resume:
            require(enabled.is_dir(),'no approved execution to resume')
            approval=read_json(enabled/'approval.json');ledger=read_json(enabled/'ledger.json')
            authority=read_json(enabled/'authority.json')
            require(authority['plan_id']==plan['plan_id'] and authority['canonical_root']==plan['canonical_root']
                    and authority['plan']==descriptor(plan_path) and authority['approval']==descriptor(enabled/'approval.json'),
                    'canonical authority or approval changed')
            _validate_ledger(plan,ledger,approval,for_resume=True)
        else:
            require(not enabled.exists(),'duplicate activation refused');enabled.mkdir();_fsync_directory(root)
            (enabled/'responses').mkdir();_fsync_directory(enabled)
            approval={'status':'APPROVED','plan_id':plan['plan_id'],'canonical_root':plan['canonical_root'],
                'approved_budget_usd':plan['proposed_budget_usd'],'scope_calls':80,'approved_at_utc':now(),
                'approval_method':'FIXTURE_ONLY' if fixture else 'EXACT_REAL_TTY'}
            _save(enabled/'approval.json',approval,exclusive=True)
            _save(enabled/'authority.json',{'status':'APPROVED','plan_id':plan['plan_id'],
                'canonical_root':plan['canonical_root'],'plan':descriptor(plan_path),
                'approval':descriptor(enabled/'approval.json'),'scope_calls':80,'max_retries':0},exclusive=True)
            ledger={'schema':'goal015_execution_ledger_v1','plan_id':plan['plan_id'],'status':'APPROVED_NOT_SENT',
                'calls':[dict(row,state='NOT_SENT',reservation_usd='0',known_cost_usd=None,usage_known=False,
                    send_status='NOT_SENT',response_received=False,completed_response=False,model_output_failure=False,
                    failure_kind=None,raw=None,response_text_path=None) for row in plan['schedule']]}
            _save(enabled/'ledger.json',ledger,exclusive=True)
        if refresh_evidence is not None:
            evidence_dir=enabled/'pricing_refresh';evidence_dir.mkdir(exist_ok=True)
            target=evidence_dir/(refresh_evidence['sha256']+'.json')
            if not target.exists():_save(target,{'evidence':refresh_evidence,'accepted_at_utc':now(),'plan_id':plan['plan_id']},exclusive=True)
        if adapter is None:
            from . import adapter
        last_checkpoint=None;failure=None;ledger['status']='RUNNING';_save(enabled/'ledger.json',ledger)
        def checkpoint(reason):
            nonlocal last_checkpoint
            if checkpoint_callback:last_checkpoint=checkpoint_callback(plan_path,enabled/'ledger.json',reason)
            else:
                from .checkpoint import create_checkpoint
                last_checkpoint=create_checkpoint(plan_path,enabled/'ledger.json',reason=reason)
            ledger['last_checkpoint']=last_checkpoint;_save(enabled/'ledger.json',ledger)
        try:
            for offset in range(0,80,2):
                block=ledger['calls'][offset:offset+2];pending=[r for r in block if r['state']=='NOT_SENT']
                if not pending:continue
                pricing_fresh(plan,refresh_evidence)
                needed=sum((_decimal(r['reserve_usd'])-_decimal(r['reservation_usd']) for r in pending),Decimal(0))
                require(_decimal(accounting(ledger)['committed_usd'])+needed<=_decimal(plan['proposed_budget_usd']),
                        'next complete paired block exceeds approved cap')
                for row in pending:row['reservation_usd']=row['reserve_usd']
                _save(enabled/'ledger.json',ledger)
                for row in pending:
                    pricing_fresh(plan,refresh_evidence)
                    payload=bound(row['payload']);adapter.validate_payload(row['builder'],payload)
                    row.update(state='SENT_PENDING',send_status='POSSIBLY_SENT',started_at_utc=now())
                    _save(enabled/'ledger.json',ledger)
                    raw=None;transport_failure=None
                    try:
                        raw=adapter.send_request(row['builder'],payload,api_allowed=True,permission=True,timeout=plan['timeout_seconds'])
                        row.update(send_status='POST_RETURNED',response_received=True,http_status=200,raw_complete=True)
                    except Exception as exc:
                        transport_failure=type(exc).__name__
                        raw=getattr(exc,'raw_response',None)
                        row.update(http_status=getattr(exc,'http_status',getattr(exc,'status_code',None)),
                            response_received=getattr(exc,'http_response_received',False) is True,
                            failure_kind=getattr(exc,'failure_kind','transport_failure'),raw_complete=getattr(exc,'raw_complete',False) is True)
                    if raw is not None:
                        row['raw']=_raw_save(enabled/'responses'/(row['call_id']+'.raw.json'),raw)
                        _save(enabled/'ledger.json',ledger)
                    if transport_failure:
                        row.update(state='UNKNOWN_USAGE',usage_known=False,transport_error_type=transport_failure)
                        if raw is not None and row['raw_complete']:
                            try:
                                failed_response=adapter.parse_response(row['builder'],raw)
                                rates=plan['pricing_contract']['rates_per_million'][row['builder']]
                                failed_usage=adapter.classify_usage(row['builder'],failed_response.usage,
                                    {'input_usd_per_million':rates['input'],'output_usd_per_million':rates['output']})
                                if failed_usage['usage_known'] is True:
                                    row.update(state='SETTLED',usage_known=True,usage=failed_usage,known_cost_usd=str(failed_usage['cost_usd']))
                            except Exception:pass
                        _save(enabled/'ledger.json',ledger)
                        if row['usage_known']:
                            require(_decimal(row['known_cost_usd'])<=_decimal(row['reserve_usd']),'HTTP usage exceeded conservative reservation; stopped')
                            require(_decimal(accounting(ledger)['committed_usd'])<=_decimal(plan['proposed_budget_usd']),'HTTP usage exceeded approved budget; stopped')
                        raise RuntimeErrorSafe('external request failure; unresolved reservations retained')
                    response=adapter.parse_response(row['builder'],raw)
                    row.update(completed_response=response.complete,response_model=response.model,
                        response_id=response.response_id,failure_kind=response.failure_kind,
                        model_output_failure=response.failure_kind in ('parse_failure','validation_failure','execution_failure','truncation','refusal'))
                    if response.failure_kind in ('response_model_mismatch','unexpected_response_feature','incomplete_response') or (
                            response.complete is not True and not row['model_output_failure']):
                        ledger['requires_new_explicit_approval']=True
                    _save(enabled/'ledger.json',ledger)
                    if response.text is not None:
                        row['response_text_path']=_raw_save(enabled/'responses'/(row['call_id']+'.text.txt'),response.text.encode())
                    rates=plan['pricing_contract']['rates_per_million'][row['builder']]
                    usage=adapter.classify_usage(row['builder'],response.usage,
                        {'input_usd_per_million':rates['input'],'output_usd_per_million':rates['output']})
                    row['usage_known']=usage['usage_known'] is True;row['usage']=usage
                    if row['usage_known']:
                        row.update(state='SETTLED',known_cost_usd=str(usage['cost_usd']))
                    else:row['state']='UNKNOWN_USAGE'
                    row['finished_at_utc']=now();_save(enabled/'ledger.json',ledger)
                    require(row['usage_known'],'unknown usage; full reservation retained')
                    require(_decimal(row['known_cost_usd'])<=_decimal(row['reserve_usd']),'observed usage exceeded conservative reservation; stopped')
                    require(_decimal(accounting(ledger)['committed_usd'])<=_decimal(plan['proposed_budget_usd']),'approved budget exhausted')
                    if response.failure_kind in ('response_model_mismatch','unexpected_response_feature','incomplete_response') or (
                            response.complete is not True and not row['model_output_failure']):
                        ledger['requires_new_explicit_approval']=True
                        _save(enabled/'ledger.json',ledger)
                        raise RuntimeErrorSafe('response contract mismatch; new explicit approval required')
                next_scene=ledger['calls'][offset+2]['scene_id'] if offset+2<80 else None
                if next_scene!=block[-1]['scene_id']:checkpoint('BUILDING_COMPLETE')
            ledger['status']='COMPLETE' if all(r['state']=='SETTLED' for r in ledger['calls']) else 'PARTIAL_WITH_UNCERTAIN_CALLS'
        except BaseException as exc:
            failure={'type':type(exc).__name__,'reason':str(exc) if isinstance(exc,RuntimeErrorSafe) else 'execution interrupted; inspect saved state'}
            ledger['status']='SAFE_STOP';ledger['halt']=failure
        finally:
            _save(enabled/'ledger.json',ledger)
            try:checkpoint('SAFE_STOP' if failure else 'FINAL')
            except Exception as exc:
                ledger['checkpoint_failure']={'type':type(exc).__name__,'reason':'local checkpoint failed; retry save only'}
                _save(enabled/'ledger.json',ledger)
        return {'status':ledger['status'],'plan_id':plan['plan_id'],'budget':accounting(ledger),
            'calls_not_sent':sum(r['state']=='NOT_SENT' for r in ledger['calls']),
            'calls_uncertain':sum(r['state'] in ('SENT_PENDING','UNKNOWN_USAGE') for r in ledger['calls']),
            'calls_settled':sum(r['state']=='SETTLED' for r in ledger['calls']),
            'checkpoint':last_checkpoint,'failure':failure,'fixture_only':fixture}
