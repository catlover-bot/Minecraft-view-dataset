"""One-off CPU-only successor preparation using the unchanged production CLI."""
from pathlib import Path
from contextlib import redirect_stdout
from datetime import datetime, timezone
from decimal import Decimal
import io,json,os,resource,runpy,socket,subprocess,sys,time
ROOT=Path('/root/workspace/Minecraft-view-dataset');sys.path.insert(0,str(ROOT))
for name in ('OPENAI_API_KEY','ANTHROPIC_API_KEY'):
    os.environ.pop(name,None)
def no_network(*args, **kwargs):
    raise RuntimeError('NO_NETWORK_DURING_GOAL006B1_CPU_PREPARE')
socket.socket.connect=no_network
socket.socket.connect_ex=no_network
from offline_evaluation.operations.goal006b import planning,runner
from offline_evaluation.server_v2.data import sha256_file
from offline_evaluation.server_v2.generation import GenerationError,read_json,utc_now,object_hash
out=ROOT/'reports/server_only/goal006b1'
code_commit=sys.argv[1]
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==code_commit
oldpath=Path('/root/minecraft-runs/goal006b-supplement20-20261015T195511Z/plan.json')
old=read_json(oldpath);oldi=old['identity'];oldsha=sha256_file(oldpath)
assert oldsha=='87760792f4b63f887c9f1a2547dd8c8513356a33018978309cb63e4fc0487aa8'
assert not any(Path(p).exists() for p in (oldi['enabled_directory'],oldi['authority_path'],oldi['authority_path']+'.lock'))
try:
    planning.load_plan(oldpath)
except GenerationError as exc:
    assert str(exc)=='supplement_implementation_hash_mismatch'
    old_rejected=str(exc)
else:
    raise AssertionError('old plan must reject the changed code')
source=read_json(planning.SOURCE_PLAN);si=source['identity']
paths=[Path(row['path']) for row in oldi['source_snapshot']['files']]
before={str(p):{'sha256':sha256_file(p),'bytes':p.stat().st_size} for p in paths}
assert all(before[r['path']]=={'sha256':r['sha256'],'bytes':r['bytes']} for r in oldi['source_snapshot']['files'])
# Keep the production-required goal006b- prefix; never reuse an existing directory.
destination=Path('/root/minecraft-runs')/('goal006b-ttyfix20-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
entry=ROOT/'offline_evaluation/operations/goal006b/cli.py'
def cli(args):
    sys.argv=[str(entry),*args];stream=io.StringIO()
    try:
        with redirect_stdout(stream):runpy.run_path(str(entry),run_name='__main__')
    except SystemExit as exc:
        if exc.code:raise
    return json.loads(stream.getvalue())
os.chdir('/tmp');start=time.monotonic()
prepared=cli(['--prepare',str(destination),'--checkpoint-verification',str(ROOT/'reports/server_only/goal006b/existing781_github_verification.json'),'--commit',code_commit])
prepare_elapsed=time.monotonic()-start
plan_path=destination/'plan.json';child=planning.load_plan(plan_path);identity=child['identity']
start_preview=time.monotonic();preview=cli(['--plan',str(plan_path),'--preview']);preview_elapsed=time.monotonic()-start_preview
assert child['api_allowed'] is False and child['status']=='UNAPPROVED_SUPPLEMENT20'
assert preview['api_allowed'] is False and preview['approved'] is False and preview['supplemental_attempted']==0
assert preview['pricing_current_utc_day'] is True
assert not any(Path(p).exists() for p in (identity['enabled_directory'],identity['authority_path'],identity['authority_path']+'.lock'))
assert sorted(str(p.relative_to(destination)) for p in destination.rglob('*'))==['plan.json']
assert sha256_file(oldpath)==oldsha
assert before=={str(p):{'sha256':sha256_file(p),'bytes':p.stat().st_size} for p in paths}
assert si['implementation_hashes']==planning.original_plan.implementation_hashes()
assert identity['implementation_hashes']==planning.implementation_hashes()
for path,sha in identity['implementation_hashes'].items():
    from hashlib import sha256
    assert sha256(subprocess.check_output(['git','show',f'{code_commit}:{path}'],cwd=ROOT)).hexdigest()==sha
changedkeys=[k for k in sorted(set(oldi)|set(identity)) if oldi.get(k)!=identity.get(k)]
assert changedkeys==['commit','created_at_utc','enabled_directory','implementation_hashes','plan_path']
changedcode=[k for k in sorted(identity['implementation_hashes']) if identity['implementation_hashes'][k]!=oldi['implementation_hashes'][k]]
assert changedcode==['offline_evaluation/operations/goal006b/cli.py']
assert set(child)==set(old) and all(child[k]==old[k] for k in child if k not in ('identity','identity_hash'))
payloads=[]
for local,row in enumerate(identity['schedule']):
    global_index=row['schedule_index']
    original=si['schedule'][global_index-160]
    expected=object_hash(planning.original_runner._payload(si,original))
    actual=object_hash(planning.original_runner._payload(identity,row))
    assert actual==expected
    payloads.append({'local_index':local,'schedule_index':global_index,'provider':row['provider'],'model':row['model'],'input_id':row['input_id'],'request_payload_sha256':actual,'intent':identity['intents'][local]})
assert [r['schedule_index'] for r in payloads]==list(range(940,960))
assert payloads[0]['request_payload_sha256']=='b31b7d3474c6fa5c023a327e555354789fa98809c62f7a3d33a8c581809e0fc0'
assert identity['source_snapshot']==oldi['source_snapshot'] and identity['parent']==oldi['parent']
assert identity['budgets']==oldi['budgets'] and identity['pricing_document']==oldi['pricing_document']
max_additional=sum((Decimal(str(planning.original_runner._reservation(identity,row)[2])) for row in identity['schedule']),Decimal(0))
max_combined=Decimal(str(identity['parent']['usage']['cost_usd_upper_bound']))+max_additional
assert max_additional==Decimal('5.293760') and max_combined==Decimal('26.317147')
assert max_combined<=Decimal(str(identity['budgets']['max_cost_usd']))
assert identity['parent']['usage']['calls']+20==identity['budgets']['max_total_calls']==991
assert identity['budgets']['max_automatic_retries']==0 and identity['budgets']['max_output_tokens']==32768
assert identity['pricing_document']['verified_on']==utc_now()[:10]=='2026-10-15'
for name,value in [('supplement_plan.json',child),('preview.json',preview)]:
    with (out/name).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')
# Canonical bytes are preserved in the report copy, independent of JSON formatting.
(out/'supplement_plan.json').write_bytes(plan_path.read_bytes())
report={'verified_at_utc':utc_now(),'canonical_plan_path':str(plan_path),'plan_sha256':sha256_file(plan_path),'plan_identity_hash':child['identity_hash'],'implementation_commit':code_commit,
 'old_plan_path':str(oldpath),'old_plan_sha256':oldsha,'old_identity_hash':old['identity_hash'],'old_load_rejected':old_rejected,
 'changed_identity_keys':changedkeys,'changed_implementation_paths':changedcode,
 'implementation_hash_changes':{p:{'old':oldi['implementation_hashes'][p],'new':identity['implementation_hashes'][p]} for p in changedcode},
 'unchanged_identity_keys':[k for k in sorted(identity) if k not in changedkeys],
 'status':'UNAPPROVED_SUPPLEMENT20_CPU_VERIFIED','api_allowed':False,'approved':False,'enabled_directory_exists':False,'child_authority_exists':False,'child_authority_lock_exists':False,'child_approval_exists':False,'child_ledger_exists':False,'new_reservations':0,'only_child_file':'plan.json',
 'source_original_files_unchanged':len(before),'source_halt_reason':identity['source_snapshot']['halt_reason'],'source_controls_sha256':{k:identity['source_snapshot'][k] for k in ('plan_sha256','ledger_sha256','approval_sha256','authority_sha256')},
 'frozen_original_implementation_files_unchanged':len(si['implementation_hashes']),'supplement_bound_implementation_files':len(identity['implementation_hashes']),'all_bound_hashes_match_pushed_commit':True,
 'exact_payload_comparisons':payloads,'manual_auth_retry_slots':1,'first_attempt_slots':19,'budgets':identity['budgets'],'max_additional_reservation_usd':str(max_additional),'max_combined_budget_usd':str(max_combined),'actual_billed_usd':None,
 'pricing_verified_on':identity['pricing_document']['verified_on'],'pricing_document_hash':identity['pricing_document_hash'],'pricing_document_unchanged':True,'pricing_refetch_required':False,'source_pricing_revisions_added':0,
 'source_full_validator_at_prepare':'continuation_v1.runner._load_enabled','metrics_recomputed':0,'research_api_requests':0,'network_disabled_in_python':True,'provider_keys_removed_without_reading_values':True,'cwd_for_cli':str(Path.cwd()),'prepare_cli':prepared,'prepare_seconds':prepare_elapsed,'preview_seconds':preview_elapsed,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
with (out/'preparation_verification.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('exact_payload_comparisons','unchanged_identity_keys')},indent=2))
