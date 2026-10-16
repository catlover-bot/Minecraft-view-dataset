"""CPU-only Goal007 integration after a verified immutable-source audit gate."""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import resource
import shlex
import shutil
import subprocess
import sys
import time

from . import audit, integration, statistics
from ...server_v2.data import sha256_file
from ...server_v2.generation import GenerationError, read_json, utc_now
from ...server_v2.io_utils import ROOT


def implementation_hashes():
    from ...operations.goal006c.planning import implementation_hashes as frozen
    result=frozen()
    for p in Path(__file__).parent.rglob('*'):
        if p.is_file() and 'tests' not in p.parts and (p.suffix=='.py' or p.name=='plot_requirements.lock'):
            result[p.relative_to(ROOT).as_posix()]=sha256_file(p)
    return dict(sorted(result.items()))


def verify_code_commit(commit):
    if not isinstance(commit,str) or not re.fullmatch('[0-9a-f]{40}',commit):
        raise GenerationError('goal007_explicit_code_commit_required')
    hashes=implementation_hashes()
    for name,expected in hashes.items():
        result=subprocess.run(['git','show',f'{commit}:{name}'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if result.returncode or hashlib.sha256(result.stdout).hexdigest()!=expected:
            raise GenerationError('goal007_code_commit_bytes_mismatch')
    return hashes


def failure_tables(integrated):
    counts=Counter()
    for kind,rows in [('attempt',integrated['attempts']),('selected_slot',integrated['rows'])]:
        for row in rows:
            if row['failure_class']!='success':
                counts[(kind,row['provider'],row['condition'],row['failure_class'],row.get('failure_kind'),row['status'])]+=1
    fields=('record_type','provider','condition','failure_class','failure_kind','status')
    taxonomy=[{**dict(zip(fields,key)),'count':count} for key,count in sorted(counts.items())]
    model=[r for r in integrated['attempts'] if r['failure_class'] in {'model_output_failure','other_completed_failure'}]
    operational=[r for r in integrated['attempts'] if r['failure_class'] in {'operational_unknown','operational_error'}]
    return taxonomy,model,operational


def run(args):
    started=time.monotonic(); destination=Path(args.destination).absolute()
    if destination.exists() or not destination.is_relative_to(Path('/root/minecraft-runs')):
        raise GenerationError('goal007_new_research_output_directory_required')
    code_hashes=verify_code_commit(args.code_commit)
    gate=audit.verify_gate(args.audit_gate)
    manifest,sources=integration.load_sources(gate)
    plans={label:source['plan'] for label,source in sources.items()}
    rq2=read_json(Path(args.rq2_audit))
    if shutil.disk_usage(destination.parent).free<10*1024**3:
        raise GenerationError('goal007_at_least10GiB_free_required')
    with audit.source_locks(plans),audit.no_python_network():
        audit.process_audit(); audit.verify_gate(gate)
        destination.mkdir(parents=True,exist_ok=False)
        data=destination/'data'; data.mkdir()
        print('GOAL007 source audit gate PASS; integrate 961 attempts into960 slots',flush=True)
        integrated=integration.integrate(manifest,sources)
        metrics=integration.evaluate_missing(integrated,sources,destination/'metric_cache')
        costs=integration.cost_audit(sources,integrated['attempts'])
        print(f"GOAL007 strict metrics: reused={metrics['reused_count']} computed={metrics['computed_count']} resource_limits={len(metrics['resource_limit_global_indices'])}",flush=True)
        taxonomy,model_errors,operational_errors=failure_tables(integrated)
        integration.write_new_json(data/'final_slots_960.json',integrated['rows'])
        integration.write_new_csv(data/'final_slots_960.csv',integrated['rows'])
        integration.write_new_csv(data/'attempts_audit.csv',integrated['attempts'])
        mapping_fields=('global_schedule_index','source_label','source_plan_path','source_plan_identity','source_run','source_protocol',
                        'original_attempt_id','selected_attempt_id','attempt_history_ids','manual_retry','selection_reason',
                        'provider','builder_model','description_model','condition','input_id','gt_grid_sha256','request_payload_hash')
        integration.write_new_csv(data/'source_to_slot_mapping.csv',[{k:r[k] for k in mapping_fields} for r in integrated['rows']])
        integration.write_new_json(data/'metric_cache_validation.json',metrics)
        integration.write_new_json(data/'cost_validation.json',costs)
        for name,rows in [('failure_taxonomy',taxonomy),('model_output_errors',model_errors),('operational_errors',operational_errors)]:
            integration.write_new_csv(data/(name+'.csv'),rows)
        print('GOAL007 predeclared building statistics: 2000 bootstrap /10000 sign-flip,20 contrasts',flush=True)
        results=statistics.analyze_slots(integrated['rows'],audit_gate={'status':'PASS','input_kind':'verified_sources'})
        validation={**integrated['validation'],'attempt_records':len(integrated['attempts']),
                    'source_audit_gate':str(Path(args.audit_gate).absolute()),'source_audit_gate_sha256':sha256_file(args.audit_gate),
                    'strict_metrics':metrics,'denominators':results['denominators']['overall'],'historical_calls':costs['historical_calls'],
                    'operational_error_attempts':len(operational_errors),'research_api_requests_sent':0,'api_key_queries':0}
        integration.write_new_json(data/'data_validation.json',validation)
        cost_rows=[]
        for label,source in sources.items():
            entries=source['ledger']['entries']; known=sum((Decimal(str(e['charged_cost_usd'])) for e in entries if e['usage'] is not None),Decimal(0))
            held=sum((Decimal(str(e['charged_cost_usd'])) for e in entries if e['usage'] is None),Decimal(0))
            cost_rows.append({'source_label':label,'calls':len(entries),'known_usage_computed_usd':str(known),
                              'unknown_usage_reservation_usd':str(held),'budget_accounted_usd':str(known+held)})
        integration.write_new_csv(data/'cost_validation.csv',cost_rows)
        replay_destination=str(Path('/root/minecraft-runs')/'goal007-analysis-replay-CHOOSE_NEW_UTC')
        command=shlex.join([str(ROOT/'.venv-server-v2-20261013/bin/python'),'-B','-m','offline_evaluation.analysis.goal007.cli',
            '--audit-gate',str(Path(args.audit_gate).absolute()),'--destination',replay_destination,
            '--code-commit',args.code_commit,'--plot-python',str(Path(args.plot_python).absolute()),'--rq2-audit',str(Path(args.rq2_audit).absolute())])
        context={'status':'PASS','input_kind':'verified_sources','fixed_scope_attempts':len(integrated['attempts']),
            'historical_calls':costs['historical_calls'],'operational_error_attempts':len(operational_errors),
            'unknown_usage_holds':costs['unknown_usage_holds'],'cost_rows':cost_rows,'cost_validation':costs,
            'source_proofs':[{'source_label':label,**proof} for label,proof in gate['github_checkpoints'].items()],
            'reproduction_command':command,'failure_taxonomy':taxonomy,'error_rows':model_errors+operational_errors,'code_commit':args.code_commit,
            'metrics_cache_validation':metrics,'source_immutable_file_count':gate['immutable_file_count']}
        bundle={'results':results,'audit_context':context,'rq2_audit':rq2}
        render_input=destination/'statistical_input.json'; integration.write_new_json(render_input,bundle)
        print('GOAL007 render verified tables, figures, and paper drafts',flush=True)
        subprocess.run([args.plot_python,'-B','-m','offline_evaluation.analysis.goal007.render','--input',str(render_input),
                        '--destination',str(destination/'results')],cwd=ROOT,check=True)
        audit.verify_gate(gate); audit.process_audit()
        if implementation_hashes()!=code_hashes:
            raise GenerationError('goal007_analysis_code_changed_during_run')
        inventory=read_json(gate['inventory_path'])
        products={str(p.relative_to(destination)):{'sha256':sha256_file(p),'bytes':p.stat().st_size}
                  for p in sorted(destination.rglob('*')) if p.is_file() and not any(s.startswith('.') for s in p.relative_to(destination).parts)}
        final={'version':integration.VERSION,'analysis_label':integration.ANALYSIS_LABEL,'status':'FIXED60_ONE_OPERATIONAL_MISSING',
            'created_at_utc':utc_now(),'created_code_commit':args.code_commit,'code_hashes':code_hashes,
            'source_manifest':{'path':gate['manifest_path'],'sha256':gate['manifest_sha256'],'identity_hash':manifest['identity_hash']},
            'sources':gate['sources'],'source_files':inventory['files'],'source_tree_inventories':inventory['trees'],
            'github_source_checkpoints':gate['github_checkpoints'],'parent160_checkpoint':gate['parent160_checkpoint'],
            'audit_gate_path':str(Path(args.audit_gate).absolute()),'audit_gate_sha256':sha256_file(args.audit_gate),
            'source_inventories_before_after_equal':True,'missing_global_indices':[942],'approved_retry_global_indices':[940],
            'all_slots_measured':False,'fixed_slots':960,'fixed_scope_attempts':961,'historical_calls':costs['historical_calls'],
            'research_api_requests_sent':0,'api_key_queries':0,'github_mutations_by_analysis':0,
            'output_root':str(destination),'products':products,'cost_validation':costs,
            'strict_metric_evaluation':metrics,'rq2_audit':{'path':str(Path(args.rq2_audit).absolute()),'sha256':sha256_file(args.rq2_audit),'status':rq2['status']},
            'analysis_method':results['inference'],'reproduction_command':command,
            'cpu_environment':{'python':sys.version.split()[0],'numpy':importlib.metadata.version('numpy')},
            'performance':{'seconds':time.monotonic()-started,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           'free_disk_bytes':shutil.disk_usage(destination).free}}
        integration.write_new_json(destination/'MANIFEST.json',final)
        print(json.dumps({'status':final['status'],'output_root':str(destination),'manifest_sha256':sha256_file(destination/'MANIFEST.json'),
            'counts':results['denominators']['overall'],'performance':final['performance'],'research_api_requests_sent':0}),flush=True)
        return final


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-gate',required=True); parser.add_argument('--destination',required=True)
    parser.add_argument('--code-commit',required=True); parser.add_argument('--plot-python',required=True)
    parser.add_argument('--rq2-audit',required=True)
    args=parser.parse_args(argv)
    try: run(args)
    except (GenerationError,audit.AuditError) as exc:
        print(json.dumps({'status':'STOP','safe_reason':str(exc),'research_api_requests_sent':0}),file=sys.stderr); return 5
    except Exception as exc:
        print(json.dumps({'status':'STOP','error_type':type(exc).__name__,'research_api_requests_sent':0}),file=sys.stderr); return 5
    return 0


if __name__=='__main__':
    raise SystemExit(main())
