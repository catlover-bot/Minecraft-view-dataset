"""Exclusive disabled-plan assembly. Never creates execution authority/state."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
from .planning import descriptor,write_json_new,validate_schedule,SELECTION_COMMIT
from .runtime import identity,require

REPO=Path(__file__).resolve().parents[3]
CODE_ROOTS=('offline_evaluation/experiments/goal015','offline_evaluation/server_v2',
            'offline_evaluation/resmoke_v2','offline_evaluation/operations/goal006a',
            'offline_evaluation/operations/goal006c','offline_evaluation/analysis/goal010v2')


def freeze_plan(run):
    run=Path(run).absolute();report=REPO/'reports/server_only/goal015'
    require(run.parent==Path('/root/minecraft-runs') and run.name.startswith('goal015-'),'new canonical Goal015 run required')
    require(not any((run/name).exists() for name in ('plan.json','enabled','authority','execution.lock')),'do not reset an existing plan/execution')
    require(subprocess.run(['git','merge-base','--is-ancestor',SELECTION_COMMIT,'HEAD'],cwd=REPO).returncode==0,'selection rule was not committed')
    rows=json.loads((run/'schedule_80.json').read_text());validate_schedule(rows)
    inputs=json.loads((run/'conditions_20_manifest.json').read_text())
    require(len(inputs)==20,'exact20 conditions required')
    for row in rows:require(descriptor(row['payload']['path'])==row['payload'],'private payload changed')
    for row in inputs:require(descriptor(row['input']['path'])==row['input'],'private input changed')
    implementations={}
    for folder in CODE_ROOTS:
        for path in sorted((REPO/folder).rglob('*.py')):
            if 'tests' not in path.parts:implementations[str(path)]=descriptor(path)['sha256']
    prompt=REPO/'offline_evaluation/resmoke_v2/prompts/builder.txt'
    implementations[str(prompt)]=descriptor(prompt)['sha256']
    evidence=json.loads((run/'PRICING_EVIDENCE.json').read_text())
    plan={'schema':'goal015_disabled_plan_v1','canonical_root':str(run),'api_allowed':False,'budget_approved':False,
          'status':'PILOT80_READY_AWAITING_EXPLICIT_APPROVAL','research_api_requests_sent':0,
          'model_account_availability':'NOT_RUN','proposed_budget_usd':'5.00','planned_calls':80,'planned_buildings':10,
          'concurrency':1,'max_retries':0,'timeout_seconds':600,'pricing_evidence':descriptor(run/'PRICING_EVIDENCE.json'),
          'pricing_contract':evidence['pricing_contract'],'implementation_hashes':implementations,
          'specification':descriptor(report/'EXPERIMENT_SPEC.md'),'selection_rule_commit':SELECTION_COMMIT,
          'selection_rule':descriptor(report/'SELECTION_RULE.md'),'schedule':rows,
          'source_bindings':descriptor(run/'source_binding_10.json'),'conditions':descriptor(run/'conditions_20_manifest.json'),
          'schedule_artifact':descriptor(run/'schedule_80.json'),'source_provenance':descriptor(run/'SOURCE_PROVENANCE.json'),
          'cost_estimate':descriptor(run/'COST_ESTIMATE.json'),
          'gt_sources':json.loads((run/'GT_DESCRIPTORS.json').read_text()),
          'evaluation':{'primary_metrics':['aligned_occupancy_iou','aligned_surface_f1_at_1'],
                        'policies':['completed_attempt_utility','common_valid_conditional_accuracy'],
                        'cluster_frame':'planned10','bootstrap_draws':2000,'bootstrap_seed':20261017,'minimum_finite_draws':1900},
          'existing_execution_environment':{'python':str(REPO/'.venv-server-v2-20261013/bin/python'),
                    'distributions':{name:importlib.metadata.version(name) for name in ('numpy','scipy','pytest')}},
          'historical_scope_modified':False,'old_budget_accounting_reference_usd':'21.56083625'}
    plan['plan_id']=identity(plan);desc=write_json_new(run/'plan.json',plan)
    write_json_new(report/'plan.json',plan)
    binding={'schema':'goal015_prepared_execution_binding_v1','plan_id':plan['plan_id'],'canonical_root':str(run),
             'plan':desc,'public_plan':descriptor(report/'plan.json'),'specification':plan['specification'],
             'implementation_files':len(implementations),'selection_rule_commit':SELECTION_COMMIT,
             'api_allowed':False,'budget_approved':False,'research_api_requests_sent':0}
    write_json_new(report/'EXECUTION_BINDING.json',binding);write_json_new(run/'EXECUTION_BINDING.json',binding)
    return binding


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    print(json.dumps(freeze_plan(parser.parse_args().run)))
