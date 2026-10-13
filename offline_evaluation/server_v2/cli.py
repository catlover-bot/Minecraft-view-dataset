"""Explicit lifecycle commands. Defaults cannot send research generation requests."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
from .io_utils import ROOT, atomic_json, read_json, writable

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for command in ('prepare','verify-gt','reevaluate','summarize','prepare-builder','dry-run','generate','evaluate'):
        c=sub.add_parser(command); c.add_argument('--run-dir',required=True)
        if command in ('verify-gt','reevaluate'): c.add_argument('--workers',type=int,default=2)
        if command in ('reevaluate','evaluate'): c.add_argument('--alignment-radius',type=int,default=1)
        if command=='evaluate': c.add_argument('--alignment-mode',choices=['bounded','strict_translation'],default='bounded')
        if command=='prepare-builder':
            c.add_argument('--config',required=True);c.add_argument('--split',choices=['smoke','evaluation'],default='smoke')
            c.add_argument('--include-oracles',action='store_true')
            c.add_argument('--builder-name',help='New immutable generation directory name; defaults to builder_SPLIT')
        if command=='generate': c.add_argument('--allow-api',action='store_true',default=False)
    for command in ('prepare-v4','goal002-verify-gt','goal002-reevaluate','prepare-v4-builder',
                    'smoke-proposal','smoke-approval-template','approve-smoke','smoke-dry-run','smoke-generate','summarize-v4','freeze'):
        c=sub.add_parser(command); c.add_argument('--run-dir',required=True)
        if command in ('goal002-verify-gt','goal002-reevaluate'):c.add_argument('--workers',type=int,default=2)
        if command=='prepare-v4-builder':
            c.add_argument('--config',required=True);c.add_argument('--split',choices=['initial_smoke','smoke','evaluation'],required=True)
            c.add_argument('--builder-name',required=True)
        if command=='smoke-proposal':
            c.add_argument('--openai-run-dir',required=True);c.add_argument('--anthropic-run-dir',required=True)
            c.add_argument('--max-tokens',type=int,default=8327680)
        if command=='approve-smoke':
            c.add_argument('--destination',required=True);c.add_argument('--approval',required=True)
        if command=='smoke-approval-template':
            c.add_argument('--enabled-directory',help='Future new canonical directory to bind into the manual approval')
        if command=='smoke-generate':c.add_argument('--allow-api',action='store_true',default=False)
        if command=='summarize-v4':c.add_argument('--smoke',action='store_true')
    c=sub.add_parser('checkpoint');c.add_argument('--run-dir',required=True);c.add_argument('--destination',required=True)
    c.add_argument('--source-allowlist',required=True);c.add_argument('--checkpoint-id',required=True);c.add_argument('--source-reference')
    c=sub.add_parser('restore');c.add_argument('--archive',required=True);c.add_argument('--destination',required=True);c.add_argument('--sha256',required=True)
    a=p.parse_args(argv)
    if a.command=='restore':
        from .checkpoint import restore_archive
        result=restore_archive(a.archive,a.destination,a.sha256)
    elif a.command=='checkpoint':
        from .checkpoint import create_checkpoint
        source=read_json(a.source_allowlist)
        if isinstance(source,dict):source=source['paths']
        result=create_checkpoint(a.destination,a.run_dir,source,
              subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),a.checkpoint_id,
              read_json(a.source_reference) if a.source_reference else None)
    else:
        run=writable(a.run_dir)
        from . import experiment
        if a.command=='prepare-v4':
            from .generation import run_lock
            with run_lock(run):
                from .cohort_v4 import build_v4_cohort,build_inputs,input_counts,protocol
                if (run/'cohort.v4.json').exists():raise FileExistsError('v4 cohort is immutable; choose a new run')
                cohort=build_v4_cohort(budgets={'context_window_tokens':200000,'max_output_tokens':16384})
                atomic_json(run/'cohort.v4.json',cohort)
                counts={}
                for split in ('smoke','evaluation','initial_smoke'):
                    rows=build_inputs(cohort,split)
                    atomic_json(run/f'inputs.{split}.json',rows)
                    counts[split]=input_counts(rows)
                atomic_json(run/'protocol.json',protocol())
                result={'cohort_sha256':cohort['cohort_sha256'],'cohort_status':cohort['status'],'split_counts':cohort['split_counts'],
                        'inputs_per_builder':counts,'research_requests_sent':0}
                atomic_json(run/'v4_preparation.json',result)
        elif a.command in ('goal002-verify-gt','goal002-reevaluate'):
            from .goal002 import verify_strict_controls,diagnose_legacy
            from .generation import run_lock
            with run_lock(run):
                result=(verify_strict_controls if a.command=='goal002-verify-gt' else diagnose_legacy)(run,a.workers)
        elif a.command=='freeze':
            from .checkpoint import freeze_completed
            result=freeze_completed(run)
        elif a.command=='prepare-v4-builder':
            from .generation import run_lock
            with run_lock(run):
                from .generation import GenerationConfig,prepare_generation,object_hash
                from .io_utils import sha256
                rows=read_json(run/f'inputs.{a.split}.json')
                config=GenerationConfig.from_dict(read_json(a.config))
                name=a.builder_name
                if Path(name).name!=name or name in ('.','..'):raise ValueError('Builder name must be a relative directory name')
                prepared=[row for row in rows if row['status']=='prepared']
                manifest=prepare_generation(prepared,config,run/name,
                    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
                planning={'split':a.split,'planned_inputs':len(rows),'prepared_inputs':len(prepared),
                          'planned_inputs_sha256':sha256(run/f'inputs.{a.split}.json'),
                          'cohort_sha256':read_json(run/'cohort.v4.json')['cohort_sha256'],
                          'infeasible_inputs':[row for row in rows if row['status']!='prepared'],
                          'generation_identity_hash':object_hash(manifest['identity'])}
                if (run/name/'selection.json').exists():
                    if read_json(run/name/'selection.json')!=planning:raise ValueError('Builder selection changed')
                else:atomic_json(run/name/'selection.json',planning)
                result=planning
        elif a.command in ('smoke-proposal','smoke-approval-template','approve-smoke','smoke-dry-run','smoke-generate'):
            from .smoke import prepare_smoke_campaign,approval_template,approve_smoke_campaign,dry_run_smoke_campaign,run_smoke_campaign
            if a.command=='smoke-proposal':result=prepare_smoke_campaign({'openai':a.openai_run_dir,'anthropic':a.anthropic_run_dir},run,max_tokens=a.max_tokens)
            elif a.command=='smoke-approval-template':result=approval_template(run,enabled_directory=a.enabled_directory)
            elif a.command=='approve-smoke':result=approve_smoke_campaign(run,a.destination,read_json(a.approval))
            elif a.command=='smoke-dry-run':result=dry_run_smoke_campaign(run)
            else:result=run_smoke_campaign(run,permission=a.allow_api)
        elif a.command=='summarize-v4':
            from .analysis_v4 import collect_planned_outcomes,summarize_v4
            from .generation import object_hash,run_lock
            with run_lock(run):
                expected=None
                if (run/'campaign.json').is_file():
                    if not a.smoke:raise ValueError('Smoke campaigns require --smoke; no research inference')
                    campaign=read_json(run/'campaign.json')
                    if campaign.get('identity_hash')!=object_hash(campaign['identity']):raise ValueError('Campaign identity hash mismatch')
                    directories=[run/r['path'] for r in campaign['identity']['runs'].values()]
                    expected={provider:r['identity_hash'] for provider,r in campaign['identity']['runs'].items()}
                else:
                    directories=[directory for directory in sorted(run.glob('builder_*'))
                                 if (directory/'selection.json').is_file() and
                                 read_json(directory/'selection.json')['split']==('initial_smoke' if a.smoke else 'evaluation')]
                rows=collect_planned_outcomes(directories,expected_identities=expected)
                result=summarize_v4(rows,smoke=a.smoke)
                atomic_json(run/('summary.v4.smoke.json' if a.smoke else 'summary.v4.json'),result)
        elif a.command=='prepare': result=experiment.prepare(run)
        elif a.command=='verify-gt':
            result=experiment.verify_gt(run,a.workers)
            if (run/'subset.json').is_file():result['oracle_inputs']=experiment.oracle_inputs(run)
        elif a.command=='reevaluate':result=experiment.reevaluate(run,a.alignment_radius,a.workers)
        elif a.command=='summarize':result=experiment.summarize(run)
        else:
            from .generation import GenerationConfig,prepare_generation,dry_run_generation,generate,evaluate_generation
            if a.command=='prepare-builder':
                rows=read_json(run/'inputs.json')
                if a.include_oracles:rows+=read_json(run/'oracle_inputs.json')
                rows=[r for r in rows if r['split']==a.split and r['status']=='prepared']
                config=GenerationConfig.from_dict(read_json(a.config))
                name=a.builder_name or f'builder_{a.split}'
                if Path(name).name != name or name in ('.','..'):
                    raise ValueError('Builder name must be a single relative directory name')
                result=prepare_generation(rows,config,run/name,
                      commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
                result={'records':len(rows),'split':a.split,'api_status':'NOT_RUN','requests_sent':0}
            elif a.command=='dry-run':result=dry_run_generation(run)
            elif a.command=='generate':
                from .generation import GenerationError
                try:
                    result=generate(run,permission=a.allow_api)
                except GenerationError as exc:
                    print(json.dumps({'status':'BLOCKED','reason':str(exc),
                                      'requests_sent_by_invocation':0 if str(exc) in ('api_disabled','explicit_api_permission_required') else None}))
                    return 2
            else:
                from .data import load_scene
                gt={scene.scene_id:scene.grid for scene in map(load_scene,experiment._scenes())}
                rows=evaluate_generation(run,gt,alignment_radius=a.alignment_radius,alignment_mode=a.alignment_mode)
                result={'evaluation_rows':len(rows),'generation_requests_sent_by_evaluation':0}
    print(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2))

if __name__=='__main__':raise SystemExit(main())
