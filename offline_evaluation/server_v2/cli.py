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
        if command=='prepare-builder':
            c.add_argument('--config',required=True);c.add_argument('--split',choices=['smoke','evaluation'],default='smoke')
            c.add_argument('--include-oracles',action='store_true')
            c.add_argument('--builder-name',help='New immutable generation directory name; defaults to builder_SPLIT')
        if command=='generate': c.add_argument('--allow-api',action='store_true',default=False)
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
        if a.command=='prepare': result=experiment.prepare(run)
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
                rows=evaluate_generation(run,gt,alignment_radius=a.alignment_radius)
                result={'evaluation_rows':len(rows),'generation_requests_sent_by_evaluation':0}
    print(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2))

if __name__=='__main__':raise SystemExit(main())
