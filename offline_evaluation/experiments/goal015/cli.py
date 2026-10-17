"""Lazy command router: status/evaluation/save never import the API adapter."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description='Goal015 paired80 preparation and separately approved execution')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('preview','status','start','execute','resume','checkpoint-local'):
        p=sub.add_parser(name);p.add_argument('--plan',type=Path,required=True)
        if name in ('start','execute','resume'):p.add_argument('--pricing-evidence',type=Path)
    for name in ('evaluate','aggregate'):
        p=sub.add_parser(name);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output-dir',type=Path)
        if name=='aggregate':p.add_argument('--evaluation',type=Path,required=True)
    p=sub.add_parser('checkpoint-upload');p.add_argument('--checkpoint-dir',type=Path,required=True)
    p.add_argument('--verification-dir',type=Path,required=True);p.add_argument('--release-id',type=int)
    p.add_argument('--allow-create',action='store_true')
    p=sub.add_parser('capture-pricing');p.add_argument('--plan',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p=sub.add_parser('seal-pricing');p.add_argument('--plan',type=Path,required=True);p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--reviewed-contract',type=Path,required=True);p.add_argument('--review-note',required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command in ('preview','status','start','execute','resume'):
        from . import runtime
        if args.command in ('preview','status'):result=getattr(runtime,args.command)(args.plan)
        else:
            evidence=runtime.descriptor(args.pricing_evidence) if args.pricing_evidence else None
            result=runtime.execute(args.plan,resume=args.command=='resume',refresh_evidence=evidence)
    elif args.command=='evaluate':
        from .evaluate import evaluate_plan
        result=evaluate_plan(args.plan,args.output_dir)
    elif args.command=='aggregate':
        from . import runtime
        from .evaluate import aggregate_rows
        plan=runtime.verify_plan(args.plan);data=runtime.read_json(args.evaluation)
        runtime.require(data.get('plan_binding',{}).get('plan_id')==plan['plan_id'],'evaluation belongs to another plan')
        result=aggregate_rows(plan['schedule'],data['rows'],fixture=plan.get('fixture_only') is True)
        if args.output_dir:
            target=runtime.path_safe(args.output_dir)
            runtime.require(target.is_relative_to(Path(plan['canonical_root'])),'aggregate output must stay in canonical root')
            target.mkdir(parents=True,exist_ok=False);runtime._save(target/'summary.json',result,exclusive=True)
    elif args.command=='checkpoint-local':
        from .checkpoint import create_save_only
        result=create_save_only(args.plan)
    elif args.command=='checkpoint-upload':
        from .checkpoint import upload_checkpoint
        result=upload_checkpoint(args.checkpoint_dir,args.verification_dir,release_id=args.release_id,allow_create=args.allow_create)
    elif args.command=='capture-pricing':
        from .pricing import capture
        result=capture(args.plan,args.output_dir)
    else:
        from .pricing import seal
        result=seal(args.plan,args.capture,args.reviewed_contract,args.review_note,args.output)
    if args.command not in ('start','execute','resume','status','preview'):
        result.setdefault('research_api_requests_sent',0)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True,allow_nan=False))
    return result


if __name__=='__main__':main()
