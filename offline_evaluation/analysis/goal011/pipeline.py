"""Ordered CPU-only Goal011 audits. This CLI has no research API execution mode."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import resource
import shutil
import time

from . import source_guard as guard
from ..goal010 import pipeline as old
from ..goal010v2.pipeline import sample_budget, compact_json

STAGES = ('source', 'tradeoff', 'material', 'rest', 'prospective')
ZERO = dict(research_api_requests_sent=0, model_queries=0, token_count_api_requests=0,
            api_key_queries=0, human_studies=0, new_generation=0, retries=0,
            checkpoint_uploads_or_mutations=0, existing_draft_mutations=0)
write_json = old.write_json
write_csv = old.write_csv
write_text = old.write_text


def now():
    return datetime.now(timezone.utc).isoformat()


def safe_root(value, stage):
    path = Path(value)
    if (not path.is_absolute() or path.parent != Path('/root/minecraft-runs')
            or not re.fullmatch(r'goal011-\d{8}T\d{6}Z(?:-[a-z0-9-]+)?', path.name)
            or any(p.is_symlink() for p in (path, *path.parents))):
        raise ValueError('fresh direct Goal011 UTC run path required')
    if stage == 0 and path.exists():
        raise FileExistsError('existing run must never be overwritten')
    if stage and not path.is_dir():
        raise ValueError('prior source stage required')
    if not path.parent.is_dir() or shutil.disk_usage(path.parent).free < 10*1024**3:
        raise ValueError('at least10 GiB free in existing run parent required')
    return path


def previous(root, stage, bound):
    results = {}
    for i in range(stage):
        path = root/f'{i}_{STAGES[i]}_RECEIPT.json'
        receipt = guard.read_json(path)
        if (receipt.get('status') != 'PASS' or receipt.get('stage') != i
                or receipt.get('binding') != bound or receipt.get('run_root') != str(root)
                or receipt.get('source_package_sha256') != guard.SOURCE_SHA):
            raise ValueError('prior stage incomplete or differently bound')
        guard.inherited.verify_products(root, receipt['products'])
        results[path.name] = guard.sha(path)
    return results


def stage_work(stage, root, out, package, old_run, v2_run, gate, proof, inventory):
    if stage == 0:
        write_json(out/'SOURCE_BINDINGS.json', proof)
        write_json(out/'PROTECTED_INVENTORY_BEFORE.json', inventory)
        write_json(out/'SOURCE_GATE.json', gate)
        return {'status':'PASS', 'protected_files':len(inventory)}
    if stage == 1:
        from .tradeoff import analyze_tradeoff
        result = analyze_tradeoff(guard.tradeoff_inputs(package, old_run, v2_run),
                                 gate=gate, plan_sha256=gate['plan_sha256'])
        write_json(out/'RESULT.json', result)
        write_json(out/'CLAIM_AUDIT_FINDINGS.json', result['claim_audit'])
        write_text(out/'TRADEOFF_EXPLANATION_JA.md', result['report_markdown'])
        for key in ('global_saved_effects', 'quintile_effects', 'valid_mask_audit',
                    'building_deltas', 'precision_design_inputs'):
            write_csv(out/(key+'.csv'), result[key])
        return {'status':'PASS', 'counts':{k:len(v) for k,v in result.items() if isinstance(v,list)}}
    if stage == 2:
        from .material_batch import run_material
        controls = old_run/'phase3_exact_oracle/E_reference_controls.json'
        data = {**package, 'goal007_manifest_path':str(guard.inherited.audit8.GOAL007/'MANIFEST.json'),
                'old_e_controls_descriptor':{'path':str(controls), 'sha256':guard.sha(controls), 'bytes':controls.stat().st_size}}
        result = run_material(data, gate=gate)
        for filename, key in (('E_CASES_101.csv','cases'), ('E_GLOBAL_CONFUSION.csv','global_confusion'),
                              ('E_MAPPING_UPPER_BOUNDS.csv','mapping')):
            write_csv(out/filename, result[key])
        compact_json(out/'E_FAILURE_MECHANISM_AUDIT.json', result['audit'])
        write_json(out/'MATERIAL_SOURCE_INVENTORY.json', result['source_inventory'])
        write_text(out/'E_SYMBOL_BINDING_FINDINGS_JA.md', result['report_markdown'])
        if result['audit'].get('status') != 'PASS' or len(result['cases']) != 101:
            raise ValueError('all101 material cases and passing audit required; partial evidence retained')
        return {'status':'PASS', 'cases':len(result['cases']), 'audit_status':result['audit'].get('status')}
    if stage == 3:
        from .rest import audit_rest30, verify_inventory
        data = {**package, 'legacy_analysis_manifests':[
            {'canonical_run_root':str(old_run)}, {'canonical_run_root':str(v2_run)}]}
        result = audit_rest30(data, gate=gate)
        verify_inventory(result['inventory'])
        compact_json(out/'REST30_FEASIBILITY.json', result['feasibility'])
        write_csv(out/'REST30_SOURCES.csv', result['sources'])
        write_json(out/'REST30_SOURCES.json', result['sources'])
        write_json(out/'REST_SOURCE_INVENTORY.json', result['inventory'])
        return {'status':'PASS', 'source_rows':len(result['sources']),
                'source_file_count':len(result['inventory']), 'source_inventory_sha256':old.digest(result['inventory'])}
    from .prospective import design_protocol
    tradeoff = guard.read_json(root/'1_tradeoff/RESULT.json')
    rest = {'feasibility':guard.read_json(root/'3_rest/REST30_FEASIBILITY.json'),
            'sources':guard.read_json(root/'3_rest/REST30_SOURCES.json')}
    result = design_protocol(package, tradeoff, rest, gate=gate)
    write_json(out/'PROSPECTIVE_DESIGN.json', result['protocol'])
    write_text(out/'PROSPECTIVE_PROTOCOL_UNAPPROVED.md', result['report_markdown'])
    return {'status':'PASS', 'design_status':result['protocol']['status']}


def run(stage, run_root, code_commit, specification_commit):
    if stage not in range(len(STAGES)):
        raise ValueError('stage0..4 required')
    root = safe_root(run_root, stage)
    bound = guard.binding(code_commit, specification_commit)
    prior = previous(root, stage, bound)
    out = root/f'{stage}_{STAGES[stage]}'
    if out.exists() or (root/f'{stage}_STARTED.json').exists():
        raise FileExistsError('retain partial stage; diagnose before any new run')
    started = time.monotonic(); cpu = time.process_time(); created = False
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    try:
        with sample_budget(1800, 3600), guard.inherited.locked_sources() as locks:
            package, old_run, v2_run, inventory, proof = guard.load_sources()
            source_gate = guard.gate(bound, proof)
            if stage:
                before = guard.read_json(root/'0_source/PROTECTED_INVENTORY_BEFORE.json')
                if before != inventory:
                    raise ValueError('protected inventory changed since initial stage')
            else:
                root.mkdir(exist_ok=False)
            write_json(root/f'{stage}_STARTED.json', {'stage':stage, 'binding':bound, 'at_utc':now()})
            created = True; out.mkdir(exist_ok=False)
            result = stage_work(stage, root, out, package, old_run, v2_run, source_gate, proof, inventory)
            if not isinstance(result, dict) or result.get('status') != 'PASS':
                raise ValueError('stage did not return a passing completion proof')
            guard.verify_inventory(inventory)
            # Inherited audit also verifies file membership, implementation hashes
            # and original immutable directory trees, not just individual bytes.
            after_package = guard.inherited.load_verified_sources()
            if old.digest(after_package) != guard.SOURCE_SHA or guard.binding(code_commit, specification_commit) != bound:
                raise ValueError('source package or code changed during stage')
            previous(root, stage, bound)
            write_json(out/'PROTECTED_AFTER.json', {'status':'PASS', 'files':len(inventory),
                'before_sha256':old.digest(inventory), 'after_sha256':old.digest(inventory),
                'byte_checks_performed':True, 'original_tree_membership_rechecked':True})
            receipt = {'status':'PASS', 'stage':stage, 'run_root':str(root), 'binding':bound,
                'previous':prior, 'source_package_sha256':guard.SOURCE_SHA,
                'source_before_after_unchanged':True, 'source_lock_count':len(locks),
                'python_network_connections_disabled':True, 'OS_level_isolation_guaranteed':False,
                'result':result, 'products':old.products(root), 'created_at_utc':now(),
                'performance':{'elapsed_seconds':time.monotonic()-started, 'cpu_seconds':time.process_time()-cpu,
                               'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                               'free_disk_bytes':shutil.disk_usage(root).free}, **ZERO}
        write_json(root/f'{stage}_{STAGES[stage]}_RECEIPT.json', receipt)
        print(json.dumps({k:receipt[k] for k in ('status','stage','result','performance')}), flush=True)
        return receipt
    except Exception as exc:
        if created:
            failure = {'status':'STOP', 'error_type':type(exc).__name__, 'reason':str(exc),
                       'historical_mutations':None,
                       'historical_integrity_status':'NOT_VERIFIED_STAGE_FAILED',
                       'historical_write_operations_by_pipeline':0, **ZERO}
            if hasattr(exc, 'report'): failure['critical_evidence'] = exc.report
            write_json(root/f'{stage}_STOP.json', failure)
        raise


def publish(run_root, code_commit, specification_commit):
    """Copy only reviewed small metadata products; never raw responses or voxels."""
    root = safe_root(run_root, 5); bound = guard.binding(code_commit, specification_commit)
    receipts = previous(root, 5, bound)
    names = {
        '1_tradeoff':('CLAIM_AUDIT_FINDINGS.json', 'TRADEOFF_EXPLANATION_JA.md', 'global_saved_effects.csv',
                      'quintile_effects.csv', 'valid_mask_audit.csv', 'precision_design_inputs.csv'),
        '2_material':('E_CASES_101.csv','E_GLOBAL_CONFUSION.csv','E_MAPPING_UPPER_BOUNDS.csv',
                      'E_FAILURE_MECHANISM_AUDIT.json','E_SYMBOL_BINDING_FINDINGS_JA.md'),
        '3_rest':('REST30_FEASIBILITY.json','REST30_SOURCES.csv'),
        '4_prospective':('PROSPECTIVE_DESIGN.json','PROSPECTIVE_PROTOCOL_UNAPPROVED.md')}
    sources = [(root/folder/name, guard.REPORT/name) for folder, group in names.items() for name in group]
    sources += [(root/name, guard.REPORT/'receipts'/name) for name in receipts]
    sources += [(root/f'{i}_{s}'/'PROTECTED_AFTER.json', guard.REPORT/'receipts'/f'{i}_PROTECTED_AFTER.json')
                for i,s in enumerate(STAGES)]
    for source, dest in sources:
        guard.inherited.regular(source)
        if source.stat().st_size > 10*1024**2 or dest.exists() or any(p.is_symlink() for p in (dest,*dest.parents)):
            raise ValueError('public copy must be new regular metadata under10MiB')
    copied = {}
    for source, dest in sources:
        dest.parent.mkdir(exist_ok=True)
        with dest.open('xb') as f: f.write(source.read_bytes())
        if guard.sha(source) != guard.sha(dest): raise ValueError('public copy differs')
        copied[dest.relative_to(guard.REPORT).as_posix()] = {
            'canonical_source':str(source), 'sha256':guard.sha(source), 'bytes':source.stat().st_size}
    write_json(guard.REPORT/'PUBLIC_COPY_RECEIPT.json', {'status':'PASS', 'run_root':str(root),
               'binding':bound, 'files':copied, 'raw_responses_copied':0, 'GT_arrays_copied':0, **ZERO})
    print(json.dumps({'status':'PASS','copied_files':len(copied),'run_root':str(root)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--stage',type=int); group.add_argument('--publish',action='store_true')
    parser.add_argument('--run-root',required=True); parser.add_argument('--code-commit',required=True)
    parser.add_argument('--specification-commit',required=True)
    args = parser.parse_args()
    if args.publish: publish(args.run_root,args.code_commit,args.specification_commit)
    else: run(args.stage,args.run_root,args.code_commit,args.specification_commit)


if __name__ == '__main__': main()
