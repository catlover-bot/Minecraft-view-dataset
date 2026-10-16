"""Copy an explicit small aggregate allowlist; retain large pairs in canonical run."""
import argparse
import gzip
import hashlib
from pathlib import Path
import time
import resource

from .. import pipeline as p
from ...goal010 import source_guard as guard
from ....server_v2.checkpoint import scan_file

ALLOWLIST=tuple([f'{i}_{name}_RECEIPT.json' for i,name in enumerate(p.STAGES)]+[
    '0_source/SOURCE_BINDINGS.json','0_source/source_gate.json',
    '1_pilot1/metric_qa_report.json','2_pilot10/metric_qa_report.json',
    '3_full/new_metric_values_960.csv','3_full/new_metric_values_960.json',
    '3_full/baseline_new_metric_values.csv','3_full/baseline_new_metric_values.json',
    '3_full/metric_qa_report.json','3_full/alignment_displacement.csv',
    '3_full/exact_material_confusion.csv','3_full/gt_material_frequency_recall.csv',
    '3_full/material_and_geometry_details.json',
    '4_summary/metric_baselines_paired.csv','4_summary/metric_baselines_paired.json',
    '4_summary/metric_denominators.csv','4_summary/metric_denominators.json',
    '4_summary/reliability_outcomes.csv','4_summary/reliability_outcomes.json',
    '4_summary/metric_rank_reversals.csv','4_summary/metric_rank_reversals.json',
    '4_summary/statistics_metadata.json','4_summary/E_capacity_diagnostic.svg',
    '4_summary/E_capacity_plot_receipt.json'])
MAX_BYTES=10*1024**2
PRODUCTION=('offline_evaluation/analysis/goal010v2/publication/__init__.py',
            'offline_evaluation/analysis/goal010v2/publication/publish.py')


def copy_one(source,destination):
    source=guard.regular(source); scan_file(source)
    data=source.read_bytes(); raw_sha=hashlib.sha256(data).hexdigest()
    encoding='identity'
    if len(data)>MAX_BYTES:
        if source.suffix!='.json': raise ValueError('only oversized JSON aggregate may be losslessly gzip encoded')
        encoded=gzip.compress(data,mtime=0); encoding='gzip'
        if gzip.decompress(encoded)!=data: raise ValueError('gzip roundtrip differs')
        destination=destination.with_name(destination.name+'.gz')
    else: encoded=data
    if len(encoded)>MAX_BYTES: raise ValueError('public aggregate exceeds10MiB; canonical data retained')
    if destination.is_symlink() or any(x.is_symlink() for x in destination.parents): raise ValueError('destination symlink rejected')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as f: f.write(encoded)
    restored=gzip.decompress(destination.read_bytes()) if encoding=='gzip' else destination.read_bytes()
    if restored!=data or p.sha(source)!=raw_sha: raise ValueError('source/copy bytes changed')
    return {'canonical_source':str(source),'source_sha256':raw_sha,'source_bytes':len(data),
            'encoding':encoding,'public_path':str(destination),'public_sha256':p.sha(destination),
            'public_bytes':destination.stat().st_size,'decoded_bytes_exactly_match_source':True}


def publish(run_root,*,copier_commit):
    root=p.safe_root(run_root,4)
    destination=p.REPORT/'analysis'
    if destination.exists() or any(x.is_symlink() for x in (destination,*destination.parents)):
        raise ValueError('fresh nonsymlink publication destination required')
    code=guard.verify_commit(copier_commit,PRODUCTION)
    started=time.monotonic(); cpu=time.process_time()
    with guard.locked_sources() as locks:
        before=guard.load_verified_sources()
        if p.oldpipe.digest(before)!=p.SOURCE_SHA: raise ValueError('source package changed')
        p.verify_old()
        final=guard.read_json(root/'4_summary_RECEIPT.json'); bound=final['binding']
        p.binding(bound['code_commit'],bound['specification_commit'])
        p.previous(root,5,bound)
        descriptors={**final['products'],'4_summary_RECEIPT.json':{'sha256':p.sha(root/'4_summary_RECEIPT.json'),'bytes':(root/'4_summary_RECEIPT.json').stat().st_size}}
        for name in ALLOWLIST:
            if name not in descriptors: raise ValueError('allowlisted output is not bound by final receipt')
            scan_file(guard.safe_child(root,name))
        destination.mkdir(exist_ok=False)
        copied={name:copy_one(guard.safe_child(root,name),destination/name) for name in ALLOWLIST}
        if p.oldpipe.digest(guard.load_verified_sources())!=p.SOURCE_SHA: raise ValueError('source package changed during copy')
        p.verify_old(); p.previous(root,5,bound)
        guard.verify_commit(copier_commit,PRODUCTION)
        receipt={'status':'PUBLIC_COPY_VERIFIED','run_root':str(root),'destination':str(destination),
            'all_decoded_bytes_match_canonical':True,'files':copied,'copied_files':len(copied),
            'retained_only_in_canonical':[name for name in descriptors if name not in ALLOWLIST],
            'copier_commit':copier_commit,'copier_hashes':code,'analysis_binding':bound,
            'source_package_sha256':p.SOURCE_SHA,'source_lock_count':len(locks),
            'canonical_final_receipt_sha256':p.sha(root/'4_summary_RECEIPT.json'),
            'performance':{'cpu_seconds':time.process_time()-cpu,'elapsed_seconds':time.monotonic()-started,
                'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},**p.ZERO}
    p.write_json(destination/'PUBLIC_COPY_RECEIPT.json',receipt)
    return receipt


def main():
    import json
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run-root',required=True)
    parser.add_argument('--copier-commit',required=True); a=parser.parse_args()
    result=publish(a.run_root,copier_commit=a.copier_commit)
    print(json.dumps({k:result[k] for k in ('status','copied_files','performance')}))


if __name__=='__main__': main()
