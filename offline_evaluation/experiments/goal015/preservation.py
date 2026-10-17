"""Extend a freshly verified retained inventory; one full closure verification."""
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time

from ...analysis.goal013a import integrity as old
from .planning import descriptor, write_json_new

PREFIXES=('offline_evaluation/experiments/goal015/','reports/server_only/goal015/')
PRIOR=Path('/root/minecraft-runs/goal014-20261017T145709Z/PROTECTED_INVENTORY_BEFORE.json')
PRIOR_SHA='63d309acc4924a1dbc8ddf9502a16b9e598b10dcabae3904518b2a1b6130e409'


def tracked():
    return [p for p in old.git('ls-files').splitlines() if not p.startswith(PREFIXES)]


def extend_verified_baseline(run):
    """The inherited check is actual preceding tool evidence, not a new claim."""
    run=Path(run);receipt=json.loads((run/'INHERITED_BASELINE_VERIFICATION.json').read_text())
    old.require(receipt['status']=='PASS','verified inherited baseline required')
    old.require(descriptor(PRIOR)['sha256']==PRIOR_SHA,'retained inventory changed')
    prior=json.loads(PRIOR.read_text());inventory=dict(prior['inventory']);trees=dict(prior['trees'])
    names=tracked();old.require(set(prior['tracked_paths'])<=set(names),'old tracked paths missing')
    for name in names:
        absolute=str(old.ROOT/name)
        if absolute not in inventory:inventory[absolute]=old.descriptor(absolute)
    manifest=json.loads((old.ROOT/'reports/server_only/goal014/MANIFEST.json').read_text())
    for field,base in [('report_files',old.ROOT/'reports/server_only/goal014'),('code_files',old.ROOT)]:
        for name,expected in manifest[field].items():
            old.require(old.descriptor(base/name)==expected,'Goal014 product changed')
    root=json.loads((old.ROOT/'reports/server_only/goal014/CONFIG.json').read_text())['run_root']
    trees[root]=old.tree_inventory(Path(root))
    result={'schema':'goal015_protected_snapshot_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'inventory':inventory,'trees':trees,'tracked_paths':names,'main_head':prior['main_head'],
            'prior':descriptor(PRIOR),'inherited_baseline_verification':descriptor(run/'INHERITED_BASELINE_VERIFICATION.json'),
            'excluded_prefixes':list(PREFIXES),'excluded_root':str(run)}
    desc=write_json_new(run/'PROTECTED_INVENTORY_BEFORE.json',result)
    receipt={'status':'PASS','phase':'before','snapshot':desc,'protected_files':len(inventory),
             'protected_tree_roots':len(trees),'tracked_files':len(names),'main':old.git('rev-parse','main'),
             'inherited_check_reused':True,'inherited_verification':result['inherited_baseline_verification'],
             'newly_protected_goal014_tree':root,'research_api_requests_sent':0}
    write_json_new(run/'INTEGRITY_BEFORE.json',receipt)
    return receipt


def verify_after(run):
    run=Path(run);start=time.monotonic();cpu=time.process_time()
    before=json.loads((run/'INTEGRITY_BEFORE.json').read_text());path=run/'PROTECTED_INVENTORY_BEFORE.json'
    old.require(descriptor(path)==before['snapshot'],'Goal015 baseline changed')
    snapshot=json.loads(path.read_text())
    with old.locks_for(snapshot['inventory']) as lock_count:
        old.verify_files(snapshot['inventory'])
        for root,expected in snapshot['trees'].items():
            old.require(old.tree_inventory(Path(root))==expected,'protected tree changed: '+root)
        old.require(tracked()==snapshot['tracked_paths'],'old tracked membership changed')
        old.require(old.git('rev-parse','main')==snapshot['main_head'],'main changed')
    result={'status':'PASS','phase':'after','checked_at_utc':datetime.now(timezone.utc).isoformat(),
            'snapshot':descriptor(path),'all_protected_bytes_membership_links_unchanged':True,
            'protected_files':len(snapshot['inventory']),'protected_tree_roots':len(snapshot['trees']),
            'tracked_files':len(snapshot['tracked_paths']),'source_locks':lock_count,
            'main_head':snapshot['main_head'],'cpu_seconds':time.process_time()-cpu,
            'elapsed_seconds':time.monotonic()-start,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'research_api_requests_sent':0,'old_environments_modified':False}
    write_json_new(run/'INTEGRITY_AFTER.json',result)
    return result
