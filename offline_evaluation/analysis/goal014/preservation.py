"""One entry and one closure byte audit, reusing fixed Goal013A inventories.

No old guards are changed. New Goal014 products are outside the protected set.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import resource
import subprocess
import time

from ..goal013a import integrity as old
from ..goal013a.run_pipeline import write_json

ROOT = old.ROOT
PREFIXES = ('offline_evaluation/analysis/goal014/', 'reports/server_only/goal014/')


def tracked():
    return [p for p in old.git('ls-files').splitlines() if not p.startswith(PREFIXES)]


def verify_baseline(snapshot):
    old.verify_files(snapshot['inventory'])
    old.require(tracked() == snapshot['tracked_paths'], 'preexisting tracked membership changed')
    old.require(old.git('rev-parse', 'main') == snapshot['main_head'], 'main changed')
    for key, inventory in snapshot['trees'].items():
        old.require(old.tree_inventory(Path(key)) == inventory, 'protected tree changed: '+key)


def before(run):
    manifest = old.read_metadata(ROOT/'reports/server_only/goal013a/MANIFEST.json')
    prior_desc = manifest['integrity']['before_snapshot']
    path = Path(prior_desc['path'])
    old.require(old.descriptor(path) == {k: prior_desc[k] for k in ('bytes', 'sha256')}, 'prior inventory hash mismatch')
    prior = old.load_snapshot(path)
    with old.locks_for(prior['inventory']) as lock_count:
        old.verify_files(prior['inventory'])
        expected_names = [p for p in tracked() if not p.startswith(old.NEW_PREFIXES)]
        old.require(expected_names == prior['tracked_paths'], 'old membership mismatch')
        old.require(old.git('rev-parse', 'main') == prior['main_head'], 'main changed')
        trees = {}
        for roots, key in [('environment_roots', 'environment_inventory'), ('cache_roots', 'cache_inventory')]:
            combined = {}
            for root in prior[roots]:
                trees[root] = old.tree_inventory(Path(root)); combined.update(trees[root])
            old.require(combined == prior[key], 'old '+key+' changed')
        root = prior['goal013_canonical_root']
        trees[root] = old.tree_inventory(Path(root))
        old.require(trees[root] == prior['goal013_canonical_tree'], 'Goal013 canonical changed')
        inventory = dict(prior['inventory'])
        for name in tracked():
            path = str(ROOT/name)
            if path not in inventory:
                inventory[path] = old.descriptor(path)
        # Goal013A now becomes protected, including its dedicated environment.
        for root in (manifest['tools_root'], manifest['run_root']):
            trees[root] = old.tree_inventory(Path(root))
        for name, expected in manifest['report_files'].items():
            old.require(inventory[str(ROOT/'reports/server_only/goal013a'/name)] == expected, 'Goal013A report mismatch')
        for name, expected in manifest['code_files'].items():
            old.require(inventory[str(ROOT/name)] == expected, 'Goal013A code mismatch')
        return {'schema': 'goal014_protected_snapshot_v1', 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'inventory': inventory, 'trees': trees, 'tracked_paths': tracked(),
                'main_head': prior['main_head'], 'entry_head': old.git('rev-parse', 'HEAD'),
                'source_locks': lock_count, 'prior_inventory': prior_desc,
                'excluded_new_run': str(run), 'excluded_new_prefixes': list(PREFIXES)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['before', 'after'])
    parser.add_argument('--run', required=True, type=Path); parser.add_argument('--snapshot-sha256')
    args = parser.parse_args(); started = time.monotonic(); cpu = time.process_time()
    run = args.run.resolve()
    old.require(run.parent == Path('/root/minecraft-runs') and run.name.startswith('goal014-'), 'new Goal014 run required')
    path = run/'PROTECTED_INVENTORY_BEFORE.json'
    if args.mode == 'before':
        snapshot = before(run); write_json(path, snapshot)
    else:
        old.require(old.descriptor(path)['sha256'] == args.snapshot_sha256, 'new snapshot changed')
        snapshot = old.read_metadata(path)
        with old.locks_for(snapshot['inventory']): verify_baseline(snapshot)
    result = {'status': 'PASS', 'phase': args.mode, 'checked_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'snapshot': {'path': str(path), **old.descriptor(path)}, 'protected_files': len(snapshot['inventory']),
              'tracked_files': len(snapshot['tracked_paths']), 'protected_tree_roots': len(snapshot['trees']),
              'source_locks': snapshot['source_locks'], 'all_original_bytes_membership_links_unchanged': True,
              'cpu_seconds': time.process_time()-cpu, 'elapsed_seconds': time.monotonic()-started,
              'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'research_api_requests': 0, 'downloads': 0, 'installs': 0, 'OS_isolation_guaranteed': False}
    write_json(run/('INTEGRITY_'+args.mode.upper()+'.json'), result)
    print(json.dumps(result))


if __name__ == '__main__': main()
