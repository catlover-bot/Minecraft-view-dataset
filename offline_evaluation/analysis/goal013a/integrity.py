"""Goal013A read-only protection of preexisting research and environments.

GT/outcome files are streamed only for bytes/SHA, never parsed. New Goal013A
tools, runs, code and reports are excluded from the immutable source set.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import stat
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT/'reports/server_only/goal013'
ENTRY_HEAD = '939503d5d6e55292c7615628ffa4116d22124418'
MAIN_HEAD = '7309327efd1c19dda793b19e8d8e6ce0b5868db9'
SNAPSHOT_SHA256 = '80208b14252de996300129275a066347a3b2796e9a6a7f232e904f373bdf4d0e'
PRIOR = Path('/root/minecraft-runs/goal013-20261017T111808Z/PROTECTED_INVENTORY_BEFORE.json')
NEW_PREFIXES = ('offline_evaluation/analysis/goal013a/', 'reports/server_only/goal013a/')
FORBIDDEN_PARTS = {'.git', '.aws', '.ssh', '.codex', '.agents', 'Microsoft', 'copilot'}


class IntegrityError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def safe(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'absolute path without traversal required')
    require(not any(p in FORBIDDEN_PARTS or (p.startswith('.env') and p != '.env.example') for p in path.parts), 'excluded path')
    require(not any(p.is_symlink() for p in path.parents), 'symlink ancestor refused')
    return path


def descriptor(path):
    """File bytes or link target; do not decode source contents."""
    path = safe(path)
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode):
        return {'symlink_target': os.readlink(path)}
    require(stat.S_ISREG(before.st_mode), 'regular file required: '+str(path))
    h = hashlib.sha256(); size = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block); size += len(block)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) and size == before.st_size,
            'source changed while hashing: '+str(path))
    return {'bytes': size, 'sha256': h.hexdigest()}


def tree_inventory(root):
    """Include missing roots, empty directories and symlinks without following them."""
    root = safe(root)
    if not root.exists() and not root.is_symlink():
        return {str(root): {'kind': 'absent'}}
    if root.is_symlink() or root.is_file():
        return {str(root): descriptor(root)}
    result = {str(root): {'kind': 'directory'}}
    def fail(error):
        raise error
    for parent, dirs, files in os.walk(root, followlinks=False, onerror=fail):
        dirs.sort(); files.sort()
        for name in dirs + files:
            p = safe(Path(parent)/name)
            if p.is_symlink() or p.is_file():
                result[str(p)] = descriptor(p)
            else:
                require(p.is_dir(), 'unexpected source object: '+str(p))
                result[str(p)] = {'kind': 'directory'}
    return result


def verify_files(inventory):
    for name, expected in inventory.items():
        require(descriptor(Path(name)) == expected, 'protected source changed: '+name)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode().strip()


def tracked():
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')[:-1]
    return [name for name in names if not name.startswith(NEW_PREFIXES)]


def read_metadata(path):
    path = safe(path)
    require(path.stat().st_size <= 32*1024**2, 'metadata byte limit')
    return json.loads(path.read_text(encoding='utf-8'))


def verify_goal013():
    """Resolve canonical paths from the committed copy receipt, not filenames."""
    manifest_path = REPORT/'MANIFEST.json'
    committed = subprocess.check_output(['git', 'show', ENTRY_HEAD+':reports/server_only/goal013/MANIFEST.json'], cwd=ROOT)
    require(descriptor(manifest_path)['sha256'] == hashlib.sha256(committed).hexdigest(), 'Goal013 manifest differs from entry commit')
    manifest = read_metadata(manifest_path)
    inventory = {str(manifest_path): descriptor(manifest_path)}
    for table, base in [('report_files', REPORT), ('code_files', ROOT)]:
        for name, expected in manifest[table].items():
            path = safe(base/name)
            require(path.is_relative_to(base), 'manifest path escaped role')
            actual = descriptor(path)
            require(actual == expected, 'Goal013 committed product changed: '+str(path))
            inventory[str(path)] = actual
    copy = read_metadata(REPORT/'PUBLIC_COPY_RECEIPT.json')
    require(copy['status'] == 'PASS' and copy['run_root'] == manifest['binding']['run_root'], 'canonical Goal013 root differs')
    canonical = safe(copy['run_root'])
    paths = {}
    for name, expected in copy['files'].items():
        path = safe(expected['canonical_source'])
        require(path.is_relative_to(canonical), 'canonical product escaped Goal013 run')
        actual = descriptor(path)
        require(actual == {k: expected[k] for k in ('bytes', 'sha256')} == descriptor(REPORT/name), 'canonical/public Goal013 mismatch')
        paths[name] = str(path); inventory[str(path)] = actual
    required = {'SOURCE_BINDING_180.csv', 'SOURCE_BINDING_180.json', 'SOURCE_BINDING_RECEIPT.json', 'PREFLIGHT_RECEIPT.json'}
    require(required <= set(paths), 'missing canonical Goal013 binding')
    coverage = read_metadata(REPORT/'SOURCE_COVERAGE.json')
    preflight = read_metadata(paths['PREFLIGHT_RECEIPT.json'])
    source_receipt = read_metadata(paths['SOURCE_BINDING_RECEIPT.json'])
    require(coverage['bound_descriptions'] == 180 and coverage['independent_source_buildings'] == 60
            and coverage['nonempty_summaries'] == 178 and coverage['unique_selected_image_files'] == 360
            and coverage['request_image_references'] == 1080, 'Goal013 frozen source coverage differs')
    require([(r['scene_id'], r['description_provider']) for r in coverage['missing_summaries']] ==
            [('buildings_100_v4_building_054', 'gemini'), ('buildings_100_v4_building_061', 'gemini')], 'Goal013 missing-summary keys differ')
    require(preflight['scored_descriptions'] == 0 and preflight['actual_association_N'] is None
            and preflight['source_binding_status'] == 'PASS' and preflight['phase1_executed'] is False
            and preflight['phase2_executed'] is False, 'Goal013 blocked result changed')
    require(source_receipt['status'] == 'PASS' and source_receipt['source_before_after_unchanged'] is True
            and source_receipt['source_inventory_sha256'] == coverage['source_inventory_sha256'], 'Goal013 source receipt differs')
    for name, expected in source_receipt['source_inventory'].items():
        actual = descriptor(Path(name))
        require(actual == expected, 'Goal013 bound source changed: '+name)
        require(name not in inventory or inventory[name] == actual, 'Goal013 source inventories disagree')
        inventory[name] = actual
    require(manifest['binding']['preflight_receipt_sha256'] == inventory[paths['PREFLIGHT_RECEIPT.json']]['sha256']
            and manifest['binding']['scoring_protocol_sha256'] == inventory[str(REPORT/'SCORING_PROTOCOL.md')]['sha256']
            and manifest['binding']['mechanical_selection_rule_sha256'] == inventory[str(REPORT/'MODEL_SELECTION_RULE.json')]['sha256'],
            'Goal013 method or receipt binding differs')
    return {'canonical_root': str(canonical), 'canonical_products': paths, 'verified_files': inventory,
            'source_coverage': coverage, 'previous_status': preflight['status'], 'previous_scored_descriptions': 0}


def resource_snapshot():
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
            memory[key+'_bytes'] = int(value.split()[0])*1024
    cg_line = next(line for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    cg = safe(Path('/sys/fs/cgroup')/cg_line[3:].lstrip('/'))
    ancestors = []; finite_available = []
    while cg.is_relative_to('/sys/fs/cgroup'):
        values = {}
        for name in ('memory.max', 'memory.high', 'memory.current', 'memory.swap.max', 'cpu.max', 'pids.max'):
            try:
                values[name] = (cg/name).read_text().strip()
            except FileNotFoundError:
                require(cg == Path('/sys/fs/cgroup'), 'non-root cgroup control unavailable')
                values[name] = 'NOT_EXPOSED_AT_CGROUP_ROOT'
        for name in ('memory.max', 'memory.high'):
            if values[name].isdigit():
                require(values['memory.current'].isdigit(), 'finite cgroup cap without memory.current')
                finite_available.append(max(0, int(values[name])-int(values['memory.current'])))
        ancestors.append({'path': str(cg), **values})
        if cg == Path('/sys/fs/cgroup'): break
        cg = cg.parent
    available = min([memory['MemAvailable_bytes'], *finite_available])
    headroom = 2*1024**3
    limit = min(6*1024**3, max(0, available-headroom))
    disk = os.statvfs(ROOT); free = disk.f_bavail*disk.f_frsize
    return {'host_memory': memory, 'cgroup_ancestors': ancestors,
            'available_memory_safe_bytes': available, 'reserved_headroom_bytes': headroom,
            'recommended_process_memory_limit_bytes': limit, 'rss_target_bytes': min(4*1024**3, limit),
            'six_GiB_limit_fits_observed_available_memory': limit == 6*1024**3,
            'free_disk_bytes': free, 'required_minimum_free_disk_bytes': 10*1024**3,
            'minimum_disk_headroom_pass': free >= 10*1024**3,
            'process_comm_only': subprocess.check_output(['ps', '-eo', 'comm=']).decode().splitlines(),
            'process_arguments_or_environment_read': False, 'OS_level_isolation_guaranteed': False,
            'limitation': 'Point-in-time availability; root cgroup controller files may be absent. Recheck before inference.'}


def source_lock_paths(inventory):
    return sorted(Path(name) for name in inventory if Path(name).is_relative_to('/root/minecraft-runs') and Path(name).name.endswith('.lock'))


@contextmanager
def locks_for(inventory):
    handles = []
    try:
        for path in source_lock_paths(inventory):
            require(not path.is_symlink(), 'source lock symlink refused')
            stream = path.open('rb')
            handles.append(stream)
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield len(handles)
    finally:
        for stream in reversed(handles):
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN); stream.close()


def make_before(tools_root, run_root):
    require(git('rev-parse', 'HEAD') == ENTRY_HEAD and git('rev-parse', 'main') == MAIN_HEAD, 'entry HEAD/main changed')
    require(git('branch', '--show-current') == 'research/server-only-20261013', 'wrong branch')
    old = read_metadata(PRIOR)
    old_integrity = read_metadata(REPORT/'INTEGRITY.json')
    require(descriptor(PRIOR)['sha256'] == old_integrity['before']['before_inventory_file_sha256'], 'retained Goal013 inventory changed')
    require(digest(old['inventory']) == old_integrity['before']['inventory_sha256'], 'retained source inventory digest differs')
    with locks_for(old['inventory']) as lock_count:
        verify_files(old['inventory'])
        inventory = dict(old['inventory'])
        names = tracked()
        require(set(old['tracked_paths']) <= set(names), 'previous tracked file missing')
        for name in names:
            actual = descriptor(ROOT/name)
            require(str(ROOT/name) not in inventory or inventory[str(ROOT/name)] == actual, 'old tracked source changed')
            inventory[str(ROOT/name)] = actual
        goal013 = verify_goal013(); inventory.update(goal013['verified_files'])
        canonical_tree = tree_inventory(Path(goal013['canonical_root']))
        inventory.update({p: d for p, d in canonical_tree.items() if 'kind' not in d})
        environment = {}
        for root in old['environment_roots']:
            environment.update(tree_inventory(Path(root)))
        require({p: d for p, d in environment.items() if 'kind' not in d} == old['environment_inventory'], 'existing virtual environment differs from Goal013')
        audit = read_metadata(REPORT/'OFFLINE_SCORER_AUDIT.json')['discovery_scope']
        cache_roots = sorted({r['path'] for r in audit['standard_model_paths']} | {audit['pip_cache']['path']})
        caches = {}
        for root in cache_roots: caches.update(tree_inventory(Path(root)))
        new_roots = [str(safe(tools_root)), str(safe(run_root))]
        require(Path(new_roots[0]).parent == Path('/root/minecraft-tools') and Path(new_roots[1]).parent == Path('/root/minecraft-runs')
                and all(Path(p).name.startswith('goal013a-') for p in new_roots), 'dedicated new Goal013A roots required')
        require(not any(Path(p).is_relative_to(new) for p in [*inventory, *environment, *caches] for new in new_roots), 'new Goal013A roots entered old protection scope')
        result = {'schema': 'goal013a_protected_snapshot_v1', 'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                  'entry_head': ENTRY_HEAD, 'main_head': MAIN_HEAD, 'tracked_paths': names,
                  'inventory': inventory, 'environment_roots': old['environment_roots'], 'environment_inventory': environment,
                  'cache_roots': cache_roots, 'cache_inventory': caches,
                  'goal013_canonical_root': goal013['canonical_root'], 'goal013_canonical_tree': canonical_tree,
                  'goal013_binding': goal013, 'prior_inventory': {'path': str(PRIOR), **descriptor(PRIOR)},
                  'excluded_new_roots': new_roots, 'excluded_new_repository_prefixes': list(NEW_PREFIXES),
                  'source_locks': lock_count, 'resource_state': resource_snapshot(),
                  'authorization_change': 'Goal013A explicitly permits fixed public source/weight acquisition, dedicated new venv install and local CPU inference; old Goal013 download/install prohibition remains unchanged for old paths.',
                  'GT_and_Builder_outcomes_content_parsed': False, 'research_api_requests_sent': 0}
    return result


def load_snapshot(path):
    require(descriptor(path)['sha256'] == SNAPSHOT_SHA256, 'protected snapshot itself changed')
    snapshot = read_metadata(path)
    require(snapshot['schema'] == 'goal013a_protected_snapshot_v1', 'wrong snapshot schema')
    return snapshot


def verify_snapshot(path):
    snapshot = load_snapshot(path)
    verify_files(snapshot['inventory'])
    require(tracked() == snapshot['tracked_paths'], 'preexisting tracked membership changed')
    require(git('rev-parse', 'main') == snapshot['main_head'], 'main changed')
    for roots, key in [('environment_roots', 'environment_inventory'), ('cache_roots', 'cache_inventory')]:
        actual = {}
        for root in snapshot[roots]: actual.update(tree_inventory(Path(root)))
        require(actual == snapshot[key], 'preexisting '+key+' changed')
    require(tree_inventory(Path(snapshot['goal013_canonical_root'])) == snapshot['goal013_canonical_tree'], 'Goal013 canonical run changed')
    return snapshot


@contextmanager
def protected_sources(snapshot_path):
    """Read-only cooperative locks plus full before/after comparison, no network hook."""
    snapshot = load_snapshot(snapshot_path)
    with locks_for(snapshot['inventory']) as count:
        verify_snapshot(snapshot_path)
        try:
            yield count
        finally:
            verify_snapshot(snapshot_path)


def receipt(snapshot, mode, snapshot_path, started, cpu):
    return {'schema': 'goal013a_integrity_receipt_v1', 'status': 'PASS', 'phase': mode,
            'checked_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'snapshot': {'path': str(snapshot_path), **descriptor(snapshot_path)},
            'protected_files': len(snapshot['inventory']), 'preexisting_tracked_files': len(snapshot['tracked_paths']),
            'inventory_sha256': digest(snapshot['inventory']),
            'existing_environment_files_and_symlinks': sum('kind' not in d for d in snapshot['environment_inventory'].values()),
            'environment_tree_sha256': digest(snapshot['environment_inventory']),
            'existing_cache_files_and_symlinks': sum('kind' not in d for d in snapshot['cache_inventory'].values()),
            'cache_tree_sha256': digest(snapshot['cache_inventory']), 'cache_roots': snapshot['cache_roots'],
            'goal013_canonical_tree_sha256': digest(snapshot['goal013_canonical_tree']),
            'source_locks': snapshot['source_locks'], 'excluded_new_roots': snapshot['excluded_new_roots'],
            'GT_and_Builder_outcomes_content_parsed': False, 'research_api_requests_sent': 0,
            'all_protected_original_bytes_unchanged': True,
            'existing_environments_and_cache_contents_membership_symlinks_unchanged': True,
            'OS_level_isolation_guaranteed': False, 'cpu_seconds': time.process_time()-cpu,
            'elapsed_seconds': time.monotonic()-started, 'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def write_new(path, value):
    path = safe(path)
    require(path.is_relative_to('/tmp') or path.is_relative_to(ROOT/'reports/server_only/goal013a'), 'new audit output path required')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False); stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['before', 'after'])
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--tools-root', type=Path)
    parser.add_argument('--run-root', type=Path)
    args = parser.parse_args(); started = time.monotonic(); cpu = time.process_time()
    if args.mode == 'before':
        require(args.tools_root is not None and args.run_root is not None, 'new tools/run roots required')
        require(not args.snapshot.exists() and not args.receipt.exists(), 'before output already exists')
        snapshot = make_before(args.tools_root, args.run_root); write_new(args.snapshot, snapshot)
    else:
        original = load_snapshot(args.snapshot)
        with locks_for(original['inventory']): snapshot = verify_snapshot(args.snapshot)
    result = receipt(snapshot, args.mode, args.snapshot, started, cpu)
    write_new(args.receipt, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
