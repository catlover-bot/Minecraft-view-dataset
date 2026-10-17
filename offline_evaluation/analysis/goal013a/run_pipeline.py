"""Committed, offline Goal013A scoring and separately gated association CLI."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
CODE = 'offline_evaluation/analysis/goal013a/'
REPORT = 'reports/server_only/goal013a/'
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
GIB = 1024**3


def require(ok, message):
    if not ok:
        raise ValueError(message)


def descriptor(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return {'path': str(path), 'sha256': h.hexdigest(), 'bytes': path.stat().st_size}


def write_json(path, value):
    path = Path(path)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf-8')
    require(not path.exists(), 'exclusive output already exists')
    temporary = path.with_name(path.name+'.partial')
    with temporary.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path)  # Atomic publish that refuses an existing final name.
    temporary.unlink()
    return descriptor(path)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode().strip()


def verify_commits(code_commit, spec_commit, config_path):
    import re
    for value in (code_commit, spec_commit):
        require(re.fullmatch('[0-9a-f]{40}', value) is not None, 'full commit SHA required')
    require(code_commit != spec_commit, 'code and specification require separate commits')
    require(git('branch', '--show-current') == 'research/server-only-20261013', 'dedicated branch required')
    subprocess.run(['git', 'merge-base', '--is-ancestor', code_commit, spec_commit], cwd=ROOT, check=True)
    subprocess.run(['git', 'merge-base', '--is-ancestor', spec_commit, 'HEAD'], cwd=ROOT, check=True)
    checked = {}
    for commit, prefix in ((code_commit, CODE), (spec_commit, REPORT)):
        names = git('ls-tree', '-r', '--name-only', commit, '--', prefix).splitlines()
        require(names, 'committed code/spec files required')
        for name in names:
            raw = subprocess.check_output(['git', 'show', commit+':'+name], cwd=ROOT)
            require((ROOT/name).read_bytes() == raw, 'code/spec changed after pre-inference freeze: '+name)
            checked[name] = descriptor(ROOT/name)
    rel_config = str(Path(config_path).relative_to(ROOT))
    require(rel_config in checked, 'config must be in the specification commit')
    require(not git('status', '--porcelain', '--', CODE), 'uncommitted code files prohibited')
    return checked


@contextmanager
def offline_guard(write_roots):
    """Process audit guard; not a claim of OS-level isolation."""
    state = {'active': True, 'network_attempts_blocked': 0, 'source_writes_blocked': 0}
    roots = [Path(p).resolve() for p in write_roots]
    def write_allowed(path):
        if isinstance(path, int):
            return True
        p = Path(os.fsdecode(path)).absolute().resolve()
        return p == Path('/dev/null') or any(p.is_relative_to(r) for r in roots)
    def hook(event, args):
        if not state['active']:
            return
        if event in ('socket.connect', 'socket.connect_ex', 'socket.getaddrinfo', 'socket.sendto'):
            state['network_attempts_blocked'] += 1
            raise RuntimeError('network prohibited during offline analysis')
        writes = []
        if event == 'open':
            path, mode, flags = args
            if (mode and any(c in mode for c in 'wax+')) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                writes = [path]
        elif event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod', 'os.utime', 'os.truncate'):
            writes = [args[0]]
        elif event in ('os.rename', 'os.link', 'os.symlink'):
            writes = list(args[:2])
        if any(not write_allowed(p) for p in writes):
            state['source_writes_blocked'] += 1
            raise RuntimeError('write outside new dedicated output/cache refused')
        if event == 'subprocess.Popen':
            argv = args[1]
            require(isinstance(argv, (list, tuple)) and len(argv) > 1
                    and argv[0] == 'git' and argv[1] in ('show', 'rev-parse', 'ls-files'),
                    'only read-only Git inventory subprocesses allowed during analysis')
        if event in ('os.system', 'os.posix_spawn', 'os.exec', 'os.fork', 'os.forkpty'):
            raise RuntimeError('process execution outside read-only Git refused')
    sys.addaudithook(hook)
    try:
        yield state
    finally:
        state['active'] = False


class Budget:
    def __init__(self, tools_root, run_root, memory_limit):
        self.tools_root = Path(tools_root); self.run_root = Path(run_root)
        self.wall = time.monotonic(); self.cpu = time.process_time()
        self.memory_limit = memory_limit
        def exceeded(signum, frame):
            # Abort calculation. The hard-limit margin permits only stack
            # unwinding, immutable-source rehash and the partial-evidence receipt.
            signal.alarm(0)
            signal.signal(signal.SIGXCPU, signal.SIG_IGN)
            raise RuntimeError('fixed CPU or wall resource limit reached')
        signal.signal(signal.SIGXCPU, exceeded); signal.signal(signal.SIGALRM, exceeded)
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        cpu_soft = math.ceil(self.cpu+3600)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_soft, cpu_soft+120))
        signal.alarm(7200)

    def snapshot(self):
        return {'cpu_seconds': time.process_time()-self.cpu,
                'elapsed_seconds': time.monotonic()-self.wall,
                'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}

    def check(self, disk=False):
        now = self.snapshot()
        require(now['cpu_seconds'] < 3600 and now['elapsed_seconds'] < 7200, 'inference time budget exceeded')
        require(now['max_rss_kib']*1024 <= self.memory_limit, 'RSS memory limit exceeded')
        if disk:
            size = sum(p.stat().st_size for root in (self.tools_root, self.run_root)
                       for p in root.rglob('*') if p.is_file() and not p.is_symlink())
            require(size <= 8*GIB, 'new work area exceeds 8 GiB')
            require(shutil.disk_usage(self.run_root).free >= 10*GIB, 'free disk below 10 GiB')
            now.update(new_work_area_bytes=size, free_disk_bytes=shutil.disk_usage(self.run_root).free)
        return now


def public_rows(rows):
    result = json.loads(json.dumps(rows, allow_nan=False))
    for row in result:
        row['token_diagnostics'].pop('token_ids', None)
    return result


def projection(pilot, current):
    # First scene is scored again after clearing parity cache; ten-scene timing
    # includes its fresh score. Factor two is an advance safety multiplier.
    return {'safety_multiplier': 2, 'remaining_buildings': 50,
            'projected_total_cpu_seconds': current['cpu_seconds'] + 2*pilot['cpu_seconds']*5,
            'projected_total_wall_seconds': current['elapsed_seconds'] + 2*pilot['elapsed_seconds']*5,
            'status': 'PASS' if current['cpu_seconds']+10*pilot['cpu_seconds'] < 3600
                      and current['elapsed_seconds']+10*pilot['elapsed_seconds'] < 7200 else 'RESOURCE_PROJECTION_FAILED'}


def score_phase(config, out, code_commit, spec_commit, binding, budget):
    from . import scorer
    from .reference_parity import run_reference_parity
    from ..goal013.source_binding import load_scoring_record
    rows = sorted(json.loads(Path(binding['canonical_products']['SOURCE_BINDING_180.json']).read_text()),
                  key=lambda r: (r['scene_id'], r['description_provider']))
    require(len(rows) == 180, 'planned 180 rows required')
    scenes = sorted({r['scene_id'] for r in rows})
    require(scenes == config['planned_scene_ids'], 'planned scene frame changed')
    require(config['score_id'] == scorer.SCORE_ID and config['tolerances'] == scorer.TOLERANCES
            and config['transform'] == scorer.TRANSFORM, 'fixed scorer configuration differs')
    session = scorer.load_model(config)
    budget.check()
    input_row = rows[0]
    require({k: input_row[k] for k in ('scene_id', 'description_provider', 'summary_sha256')}
            == config['parity_input'], 'fixed parity input differs')
    payload = load_scoring_record(input_row)
    start = budget.snapshot()
    parity = run_reference_parity(session, payload['images_ordered'], [payload['summary']]*6)
    parity['resources'] = budget.snapshot()
    parity['stage_cpu_seconds'] = parity['resources']['cpu_seconds']-start['cpu_seconds']
    parity['stage_elapsed_seconds'] = parity['resources']['elapsed_seconds']-start['elapsed_seconds']
    parity['input_selection'] = config['parity_input']
    write_json(out/'REFERENCE_PARITY.json', parity)
    require(parity['status'] == 'PASS', 'real reference parity gate failed')
    session.image_cache.clear(); session.text_cache.clear()
    session.cache_counts = dict.fromkeys(session.cache_counts, 0)
    scored = []; crops = {}; pilots = []
    fixed_crops = {r['path']: r for r in scorer.fixed_crop_examples(rows)}
    total_start = budget.snapshot()
    for scene_index, scene in enumerate(scenes, 1):
        scene_rows = [r for r in rows if r['scene_id'] == scene]
        paths = [i['path'] for i in scene_rows[0]['images_ordered']]
        require(all(r['images_ordered'] == scene_rows[0]['images_ordered'] for r in scene_rows), 'provider image bindings differ')
        verified_payload = load_scoring_record(next(r for r in scene_rows if r['summary_status'] == 'VALID_ORIGINAL_SUMMARY'))
        require(verified_payload['images_ordered'] == paths, 'verified image source order differs')
        features = session.encode_images(paths, batch_size=8)
        for image_source in scene_rows[0]['images_ordered']:
            path = image_source['path']
            preview, metadata = session.crop_preview(path)
            crops[path] = {**image_source, **metadata, 'scene_id': scene}
            if path in fixed_crops:
                crop_path = out/(scene+'_fixed_first_crop.png')
                with crop_path.open('xb') as stream:
                    preview.save(stream, format='PNG')
                crops[path]['canonical_preview'] = descriptor(crop_path)
        for origin in scene_rows:
            row = dict(origin)
            if origin['summary_status'] == 'VALID_ORIGINAL_SUMMARY':
                payload = load_scoring_record(origin)
                require(payload['images_ordered'] == paths, 'verified payload views differ')
                token = session.token_diagnostics(payload['summary'])
                text = session.encode_texts([payload['summary']], batch_size=8)[0]
                row.update(scorer.score_six(features, text))
            else:
                token = session.token_diagnostics(None)
                row.update(scorer.aggregate_six([None]*6))
                row.update(score_status='MISSING_SUMMARY', null_reason=origin['summary_status'])
            row.update(token_diagnostics=token, token_coverage_status=token['token_coverage_status'],
                       summary_token_count_untruncated=token['summary_token_count_untruncated'],
                       weight_sha256=scorer.CHECKPOINT_SHA256, transform_sha256=scorer.json_hash(scorer.TRANSFORM),
                       tokenizer_sha256=session.tokenizer_identity, source_binding_status='PASS')
            scored.append(row)
            write_json(out/('row_%03d.json' % len(scored)), row)
            budget.check()
        write_json(out/('scene_%02d_scores.json' % scene_index), scored[-3:])
        budget.check(disk=True)
        if scene_index in (1, 10):
            now = budget.check(disk=True)
            pilot = {'status': 'PASS', 'buildings': scene_index, 'planned_rows': 3*scene_index,
                     'computed_rows': sum(r['score_status'] == 'COMPUTATION_VALID' for r in scored),
                     'cpu_seconds': now['cpu_seconds']-total_start['cpu_seconds'],
                     'elapsed_seconds': now['elapsed_seconds']-total_start['elapsed_seconds'],
                     'max_rss_kib': now['max_rss_kib'], 'cache': dict(session.cache_counts),
                     'Builder_outcomes_joined': False, 'score_quality_used_for_pass': False,
                     'scene_ids': scenes[:scene_index]}
            if scene_index == 10:
                pilot['projection'] = projection(pilot, now)
            write_json(out/('PILOT_%02d.json' % scene_index), pilot); pilots.append(pilot)
            print(json.dumps({'pilot': scene_index, 'resources': pilot}), flush=True)
            require(scene_index != 10 or pilot['projection']['status'] == 'PASS', 'pilot projects resource budget exceeded')
        if scene_index % 10 == 0:
            print(json.dumps({'scored_buildings': scene_index, 'resources': budget.check()}), flush=True)
    coverage = scorer.coverage_gate(scored)
    score_file = write_json(out/'SCORES_180_CANONICAL.json', scored)
    safe_score_file = write_json(out/'SCORES_180.json', public_rows(scored))
    gate_file = write_json(out/'TOKEN_REPRESENTATIVENESS.json', coverage)
    write_json(out/'CROP_METADATA_360.json', list(crops.values()))
    valid = sum(r['score_status'] == 'COMPUTATION_VALID' for r in scored)
    summary = {'status': 'PASS', 'score_id': scorer.SCORE_ID, 'planned_rows': 180,
               'planned_buildings': 60, 'planned_image_references': 1080, 'unique_images': len(crops),
               'computed_rows': valid, 'computed_image_text_pairs': 6*valid,
               'row_status_counts': dict(Counter(r['score_status'] for r in scored)),
               'truncated_nonempty_rows': sum(r['token_diagnostics'].get('truncated', False) for r in scored),
               'token_coverage_counts': dict(Counter(r['token_coverage_status'] for r in scored)),
               'representativeness_gate': coverage, 'cache': dict(session.cache_counts),
               'resources': budget.check(disk=True), 'research_api_requests_sent': 0,
               'human_quality_validity': 'NOT_VALIDATED', 'old_RQ2_modified': False}
    write_json(out/'SCORING_SUMMARY.json', summary)
    return {'status': 'PASS', 'source_binding_status': 'PASS', 'scorer_validation_status': parity['status'],
            'score_id': scorer.SCORE_ID, 'code_commit': code_commit, 'spec_commit': spec_commit,
            'score_artifact': safe_score_file, 'canonical_token_score_artifact': score_file,
            'source_binding_artifact': descriptor(binding['canonical_products']['SOURCE_BINDING_180.json']),
            'representativeness_artifact': gate_file, 'specification_artifact': descriptor(config['_config_path']),
            'reference_parity_artifact': descriptor(out/'REFERENCE_PARITY.json'),
            'phase2_allowed': coverage['phase2_allowed'], 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('score', 'associate'))
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--code-commit', required=True)
    parser.add_argument('--spec-commit', required=True)
    parser.add_argument('--freeze-sha256')
    args = parser.parse_args()
    require(sys.dont_write_bytecode and all(os.environ.get(k) == '1' for k in THREAD_ENV), '-B and all single-thread environment settings required')
    committed_files = verify_commits(args.code_commit, args.spec_commit, args.config)
    config = json.loads(args.config.read_text()); config['_config_path'] = str(args.config.resolve())
    tools_root = Path(config['tools_root']); run_root = Path(config['run_root'])
    require(tools_root.is_absolute() and tools_root.parent == Path('/root/minecraft-tools')
            and tools_root.name.startswith('goal013a-') and tools_root.resolve() == tools_root,
            'new dedicated tools root required')
    require(run_root.is_absolute() and run_root.parent == Path('/root/minecraft-runs')
            and run_root.name.startswith('goal013a-') and run_root.resolve() == run_root,
            'new dedicated run root required')
    require(Path(sys.prefix) == tools_root/'venv', 'dedicated new virtualenv required')
    for variable, expected in {'TMPDIR': tools_root/'tmp', 'XDG_CACHE_HOME': tools_root/'cache',
                               'TORCH_HOME': tools_root/'cache/torch', 'PIP_CACHE_DIR': tools_root/'cache/pip'}.items():
        require(os.environ.get(variable) == str(expected), 'dedicated environment setting required: '+variable)
    for name, version in config['installed_distributions'].items():
        require(importlib.metadata.version(name) == version, 'installed dependency differs: '+name)
    out = run_root/args.stage
    out.mkdir(exist_ok=False)
    committed_artifact = write_json(out/'COMMITTED_INPUT_FILES.json', committed_files)
    from . import integrity
    resource_before = integrity.resource_snapshot()
    require(resource_before['recommended_process_memory_limit_bytes'] >= config['memory_limit_bytes'], 'available memory below fixed conservative limit')
    write_json(out/'RESOURCES_BEFORE.json', resource_before)
    snapshot = run_root/'PROTECTED_INVENTORY_BEFORE.json'
    budget = None
    try:
        with offline_guard([out, tools_root/'tmp', tools_root/'cache']) as guard:
            with integrity.protected_sources(snapshot) as lock_count:
                budget = Budget(tools_root, run_root, config['memory_limit_bytes'])
                budget.check(disk=True)
                if args.stage == 'score':
                    binding = integrity.verify_goal013()
                    frozen = score_phase(config, out, args.code_commit, args.spec_commit, binding, budget)
                    frozen['committed_code_and_spec_artifact'] = committed_artifact
                else:
                    from .outcome_binding import load_frozen_outcomes
                    from .associations import analyze
                    rows, scenes, join = load_frozen_outcomes(run_root/'score'/'SCORE_FREEZE.json',
                                                            expected_freeze_sha256=args.freeze_sha256)
                    write_json(out/'OUTCOME_BINDING.json', join)
                    result = analyze(rows, scenes, phase2_gate=join)
                    write_json(out/'ASSOCIATIONS_FULL.json', result)
                    budget.check(disk=True)
            # PASS/freeze is written only after the old-source exit recheck.
            receipt = {'status': 'PASS', 'source_locks': lock_count, 'old_sources_unchanged': True,
                       'guard': dict(guard), 'research_api_requests_sent': 0, 'OS_level_isolation_guaranteed': False,
                       'resources_including_exit_integrity_check': budget.check(disk=True)}
            write_json(out/'EXECUTION_RECEIPT.json', receipt)
            if args.stage == 'score':
                frozen['execution_receipt'] = descriptor(out/'EXECUTION_RECEIPT.json')
                print(json.dumps({'score_freeze': write_json(out/'SCORE_FREEZE.json', frozen)}), flush=True)
                if not frozen['phase2_allowed']:
                    write_json(out/'PHASE2_NOT_RUN.json', {'status': 'NOT_RUN_REPRESENTATIVENESS_GATE',
                        'outcome_join_performed': False, 'association_N': None, 'coefficients': None,
                        'bootstrap_draws_executed': 0, 'gate_artifact': frozen['representativeness_artifact']})
    except BaseException as exc:
        if not (out/'STOP_RECEIPT.json').exists():
            partial_paths = sorted(out.glob('row_*.json'))
            partial = [json.loads(p.read_text()) for p in partial_paths]
            write_json(out/'STOP_RECEIPT.json', {'status': 'STOPPED_WITH_PARTIAL_EVIDENCE',
                'exception_class': type(exc).__name__, 'reason': str(exc),
                'retained_planned_rows': len(partial), 'remaining_planned_rows': 180-len(partial) if args.stage == 'score' else None,
                'incomplete_output_files': [p.name for p in sorted(out.glob('*.partial'))],
                'retained_computed_rows': sum(r['score_status'] == 'COMPUTATION_VALID' for r in partial),
                'resources': budget.snapshot() if budget else None, 'research_api_requests_sent': 0})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
