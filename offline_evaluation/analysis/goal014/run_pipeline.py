"""Offline, exclusive-output Goal014 execution against a committed plan.

Only the existing fixed model adapter is used. Features/QA are written and
frozen before any Builder or surface outcome is read by this entry point.
"""
from __future__ import annotations

import argparse
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

from ..goal013a.run_pipeline import descriptor, offline_guard, require, write_json
from ..goal013a import integrity

ROOT = Path(__file__).resolve().parents[3]
PREFIXES = ('offline_evaluation/analysis/goal014/', 'reports/server_only/goal014/')
GIB = 1024**3


def read_bound(desc):
    actual = descriptor(desc['path'])
    require(all(actual[k] == desc[k] for k in ('sha256', 'bytes')), 'bound file changed: '+desc['path'])
    return json.loads(Path(desc['path']).read_text())


def verify_commit(commit, config_path):
    import re
    require(re.fullmatch('[0-9a-f]{40}', commit), 'full plan commit required')
    require(integrity.git('branch', '--show-current') == 'research/server-only-20261013', 'dedicated branch required')
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=ROOT, check=True)
    files = integrity.git('ls-tree', '-r', '--name-only', commit, '--', *PREFIXES).splitlines()
    require(str(config_path.relative_to(ROOT)) in files and any(p.endswith('ANALYSIS_PLAN.md') for p in files), 'committed config/plan absent')
    require(not integrity.git('status', '--porcelain', '--', PREFIXES[0]), 'uncommitted code refused')
    records = {}
    for name in files:
        raw = subprocess.check_output(['git', 'show', commit+':'+name], cwd=ROOT)
        require((ROOT/name).read_bytes() == raw, 'precalculation code/plan changed: '+name)
        records[name] = descriptor(ROOT/name)
    return records


def configure_environment(run):
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '1'
    for name in ('TMPDIR', 'TMP', 'TEMP'):
        os.environ[name] = str(run/'tmp')
    for name in ('XDG_CACHE_HOME', 'TORCH_HOME', 'PIP_CACHE_DIR', 'HF_HOME', 'MPLCONFIGDIR'):
        os.environ[name] = str(run/'cache')


class Budget:
    def __init__(self, config, run):
        self.config = config; self.run = run; self.cpu = time.process_time(); self.wall = time.monotonic()
        state = integrity.resource_snapshot()
        self.memory_limit = min(config['limits']['rss_max_bytes'], state['recommended_process_memory_limit_bytes'])
        require(self.memory_limit >= 2*GIB, 'insufficient safe available memory')
        self.remaining_cpu = config['limits']['cpu_seconds']-config['pre_execution_cpu_reserve_seconds']-config['closure_cpu_reserve_seconds']
        remaining_wall = config['limits']['wall_seconds']-(time.time()-config['goal_start_epoch'])
        require(remaining_wall > 0 and self.remaining_cpu > 0, 'goal budget exhausted before computation')
        def exceeded(*unused):
            signal.alarm(0)
            raise RuntimeError('fixed Goal014 CPU/wall limit reached')
        signal.signal(signal.SIGXCPU, exceeded); signal.signal(signal.SIGALRM, exceeded)
        resource.setrlimit(resource.RLIMIT_AS, (self.memory_limit, self.memory_limit))
        soft = math.ceil(time.process_time()+self.remaining_cpu)
        resource.setrlimit(resource.RLIMIT_CPU, (soft, soft+10))
        signal.alarm(max(1, math.floor(remaining_wall)))
        self.initial_memory = state
        self.check()

    def check(self):
        usage = {'cpu_seconds': time.process_time()-self.cpu, 'elapsed_seconds': time.monotonic()-self.wall,
                 'goal_elapsed_seconds': time.time()-self.config['goal_start_epoch'],
                 'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 'free_disk_bytes': shutil.disk_usage(self.run).free,
                 'run_bytes': sum(p.stat().st_size for p in self.run.rglob('*') if p.is_file() and not p.is_symlink())}
        require(usage['cpu_seconds'] < self.remaining_cpu, 'CPU allowance exceeded')
        require(usage['goal_elapsed_seconds'] < self.config['limits']['wall_seconds'], 'goal wall allowance exceeded')
        require(usage['max_rss_kib']*1024 <= self.memory_limit, 'RSS limit exceeded')
        require(usage['run_bytes'] <= self.config['limits']['run_bytes'], 'run size limit exceeded')
        require(usage['free_disk_bytes'] >= self.config['limits']['free_disk_bytes'], 'disk headroom limit exceeded')
        return usage


def load_sources(config):
    old_manifest = read_bound(config['old_manifest'])
    old_config = read_bound(config['old_config'])
    freeze = read_bound(config['score_freeze'])
    scores = read_bound(config['score']); sources = read_bound(config['source_binding'])
    require(freeze['status'] == 'PASS' and freeze['phase2_allowed'], 'old freeze gate failed')
    require(freeze['specification_artifact'] == config['old_config'], 'old scorer config is not freeze-bound')
    require(Path(sys.prefix) == Path(old_config['tools_root'])/'venv' and sys.dont_write_bytecode,
            'existing dedicated venv and disabled bytecode required')
    require(freeze['score_artifact'] == config['score'] and freeze['source_binding_artifact'] == config['source_binding'], 'freeze binding differs')
    for name, expected in old_manifest['code_files'].items():
        current = descriptor(ROOT/name)
        require(all(current[k] == expected[k] for k in ('sha256', 'bytes')), 'fixed old code changed')
    for name, expected in old_config['installed_distributions'].items():
        require(importlib.metadata.version(name) == expected, 'fixed installed dependency version differs: '+name)
    key = lambda r: (r['scene_id'], r['description_provider'])
    scores.sort(key=key); sources.sort(key=key)
    require(len(scores) == len(sources) == 180, '180 planned rows required')
    for score, source in zip(scores, sources):
        require(all(score.get(k) == v for k, v in source.items() if k not in ('score', 'score_status')),
                'original score/source binding differs')
    return old_config, scores, sources


def extract(config, old_config, scores, sources, run, budget):
    import numpy as np
    from ..goal013a import scorer
    from ..goal013a.reference_parity import run_reference_parity
    from ..goal013.source_binding import load_scoring_record
    from . import diagnostics
    identities = {'score_sha256': config['score']['sha256'], 'config_sha256': config['old_config']['sha256'],
                  'source_binding_sha256': config['source_binding']['sha256']}
    metadata = diagnostics.build_feature_metadata(scores, old_config, **identities)
    old_run = Path(read_bound(config['old_manifest'])['run_root'])
    candidates = sorted(str(p) for p in old_run.rglob('*') if p.is_file() and
                        (p.suffix in ('.npy', '.npz', '.pkl', '.pickle', '.pt') or 'feature' in p.name.lower() or 'embedding' in p.name.lower()))
    require(not candidates, 'potential old disk features found: inspect provenance before any re-encoding')
    write_json(run/'FEATURE_AVAILABILITY.json', {'old_run': str(old_run), 'candidate_files': candidates,
               'status': 'NO_SAVED_FEATURE_ARTIFACT', 'memory_cache_is_not_disk_cache': True})
    session = scorer.load_model(old_config)
    budget.check()
    source_by_key = {(r['scene_id'], r['description_provider']): r for r in sources}
    first = source_by_key[(metadata['text_rows'][0]['scene_id'], metadata['text_rows'][0]['description_provider'])]
    payload = load_scoring_record(first)
    actual = {'parity': {'images': 0, 'texts': 0, 'image_batches': 0, 'text_batches': 0},
              'extraction': {'images': 0, 'texts': 0, 'image_batches': 0, 'text_batches': 0}}
    phase = ['parity']
    encode_image, encode_text = session.model.encode_image, session.model.encode_text
    def counted_image(tensor):
        actual[phase[0]]['images'] += len(tensor); actual[phase[0]]['image_batches'] += 1
        return encode_image(tensor)
    def counted_text(tensor):
        actual[phase[0]]['texts'] += len(tensor); actual[phase[0]]['text_batches'] += 1
        return encode_text(tensor)
    session.model.encode_image = counted_image; session.model.encode_text = counted_text
    start = budget.check()
    parity = run_reference_parity(session, payload['images_ordered'][:1], [payload['summary']])
    parity['resources_after'] = budget.check()
    parity['cpu_seconds'] = parity['resources_after']['cpu_seconds']-start['cpu_seconds']
    parity['elapsed_seconds'] = parity['resources_after']['elapsed_seconds']-start['elapsed_seconds']
    parity['input_identity'] = metadata['text_rows'][0]
    write_json(run/'REFERENCE_PARITY.json', parity)
    require(parity['status'] == 'PASS', 'real author reference parity failed')
    session.image_cache.clear(); session.text_cache.clear()
    session.cache_counts = dict.fromkeys(session.cache_counts, 0); phase[0] = 'extraction'
    texts = []; token_checks = []
    for identity in metadata['text_rows']:
        row = source_by_key[(identity['scene_id'], identity['description_provider'])]
        payload = load_scoring_record(row)
        token = session.token_diagnostics(payload['summary'])
        require(token['token_ids_sha256'] == identity['token_ids_sha256']
                and token['prefixed_text_sha256'] == identity['prefixed_text_sha256']
                and token['summary_sha256'] == identity['summary_sha256'], 'new tokenizer input differs')
        token_checks.append({k: identity[k] for k in ('scene_id', 'description_provider', 'token_ids_sha256', 'summary_sha256')})
        texts.append(payload['summary'])
    # Scene batches preserve exactly the six-view extraction grouping of Goal013A.
    image_pieces = []
    for offset in range(0, len(metadata['image_rows']), 6):
        image_pieces.append(session.encode_images([r['path'] for r in metadata['image_rows'][offset:offset+6]], batch_size=8))
        budget.check()
        if offset % 60 == 0: print(json.dumps({'stage': 'images', 'encoded': offset+6}), flush=True)
    image_features = np.vstack(image_pieces)
    text_features = session.encode_texts(texts, batch_size=8)
    require(actual['extraction']['images'] == 360 and actual['extraction']['texts'] == 178, 'one extraction pass required')
    bundle = diagnostics.save_feature_bundle(run/'private', image_features, text_features, metadata)
    write_json(run/'TOKEN_REPLAY.json', {'status': 'PASS', 'rows': token_checks, 'raw_text_or_token_ids_exported': False})
    write_json(run/'INFERENCE_COUNTS.json', {'actual_model_forward_inputs': actual, 'cache': session.cache_counts,
               'research_api_requests': 0, 'external_model_requests': 0, 'downloads': 0, 'installs': 0,
               'resources_after_extraction': budget.check(), 'feature_manifest': descriptor(run/'private/FEATURE_MANIFEST.json')})
    del session, texts
    # Reopen with allow_pickle=False and verify hashes before arithmetic.
    image_features, text_features, loaded_metadata = diagnostics.load_feature_bundle(run/'private/FEATURE_MANIFEST.json', metadata)
    result = diagnostics.compute_diagnostics(image_features, text_features, loaded_metadata, scores, old_config, **identities)
    matrix = diagnostics.save_numeric_array(run/'private/MATCHING_MATRIX_178x60.npy', result['matrix'])
    rows = write_json(run/'QA_ROWS_180.json', result['rows'])
    replay = write_json(run/'SCORE_REPLAY.json', result['replay'])
    freeze = write_json(run/'QA_FREEZE.json', {'status': 'PASS', 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
             'matrix': matrix, 'rows': rows, 'replay': replay, 'features': descriptor(run/'private/FEATURE_MANIFEST.json'),
             'original_score': config['score'], 'Builder_and_surface_outcomes_loaded': False})
    return result['rows'], freeze


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--plan-commit', required=True)
    args = parser.parse_args(); config = json.loads(args.config.read_text()); run = Path(config['run_root'])
    require(run.parent == Path('/root/minecraft-runs') and run.name.startswith('goal014-') and not run.is_symlink(), 'dedicated run required')
    require(not (run/'EXECUTION_START.json').exists(), 'execution already started; preserve results and inspect before any retry')
    committed = verify_commit(args.plan_commit, args.config.resolve())
    configure_environment(run); budget = Budget(config, run)
    write_json(run/'EXECUTION_START.json', {'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
               'plan_commit': args.plan_commit, 'committed_files': committed, 'config': descriptor(args.config),
               'initial_resources': budget.initial_memory, 'process_memory_limit_bytes': budget.memory_limit,
               'pre_execution_cpu_reserve_seconds': config['pre_execution_cpu_reserve_seconds'],
               'closure_cpu_reserve_seconds': config['closure_cpu_reserve_seconds']})
    snapshot = read_bound(config['protected_snapshot'])
    status = 'FAILED'; failure = None; guard_state = None
    try:
        with integrity.locks_for(snapshot['inventory']) as lock_count, offline_guard([run]) as guard_state:
            old_config, scores, sources = load_sources(config)
            qa_rows, qa_freeze = extract(config, old_config, scores, sources, run, budget)
            require(descriptor(qa_freeze['path']) == qa_freeze, 'QA freeze changed')
            from .binding import load_inputs
            outcome_rows, scenes, outcome_receipt = load_inputs()
            write_json(run/'OUTCOME_BINDING.json', outcome_receipt)
            write_json(run/'private/OUTCOME_ROWS_180.json', outcome_rows)
            budget.check()
            from .statistics import analyze
            from ..goal013a.associations import bootstrap_indices
            from .diagnostics import save_numeric_array
            save_numeric_array(run/'private/BOOTSTRAP_INDICES_2000x60.npy', bootstrap_indices(scenes))
            results = analyze(scores, outcome_rows, qa_rows, scenes)
            full = write_json(run/'private/RESULTS_FULL.json', results)
            summary = json.loads(json.dumps(results))
            for question in ('q_a', 'q_b', 'q_c'):
                for row in summary[question]: row.pop('bootstrap_draws', None)
            summary['canonical_full_results'] = full
            summary['qa_freeze'] = qa_freeze
            write_json(run/'RESULTS_SUMMARY.json', summary)
            budget.check(); status = 'PASS'
    except BaseException as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        signal.alarm(0)
        write_json(run/'EXECUTION_RECEIPT.json', {'status': status, 'failure': failure,
                   'finished_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                   'cpu_seconds': time.process_time()-budget.cpu, 'elapsed_seconds': time.monotonic()-budget.wall,
                   'goal_elapsed_seconds': time.time()-config['goal_start_epoch'],
                   'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'guard': guard_state,
                   'source_locks': snapshot['source_locks'], 'downloads': 0, 'installs': 0, 'research_api_requests': 0,
                   'external_model_requests': 0, 'OS_isolation_guaranteed': False})
    print(json.dumps({'status': status, 'run': str(run)}))


if __name__ == '__main__': main()
