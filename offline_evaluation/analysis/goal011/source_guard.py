"""Bind new definitions and all inherited source proofs without modifying them."""
from __future__ import annotations

from pathlib import Path
import re
import subprocess

from ..goal010 import source_guard as inherited, pipeline as old
from ..goal010v2 import pipeline as v2

ROOT = inherited.ROOT
REPORT = ROOT / 'reports/server_only/goal011'
PLAN = 'reports/server_only/goal011/ANALYSIS_PLAN.json'
QUESTIONS = 'reports/server_only/goal011/QUESTION_REGISTER.md'
SPECS = (PLAN, QUESTIONS)
PRODUCTION = tuple('offline_evaluation/analysis/goal011/' + n + '.py' for n in (
    '__init__', 'source_guard', 'pipeline', 'tradeoff', 'material', 'material_batch', 'rest', 'prospective'))
V2_REPORT = ROOT / 'reports/server_only/goal010v2'
V2_MANIFEST_SHA = 'f286aa20562b0954fc42ab479100b416066eedb0551ea4d9df34264773f98a14'
V2_COMMIT = 'd862a55134bba85a6a5e9677bcc6e94c719ca1aa'
SOURCE_SHA = v2.SOURCE_SHA
sha = inherited.sha256_file
read_json = inherited.read_json


def binding(code_commit, specification_commit):
    present = {p.relative_to(ROOT).as_posix() for p in (ROOT/'offline_evaluation/analysis/goal011').glob('*.py')}
    if present != set(PRODUCTION):
        raise ValueError('production code differs from explicit allowlist')
    code = inherited.verify_commit(code_commit, [*PRODUCTION, *SPECS])
    specs = inherited.verify_commit(specification_commit, SPECS)
    if any(code[p] != specs[p] for p in SPECS):
        raise ValueError('analysis definitions differ from precomputation commit')
    return {'code_commit': code_commit, 'specification_commit': specification_commit,
            'plan_sha256': specs[PLAN], 'analysis_plan_sha256': specs[PLAN],
            'question_register_sha256': specs[QUESTIONS], 'hashes': code}


def _merge(inventory, path, descriptor=None):
    p = inherited.regular(path)
    actual = {'bytes': p.stat().st_size, 'sha256': sha(p)}
    if descriptor is not None and actual != {k: descriptor[k] for k in ('bytes', 'sha256')}:
        raise ValueError('protected source hash mismatch: '+str(p))
    if str(p) in inventory and inventory[str(p)] != actual:
        raise ValueError('source changed while inventoried: '+str(p))
    inventory[str(p)] = actual


def verify_inventory(inventory):
    for path, descriptor in inventory.items():
        _merge({}, path, descriptor)


def protected_tracked_paths():
    """Include every pre-existing tracked file; only the two new trees are exempt."""
    listing = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT,
                             capture_output=True, check=True).stdout.decode('utf-8')
    excluded = ('offline_evaluation/analysis/goal011/', 'reports/server_only/goal011/')
    return [inherited.safe_child(ROOT, name) for name in listing.split('\0')
            if name and not name.startswith(excluded)]


def load_sources():
    """Caller holds inherited locks and Python network denial throughout."""
    package = inherited.load_verified_sources()
    if old.digest(package) != SOURCE_SHA:
        raise ValueError('original source package changed')
    old_run = v2.verify_old()
    manifest_path = V2_REPORT/'MANIFEST.json'
    if sha(inherited.regular(manifest_path)) != V2_MANIFEST_SHA:
        raise ValueError('Goal010v2 manifest changed')
    manifest = read_json(manifest_path)
    inherited.verify_products(V2_REPORT, manifest['products'])
    inherited.verify_products(ROOT, manifest['code_files'])
    v2_run = Path(read_json(V2_REPORT/'analysis/PUBLIC_COPY_RECEIPT.json')['run_root'])
    receipts = v2.previous(v2_run, 5, v2.binding(V2_COMMIT, V2_COMMIT))
    inventory = dict(package['goal007_manifest']['source_files'])
    # Verify the full inherited original inventory, including plans/ledgers and
    # checkpoint proofs. No credentials or arbitrary home directories are read.
    verify_inventory(inventory)
    tracked = protected_tracked_paths()
    for path in tracked:
        _merge(inventory, path)
    for report, run in ((v2.OLD, old_run), (V2_REPORT, v2_run)):
        _merge(inventory, report/'MANIFEST.json')
        m = read_json(report/'MANIFEST.json')
        for rel, descriptor in m['products'].items():
            _merge(inventory, inherited.safe_child(report, rel), descriptor)
        for rel, descriptor in m['code_files'].items():
            _merge(inventory, inherited.safe_child(ROOT, rel), descriptor)
        receipt_names = ([f'PHASE{i}_RECEIPT.json' for i in range(6)] if report == v2.OLD
                         else [f'{i}_{v2.STAGES[i]}_RECEIPT.json' for i in range(5)])
        for name in receipt_names:
            _merge(inventory, run/name)
            for rel, descriptor in read_json(run/name)['products'].items():
                _merge(inventory, inherited.safe_child(run, rel), descriptor)
        if report == v2.OLD:
            pred_root = run/'phase1_baselines'
            for row in read_json(pred_root/'baseline_predictions_manifest.json')['predictions']:
                for descriptor in row['artifacts'].values():
                    _merge(inventory, inherited.safe_child(pred_root/'predictions', descriptor['path']), descriptor)
    proof = {'status':'PASS', 'source_package_sha256':SOURCE_SHA,
             'protected_inventory_sha256':old.digest(inventory), 'protected_file_count':len(inventory),
             'protected_tracked_file_count':len(tracked),
             'canonical_goal010_run_root':str(old_run), 'canonical_goal010v2_run_root':str(v2_run),
             'goal010_manifest_sha256':v2.OLD_MANIFEST_SHA,
             'goal010v2_manifest_sha256':V2_MANIFEST_SHA, 'goal010v2_receipts':receipts,
             'historical_scope':old.package_receipt(package)}
    return package, old_run, v2_run, inventory, proof


def gate(bound, proof):
    if proof.get('status') != 'PASS' or proof.get('source_package_sha256') != SOURCE_SHA:
        raise ValueError('verified historical source proof required')
    return {'status':'PASS', 'input_kind':'verified_sources', 'phase0_binding':'PASS',
            'definitions_frozen':True, **bound, 'source_package_sha256':SOURCE_SHA,
            'specification_sha256':bound['plan_sha256'],
            'goal007_manifest_sha256':inherited.audit8.MANIFEST_SHA256}


def tradeoff_inputs(package, old_run, v2_run):
    return {
        'metric_rows':read_json(v2_run/'3_full/new_metric_values_960.json'),
        'baseline_rows':read_json(v2_run/'3_full/baseline_new_metric_values.json'),
        'saved_summaries':read_json(v2_run/'4_summary/metric_baselines_paired.json'),
        'old_summaries':read_json(old_run/'phase1_baselines/baseline_matched_deltas.json'),
        'baseline_manifest':read_json(old_run/'phase1_baselines/baseline_predictions_manifest.json'),
        'baseline_evaluations':read_json(old_run/'phase1_baselines/baseline_per_building.json'),
        'cohort_rows':[r for r in package['cohort']['scenes'] if r['split']=='evaluation'],
        'input_records':package['full_manifest']['identity']['full_schedule'],
        'original_slots':package['slots'],
        'integrity_reference':old.package_receipt(package)}
