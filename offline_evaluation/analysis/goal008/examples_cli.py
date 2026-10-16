"""Render fixed Goal008 examples only after source, Phase1 and commit verification."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import resource
import shlex
import shutil
import subprocess
import sys
import time

from . import audit, examples
from ..goal007.integration import write_new_json
from ...server_v2.data import sha256_file
from ...server_v2.generation import utc_now

ROOT = audit.ROOT
DEFINITION = ROOT / 'reports/server_only/goal008/example_definition.md'
FROZEN_FILES = [f'offline_evaluation/analysis/goal008/{name}' for name in (
    'examples.py', 'examples_cli.py', 'tests/test_examples.py', 'tests/test_examples_cli.py')
] + ['reports/server_only/goal008/example_definition.md']


def verify_commit_files(commit, expected):
    if not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('explicit full code commit required')
    for name, digest in expected.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe code binding path')
        if sha256_file(audit.regular(ROOT / relative)) != digest:
            raise ValueError('current code or definition hash mismatch: ' + name)
        stored = subprocess.run(['git', 'show', f'{commit}:{name}'], cwd=ROOT, capture_output=True)
        if stored.returncode or hashlib.sha256(stored.stdout).hexdigest() != digest:
            raise ValueError('frozen commit mismatch: ' + name)


def verify_own_commit(commit):
    hashes = {name: sha256_file(audit.regular(ROOT / name)) for name in FROZEN_FILES}
    verify_commit_files(commit, hashes)
    if hashes['reports/server_only/goal008/example_definition.md'] != examples.DEFINITION_SHA256:
        raise ValueError('pre-selection specification changed')
    return hashes


def verify_phase1(phase0_gate, phase1_manifest):
    """Hash verification only: proxy observations are never used for selection."""
    path = audit.regular(phase1_manifest)
    manifest = audit.read_json(path)
    if manifest.get('status') != 'PHASE0_PHASE1_PASS' or manifest.get('version') != 'goal008_phase1_v1':
        raise ValueError('Phase0 and Phase1 PASS required before source selection')
    if (Path(manifest['source_gate_path']).resolve() != Path(phase0_gate).resolve()
            or manifest['source_gate_sha256'] != sha256_file(audit.regular(phase0_gate))):
        raise ValueError('Phase1 source gate binding mismatch')
    if (manifest.get('matched_descriptions') != 180 or manifest.get('source_bound_builder_observations') != 360
            or manifest.get('independent_buildings') != 60 or manifest.get('planned_associations') != 360
            or manifest.get('source_inventories_before_after_unchanged') is not True
            or manifest.get('historical_RQ2_status') != 'NOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES'):
        raise ValueError('Phase1 fixed scope or immutable-source evidence mismatch')
    for field in ('research_api_requests_sent', 'api_key_queries', 'model_queries', 'human_experiments', 'checkpoint_mutations'):
        if manifest.get(field) != 0:
            raise ValueError('Phase1 zero-operation evidence mismatch: ' + field)
    expected_products = {'definitions_frozen.json', 'proxy_scores_180.json', 'proxy_scores_180.csv',
        'exploratory_associations.json', 'exploratory_associations_360.csv', 'proxy_descriptive.csv',
        'joined_descriptions_180.json', 'source_map_180.csv'}
    if set(manifest.get('products', {})) != expected_products:
        raise ValueError('Phase1 product set mismatch')
    for name, descriptor in manifest['products'].items():
        product = audit.regular(path.parent / name)
        if product.stat().st_size != descriptor['bytes'] or sha256_file(product) != descriptor['sha256']:
            raise ValueError('Phase1 product hash mismatch: ' + name)
    verify_commit_files(manifest['created_code_commit'], manifest['code_and_definition_hashes'])
    for relative, field in [('MANIFEST.json', 'goal007_manifest_sha256'),
                            ('data/final_slots_960.json', 'goal007_slots_sha256'),
                            ('results/comparisons_20.csv', 'goal007_comparisons_sha256')]:
        if sha256_file(audit.regular(audit.GOAL007 / relative)) != manifest[field]:
            raise ValueError('frozen Goal007 source hash mismatch: ' + relative)
    gate = audit.verify_report(phase0_gate)
    return gate, manifest


def bind_example_sources(slots, descriptions):
    """Bind fixed OpenAI roles to the audited original descriptions and GT."""
    index = {(row['scene_id'], row['description_provider']): row for row in descriptions}
    scenes = {row['scene_id'] for row in slots}
    if (len(descriptions) != 180 or len(index) != 180 or len(scenes) != 60
            or set(index) != {(scene, provider) for scene in scenes for provider in audit.DESCRIPTION_PROVIDERS}):
        raise ValueError('complete unique 180-description source map required')
    gt_sources, selected_descriptions = {}, {}
    for scene in sorted(scenes):
        source = index[(scene, 'openai')]
        for provider in audit.DESCRIPTION_PROVIDERS:
            other = index[(scene, provider)]
            for field in ('gt_descriptor', 'gt_descriptor_sha256', 'gt_bbox_path', 'gt_voxels_path'):
                if other[field] != source[field]:
                    raise ValueError('per-scene GT source disagreement')
        if source.get('O_J_typed_tree_equal') is not True or source['source_sha256'] != source['input_sha256']:
            raise ValueError('original O/lossless J source binding mismatch')
        fixed = [row for row in slots if row['scene_id'] == scene and row['provider'] == 'openai'
                 and ((row['condition'] in ('O', 'J') and row['description_model'] == 'openai')
                      or (row['condition'] in ('P', 'E') and row['description_model'] is None))]
        if len(fixed) != 4 or {row['condition'] for row in fixed} != {'O', 'J', 'P', 'E'}:
            raise ValueError('one slot for each fixed example condition required')
        for row in fixed:
            if (row['source_gt_descriptor_hash'] != source['gt_descriptor_sha256']
                    or row['gt_grid_sha256'] != source['gt_descriptor']['grid_sha256']):
                raise ValueError('example slot GT binding mismatch')
            if row['condition'] in ('O', 'J'):
                prefix = '' if row['condition'] == 'O' else 'j_'
                if row['input_id'] != source[prefix + 'input_id'] or row['input_sha256'] != source[prefix + 'input_sha256']:
                    raise ValueError('example O/J input binding mismatch')
            if row['condition'] == 'O':
                observed = source['builder_observations']['openai']
                for field in ('attempt_id', 'global_schedule_index', 'status', 'source_plan_identity'):
                    if row[field] != observed[field]:
                        raise ValueError('example O selected-attempt binding mismatch')
        gt_sources[scene] = {'kind': 'gt_numpy', 'path': source['gt_voxels_path'],
            'sha256': source['gt_descriptor']['voxels_sha256'], 'bbox_path': source['gt_bbox_path'],
            'bbox_sha256': source['gt_descriptor']['bbox_sha256'], 'verified': True}
        selected_descriptions[scene] = source
    return gt_sources, selected_descriptions


def build_packages(selection, descriptions, gate):
    packages = []
    for chosen in selection['selected']:
        gt = chosen['gt_source']
        panels = {'GT': {'status': 'success', 'source': deepcopy(gt), 'metrics': None,
                         'world_voxels': examples.load_source_world(gt, audit_gate=gate)}}
        for condition in ('O', 'J', 'P', 'E'):
            slot = chosen['slots'][condition]
            if slot is None:
                raise ValueError('canonical selected slot missing')
            panel = deepcopy(slot)
            panel['world_voxels'] = None
            if slot['status'] == 'success':
                artifact = slot['artifacts']['voxels.json']
                relative = Path(artifact['path'])
                run = Path(slot['source_run']).resolve()
                path = audit.regular(run / relative)
                if relative.is_absolute() or '..' in relative.parts or not path.resolve().is_relative_to(run):
                    raise ValueError('selected artifact escapes its original run')
                if path.stat().st_size != artifact['bytes']:
                    raise ValueError('selected voxel artifact byte count mismatch')
                panel['source'] = {'kind': 'server_v2_grid_json', 'path': str(path), 'sha256': artifact['sha256']}
                panel['world_voxels'] = examples.load_source_world(panel['source'], audit_gate=gate)
            else:
                if condition != 'E' or slot.get('metrics') is not None or 'voxels.json' in slot.get('artifacts', {}):
                    raise ValueError('failed example must retain no invented geometry or metrics')
                panel['source'] = {field: deepcopy(slot.get(field)) for field in (
                    'source_run', 'source_plan_path', 'source_plan_identity', 'source_entry_hash', 'attempt_id', 'artifacts')}
            panels[condition] = panel
        description = descriptions[chosen['scene_id']]
        packages.append({'scene_id': chosen['scene_id'], 'panels': panels,
            'description': examples.description_excerpt(description['source_path'], description['source_sha256'], audit_gate=gate),
            'gt_bbox_source': {'path': gt['bbox_path'], 'sha256': gt['bbox_sha256'], 'bbox': audit.read_json(gt['bbox_path'])}})
    return packages


def run_from_verified(phase0_gate, phase1_manifest, new_destination, code_commit):
    started = time.monotonic()
    out = Path(new_destination).resolve()
    if out.exists() or not out.is_relative_to(Path('/root/minecraft-runs')) or not out.parent.is_dir():
        raise ValueError('fresh destination under an existing research run directory required')
    if shutil.disk_usage(out.parent).free < 10 * 1024**3:
        raise ValueError('10GiB free disk required')
    own_hashes = verify_own_commit(code_commit)
    gate, phase1 = verify_phase1(phase0_gate, phase1_manifest)
    original_manifest = audit.read_json(audit.GOAL007 / 'MANIFEST.json')
    plans = {name: audit.read_json(source['plan_path']) for name, source in original_manifest['sources'].items()}
    source_gate_sha = sha256_file(phase0_gate)
    phase1_sha = sha256_file(phase1_manifest)
    with audit.old_audit.source_locks(plans), audit.old_audit.no_python_network():
        audit.old_audit.process_audit()
        audit.verify_report(gate)
        out.mkdir(exist_ok=False)
        gate_for_examples = {'status': 'PASS', 'input_kind': 'verified_sources', 'phase0_status': 'PASS', 'phase1_status': 'PASS'}
        frozen = {'status': 'FROZEN_BEFORE_EXAMPLE_SELECTION', 'at_utc': utc_now(), 'code_commit': code_commit,
            'code_and_definition_hashes': own_hashes, 'phase0_gate_sha256': source_gate_sha,
            'phase1_manifest_sha256': phase1_sha, 'selection_uses_proxy_scores': False}
        write_new_json(out / 'example_inputs_frozen.json', frozen)
        slots = audit.read_json(audit.GOAL007 / 'data/final_slots_960.json')
        descriptions = audit.read_json(gate['description_sources_path'])
        gt_sources, selected_descriptions = bind_example_sources(slots, descriptions)
        try:
            selection = examples.select_examples(slots, gt_sources, definition_path=DEFINITION, audit_gate=gate_for_examples)
        except examples.ExampleSelectionError as exc:
            write_new_json(out / 'example_selection_blocked.json', exc.audit)
            raise
        packages = build_packages(selection, selected_descriptions, gate_for_examples)
        rendered = examples.render_examples(packages, selection, out / 'figures', definition_path=DEFINITION, audit_gate=gate_for_examples)
        audit.verify_report(gate)
        verify_phase1(phase0_gate, phase1_manifest)
        audit.old_audit.process_audit()
        if (verify_own_commit(code_commit) != own_hashes or sha256_file(phase0_gate) != source_gate_sha
                or sha256_file(phase1_manifest) != phase1_sha):
            raise ValueError('frozen examples inputs changed during rendering')
        argv = [sys.executable, '-B', '-m', 'offline_evaluation.analysis.goal008.examples_cli',
                '--phase0-gate', str(Path(phase0_gate).resolve()), '--phase1-manifest', str(Path(phase1_manifest).resolve()),
                '--destination', str(out), '--code-commit', code_commit]
        products = {str(path.relative_to(out)): {'sha256': sha256_file(path), 'bytes': path.stat().st_size}
                    for path in sorted(out.rglob('*')) if path.is_file() and '.matplotlib-cache' not in path.parts}
        final = {'status': 'PHASE2_EXAMPLES_PASS', 'version': 'goal008_examples_v1', 'created_at_utc': utc_now(),
            'created_code_commit': code_commit, 'code_and_definition_hashes': own_hashes,
            'phase0_gate_path': str(Path(phase0_gate).resolve()), 'phase0_gate_sha256': source_gate_sha,
            'phase1_manifest_path': str(Path(phase1_manifest).resolve()), 'phase1_manifest_sha256': phase1_sha,
            'phase1_created_code_commit': phase1['created_code_commit'], 'goal007_manifest_sha256': audit.MANIFEST_SHA256,
            'goal007_slots_sha256': phase1['goal007_slots_sha256'], 'goal007_comparisons_sha256': phase1['goal007_comparisons_sha256'],
            'source_inventories_before_after_unchanged': True,
            'original_immutable_file_count': gate['original_immutable_file_count'], 'goal007_file_count': gate['goal007_file_count'],
            'eligible_buildings': selection['eligible_count'], 'selected_buildings': rendered['examples'],
            'selection_rule_id': examples.RULE_ID, 'proxy_scores_used_for_selection': False,
            'rendered_surface_policy': rendered['mesh_policy'], 'rendered_coordinate_policy': rendered['coordinate_policy'],
            'research_api_requests_sent': 0, 'api_key_queries': 0, 'model_queries': 0, 'human_experiments': 0,
            'checkpoint_mutations': 0, 'reconstruction_metrics_recomputed': False, 'products': products,
            'reproduction': {'cwd': str(ROOT), 'argv': argv, 'shell_command': shlex.join(argv),
                             'note': 'Choose a new --destination; existing results are never overwritten.'},
            'performance': {'seconds': time.monotonic() - started,
                            'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            'matplotlib_version': rendered['matplotlib_version'], 'free_disk_bytes': shutil.disk_usage(out).free}}
        write_new_json(out / 'EXAMPLES_MANIFEST.json', final)
        print(json.dumps({key: final[key] for key in ('status', 'selected_buildings', 'eligible_buildings', 'performance', 'research_api_requests_sent')}), flush=True)
        return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase0-gate', required=True)
    parser.add_argument('--phase1-manifest', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--code-commit', required=True)
    args = parser.parse_args(argv)
    try:
        run_from_verified(args.phase0_gate, args.phase1_manifest, args.destination, args.code_commit)
    except Exception as exc:
        print(json.dumps({'status': 'STOP', 'error_type': type(exc).__name__,
            'safe_reason': str(exc) if isinstance(exc, (ValueError, audit.AuditError)) else None,
            'research_api_requests_sent': 0}), flush=True)
        return 5
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
