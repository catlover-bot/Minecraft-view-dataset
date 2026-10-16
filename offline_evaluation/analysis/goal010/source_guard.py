"""Read-only source binding for Goal010, with no download or credential access."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import subprocess

from ..goal008 import audit as audit8
from ...server_v2.data import sha256_file

ROOT = audit8.ROOT
GOAL008 = ROOT / 'reports/server_only/goal008'
GOAL008_MANIFEST_SHA256 = '20e4433b3a56132f6031cd4ac08da14566f69795ad412a47d174503b0c053df6'
read_json = audit8.read_json
regular = audit8.regular


def safe_child(root, name):
    root = Path(root).resolve()
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('relative source path required')
    path = regular(root / relative)
    if not path.resolve().is_relative_to(root):
        raise ValueError('source path escaped its bound root')
    return path


def verify_products(root, products):
    for name, descriptor in products.items():
        path = safe_child(root, name)
        if path.stat().st_size != descriptor['bytes'] or sha256_file(path) != descriptor['sha256']:
            raise ValueError('bound product hash mismatch: ' + str(path))


def verify_goal008():
    path = regular(GOAL008 / 'MANIFEST.json')
    if sha256_file(path) != GOAL008_MANIFEST_SHA256:
        raise ValueError('frozen Goal008 manifest changed')
    manifest = read_json(path)
    verify_products(GOAL008, manifest['products'])
    for mapping in ('code_hashes', 'frozen_old_code_hashes'):
        for name, digest in manifest[mapping].items():
            path = Path(name)
            path = regular(path if path.is_absolute() else ROOT / path)
            if sha256_file(path) != digest:
                raise ValueError('frozen implementation hash changed: ' + name)
    canonical_count = 0
    for name in ('phase01_public_copy.json', 'examples_publication_public_copy.json'):
        receipt = read_json(GOAL008 / name)
        for descriptor in receipt['files'].values():
            path = regular(descriptor.get('canonical_source') or descriptor.get('source'))
            if sha256_file(path) != descriptor['sha256'] or path.stat().st_size != descriptor['bytes']:
                raise ValueError('Goal008 canonical output changed: ' + str(path))
            canonical_count += 1
    if canonical_count != 59:
        raise ValueError('Goal008 canonical product count changed')
    initial = read_json(regular(GOAL008 / 'initial_examples_manifest.json'))
    argv = initial['reproduction']['argv']
    example_root = Path(argv[argv.index('--destination') + 1])
    canonical_manifest = regular(example_root / 'EXAMPLES_MANIFEST.json')
    if sha256_file(canonical_manifest) != sha256_file(GOAL008 / 'initial_examples_manifest.json'):
        raise ValueError('Goal008 retained initial examples manifest changed')
    verify_products(example_root, initial['products'])
    gate = audit8.verify_report(GOAL008 / 'audit/source_gate.json')
    return manifest, gate


def load_verified_sources():
    goal008, gate = verify_goal008()
    goal007, _, full, cohort, slots = audit8.verify_goal007()
    descriptions = read_json(GOAL008 / 'audit/description_sources.json')
    scores = read_json(GOAL008 / 'phase1/proxy_scores_180.json')
    associations = read_json(GOAL008 / 'phase1/exploratory_associations.json')
    plans = {name: read_json(value['plan_path']) for name, value in goal007['sources'].items()}
    return {'goal007_manifest': goal007, 'goal008_manifest': goal008, 'goal008_gate': gate,
            'full_manifest': full, 'cohort': cohort, 'slots': slots,
            'descriptions': descriptions, 'proxy_scores': scores,
            'associations': associations, 'plans': plans}


@contextmanager
def locked_sources():
    """Lock existing writers read-only; deny Python socket connection attempts."""
    manifest = read_json(audit8.GOAL007 / 'MANIFEST.json')
    plans = {name: read_json(value['plan_path']) for name, value in manifest['sources'].items()}
    with audit8.old_audit.source_locks(plans) as locks, audit8.old_audit.no_python_network():
        audit8.old_audit.process_audit()
        yield locks
        audit8.old_audit.process_audit()


def require_gate(gate, specification_sha256):
    if (not isinstance(gate, dict) or gate.get('status') != 'PASS'
            or gate.get('input_kind') not in {'fixture', 'verified_sources'}
            or gate.get('phase0_binding') != 'PASS' or gate.get('definitions_frozen') is not True
            or gate.get('specification_sha256') != specification_sha256
            or not re.fullmatch('[0-9a-f]{64}', specification_sha256 or '')):
        raise ValueError('passing source gate and frozen post-hoc specification required')


def verify_commit(commit, relative_files):
    if not re.fullmatch('[0-9a-f]{40}', commit or ''):
        raise ValueError('explicit full code/specification commit required')
    result = {}
    for name in relative_files:
        path = safe_child(ROOT, name)
        blob = subprocess.run(['git', 'show', f'{commit}:{name}'], cwd=ROOT, capture_output=True)
        digest = sha256_file(path)
        if blob.returncode or hashlib.sha256(blob.stdout).hexdigest() != digest:
            raise ValueError('commit does not bind current file: ' + name)
        result[str(name)] = digest
    return result


def metadata_rows(package):
    """Observed covariates only; never passed to a baseline constructor."""
    cohort = {r['scene_id']: r for r in package['cohort']['scenes']}
    scores = {(r['scene_id'], r['description_provider']): r for r in package['proxy_scores']}
    def string_chars(value):
        if isinstance(value, str):
            return len(value) if value.strip() else 0
        if isinstance(value, list):
            return sum(map(string_chars, value))
        if isinstance(value, dict):
            return sum(map(string_chars, value.values()))
        return 0
    output = []
    if len(cohort) != len(package['cohort']['scenes']) or len(scores) != len(package['proxy_scores']):
        raise ValueError('duplicate source metadata identity')
    for source in package['descriptions']:
        path = regular(source['source_path'])
        if sha256_file(path) != source['source_sha256']:
            raise ValueError('description source changed')
        text = path.read_bytes().decode('utf-8')
        record = json.loads(text)
        row = scores[source['scene_id'], source['description_provider']]
        for key in ('input_id', 'input_sha256', 'source_sha256'):
            if row[key] != source[key]:
                raise ValueError('proxy-description metadata binding mismatch')
        for component in ('bbox', 'voxels'):
            key = 'gt_' + component + '_sha256'
            if row[key] != source['gt_descriptor'][component + '_sha256']:
                raise ValueError('proxy GT metadata binding mismatch')
            if sha256_file(regular(source['gt_' + component + '_path'])) != row[key]:
                raise ValueError('GT metadata file changed')
        bbox = read_json(source['gt_bbox_path'])
        volume = 1
        for axis in 'xyz':
            if (type(bbox[axis + 'min']) is not int or type(bbox[axis + 'max']) is not int
                    or bbox[axis + 'min'] > bbox[axis + 'max']):
                raise ValueError('invalid inclusive storage bbox extent')
            volume *= bbox[axis + 'max'] - bbox[axis + 'min'] + 1
        occupied = row['gt_material_target']['occupied_voxels']
        if type(occupied) is not int or not 0 < occupied <= volume:
            raise ValueError('invalid occupied count for storage volume')
        stratum = cohort[source['scene_id']]['size_rank_stratum']
        if type(stratum) is not int or not 0 <= stratum < 5:
            raise ValueError('invalid fixed size stratum')
        output.append({**{key: row[key] for key in ('scene_id', 'description_provider', 'input_id',
            'input_sha256', 'source_sha256', 'gt_bbox_sha256', 'gt_voxels_sha256')},
            'description_chars': len(text),
            'human_text_chars': sum(string_chars(record.get(k)) for k in audit8.HUMAN_FIELDS),
            'gt_occupied_count': occupied, 'gt_storage_bbox_volume': volume,
            'gt_occupancy_density': occupied / volume,
            'size_rank_stratum': cohort[source['scene_id']]['size_rank_stratum']})
    return output
