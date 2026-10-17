"""Read-only feasibility and prior-exposure audit for the frozen rest30.

Never constructs a prediction, calculates new geometry, tokenizes, writes files,
queries a model, or certifies that historical data were truly unseen. The caller
holds source locks/network denial and writes returned safe metadata exclusively.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import time

from ...server_v2.cohort_v4 import canonicalize_saved_text, freeze_ranked_split, json_text

ROOT = Path(__file__).resolve().parents[3]
PROVIDERS = ('openai', 'anthropic', 'gemini')
REPO_SCOPE = ('datasets', 'outputs', 'experiments', 'offline_evaluation',
              'reports', 'figures', 'human_studies')
TEXT_SUFFIXES = frozenset(('.json', '.jsonl', '.csv', '.tsv', '.md', '.txt', '.svg', '.tex',
                           '.yaml', '.yml', '.py', '.ipynb', '.html'))
BLOCKED_PARTS = frozenset(('.git', '.codex', '.agents', '.aws', '.ssh', '__pycache__', 'node_modules', '.pytest_cache'))
BLOCKED_FILES = frozenset(('authority.json', 'approval.json', 'enabled.json', '.env', 'credentials', 'credentials.json', 'token.json', 'tokens.json', 'secrets.json'))
TOKEN = re.compile(rb'buildings_100_v\d+(?:_|/|\\)building_\d{3}(?!\d)|(?<![A-Za-z0-9_])building_\d{3}(?!\d)')


class RestAuditError(ValueError):
    """Invalid gate, scope, or immutable source binding; caller must stop."""


@dataclass(frozen=True)
class RestLimits:
    max_search_files: int = 100_000
    max_search_bytes: int = 1024**3
    max_text_file_bytes: int = 32 * 1024**2
    max_source_file_bytes: int = 128 * 1024**2
    max_description_bytes: int = 2_000_000
    max_evidence_rows: int = 100_000
    cpu_seconds: float = 600.

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise RestAuditError('positive resource limits required: ' + name)


def _digest(value):
    return hashlib.sha256(json_text(value).encode('utf-8')).hexdigest()


def _gate(gate):
    if (not isinstance(gate, dict) or gate.get('status') != 'PASS'
            or gate.get('input_kind') not in ('fixture', 'verified_sources')
            or gate.get('phase0_binding') != 'PASS' or gate.get('definitions_frozen') is not True
            or any(not re.fullmatch('[0-9a-f]{64}', gate.get(k, ''))
                   for k in ('question_register_sha256', 'analysis_plan_sha256'))
            or not re.fullmatch('[0-9a-f]{40}', gate.get('code_commit', ''))):
        raise RestAuditError('frozen question/analysis plan, code and source gate required')


def _safe(path, root=None):
    p = Path(path)
    if not p.is_absolute() or '..' in p.parts or any(q.is_symlink() for q in (p, *p.parents)):
        raise RestAuditError('absolute non-symlink source path required')
    if root is not None and not p.is_relative_to(Path(root)):
        raise RestAuditError('source escaped its bound root')
    return p


def _relative(root, value):
    rel = Path(value)
    if rel.is_absolute() or '..' in rel.parts:
        raise RestAuditError('relative cohort source path required')
    return _safe(Path(root) / rel, root)


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def _record(path, inventory, expected=None, *, max_bytes=None):
    p = _safe(path)
    if not p.is_file(): raise RestAuditError('bound source is absent: ' + str(p))
    size = p.stat().st_size
    if max_bytes is not None and size > max_bytes: raise RestAuditError('bound source exceeds byte limit')
    value = {'bytes': size, 'sha256': _hash(p)}
    if expected is not None:
        if value['sha256'] != expected['sha256'] or size != expected.get('size_bytes', expected.get('bytes')):
            raise RestAuditError('immutable source bytes/hash mismatch: ' + str(p))
    if str(p) in inventory and inventory[str(p)] != value: raise RestAuditError('source changed during audit')
    inventory[str(p)] = value
    return value


def verify_inventory(inventory):
    for name, expected in inventory.items():
        p = _safe(name)
        if not p.is_file() or p.stat().st_size != expected['bytes'] or _hash(p) != expected['sha256']:
            raise RestAuditError('source inventory changed: ' + name)


def validate_split(cohort):
    """Replay the *saved* GT-ranked design; never load or rerank GT arrays."""
    rows = cohort['scenes']
    if (len(rows) != 100 or len({r['scene_id'] for r in rows}) != 100
            or cohort.get('population_scenes') != 100 or cohort.get('selection_uses_model_results') is not False
            or Counter(r['split'] for r in rows) != {'evaluation': 60, 'smoke': 10, 'rest': 30}):
        raise RestAuditError('unique frozen 100-scene 60/10/30 split required')
    rebuilt = freeze_ranked_split(rows, seed=cohort['seed'])
    expected = {r['scene_id']: r for r in rebuilt['scenes']}
    for row in rows:
        if not re.fullmatch(r'buildings_100_v4_building_\d{3}', row['scene_id']):
            raise RestAuditError('exact original v4 scene identity required')
        if any(row[k] != expected[row['scene_id']][k] for k in ('rank', 'split', 'size_rank_stratum', 'stratum_draw_order', 'gt_occupied')):
            raise RestAuditError('frozen ranked assignment differs')
    groups = {s: {r['scene_id'] for r in rows if r['split'] == s} for s in ('evaluation', 'smoke', 'rest')}
    if any(groups[a] & groups[b] for a, b in [('rest', 'evaluation'), ('rest', 'smoke'), ('evaluation', 'smoke')]):
        raise RestAuditError('cohort splits overlap')
    for q in range(5):
        if Counter(r['split'] for r in rows if r['size_rank_stratum'] == q) != {'evaluation': 12, 'smoke': 2, 'rest': 6}:
            raise RestAuditError('each quintile must have 12 evaluation /2 smoke /6 rest')
    return sorted((r for r in rows if r['split'] == 'rest'), key=lambda r: r['scene_id'])


def search_scope(package, *, root=ROOT):
    """Only explicit research dirs and roots already named by frozen manifests."""
    root = _safe(Path(root)); values = []
    for rel in REPO_SCOPE:
        values.append({'path': str(root / rel), 'origin': 'FIXED_REPOSITORY_RESEARCH_SCOPE'})
    values.append({'path': str(Path(package['full_manifest']['identity']['cohort_path']).parent),
                   'origin': 'FROZEN_FULL_MANIFEST_COHORT_PARENT'})
    for label, plan in sorted(package.get('plans', {}).items()):
        values.append({'path': plan['identity']['enabled_directory'], 'origin': 'VERIFIED_SOURCE_PLAN:' + label})
    for index, manifest in enumerate(package.get('legacy_analysis_manifests', [])):
        values.append({'path': manifest['canonical_run_root'], 'origin': 'VERIFIED_LEGACY_MANIFEST:' + str(index)})
    unique = {}
    for value in values:
        p = _safe(value['path'])
        if p == root: raise RestAuditError('whole repository is not an exposure search root')
        if not p.is_relative_to(root):
            run_parent = Path('/root/minecraft-runs')
            if not p.is_relative_to(run_parent) or not p.relative_to(run_parent).parts or not p.relative_to(run_parent).parts[0].startswith('goal'):
                raise RestAuditError('external scope must be one manifest-bound goal run directory')
        if _excluded(p): raise RestAuditError('excluded directory cannot be an exposure search root')
        unique.setdefault(str(p), []).append(value['origin'])
    return [{'path': p, 'origins': origin} for p, origin in sorted(unique.items())]


def _excluded(path):
    parts = path.parts
    return (any(p in BLOCKED_PARTS or p.startswith('.env') or p.startswith('.venv') for p in parts)
            or path.name in BLOCKED_FILES or path.name.lower().endswith(('.pem', '.key'))
            or 'goal011' in parts or any(p.startswith('goal011-') for p in parts))


def _files(scope, max_files=100_000):
    result, skipped = {}, []
    for spec in scope:
        directory = _safe(spec['path'])
        if _excluded(directory): raise RestAuditError('excluded directory cannot be an exposure search root')
        if not directory.exists():
            skipped.append({'path': str(directory), 'reason': 'SCOPE_DIRECTORY_ABSENT'}); continue
        if not directory.is_dir(): raise RestAuditError('search scope must be a directory')
        for base, dirs, files in os.walk(directory, followlinks=False):
            for name in list(dirs):
                p = Path(base) / name
                if p.is_symlink() or _excluded(p):
                    dirs.remove(name)
                    skipped.append({'path': str(p), 'reason': 'SYMLINK_OR_EXCLUDED_DIRECTORY'})
            for name in files:
                p = Path(base) / name
                if p.is_symlink() or _excluded(p) or not p.is_file():
                    skipped.append({'path': str(p), 'reason': 'SYMLINK_OR_EXCLUDED_FILE'}); continue
                stat = p.stat()
                result[str(p)] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
                if len(result) > max_files:
                    raise RestAuditError('candidate file inventory resource limit; scope not certified')
    return dict(sorted(result.items())), sorted(skipped, key=lambda r: (r['path'], r['reason']))


def _classification(path, match_type):
    text = str(path).lower()
    if 'human' in text: kind = 'HUMAN_STUDY_CONFIGURATION_OR_RECORD_NOT_PROOF_OF_COMPLETED_STUDY'
    elif '/figures/' in text or '/examples/' in text or '/capture_rebuild' in text: kind = 'FIGURE_OR_RENDER_ARTIFACT'
    elif '/rebuild_' in text and Path(path).name in ('voxels.npy', 'voxels.json', 'actions.json', 'plan.json'):
        kind = 'PRIOR_BUILDER_OUTPUT_ARTIFACT_NOT_REVALIDATED'
    elif any(x in text for x in ('/analysis/', '/evaluation/', '/results', 'scores', 'scored_', 'metrics')):
        kind = 'QUALITY_ANALYSIS_OR_RESULT_REFERENCE'
    elif '/description_' in text or '/datasets/' in text: kind = 'SOURCE_PREPARATION_OR_DESCRIPTION'
    else: kind = 'DOCUMENT_OR_MANIFEST_REFERENCE'
    return {'kind': kind, 'identity_match': match_type,
            'schema_v5_named_in_path': 'v5' in text,
            'references_exact_v4_scene_identifier': match_type in ('EXACT_V4_SCENE_ID', 'EXACT_V4_SCENE_PATH'),
            'original_artifact_source_chain_verified': False,
            'model_success_or_completed_human_experiment_inferred': False}


def _matches(data, path, ids, gt_hashes):
    result = set(); text = str(path).replace('\\', '/').encode()
    for token in TOKEN.findall(text + b'\n' + data):
        s = token.decode('ascii').replace('\\', '/')
        building = s[-12:]
        sid = 'buildings_100_v4_' + building
        if sid not in ids: continue
        kind = ('EXACT_V4_SCENE_ID' if s.startswith('buildings_100_v4_') else
                'EXACT_V4_SCENE_PATH' if s.startswith('buildings_100_v4/') else
                'OTHER_DATASET_VERSION_BASENAME_ONLY' if s.startswith('buildings_100_v') else 'AMBIGUOUS_BASENAME_ONLY')
        result.add((sid, kind))
    for digest, scene_ids in gt_hashes.items():
        if digest.encode('ascii') in data:
            for sid in scene_ids: result.add((sid, 'GT_CONTENT_HASH_REFERENCE_NOT_UNIQUE_SCENE_PROOF'))
    return result


def audit_exposures(scope, scene_rows, inventory, *, limits=RestLimits()):
    """Search IDs/hashes; export no text snippets, bodies, credentials or pixels."""
    before, excluded = _files(scope, limits.max_search_files); ids = {s['scene_id'] for s in scene_rows}
    gt_hashes = defaultdict(set)
    for row in scene_rows:
        for f in row['source_files']:
            if f['kind'] == 'GT': gt_hashes[f['sha256']].add(row['scene_id'])
    hits, unknown = [], []; scanned = 0; scanned_bytes = 0; started = time.process_time()
    for name, state in before.items():
        p = Path(name); text_file = p.suffix.lower() in TEXT_SUFFIXES
        path_matches = _matches(b'', p, ids, {})
        if not text_file and not path_matches:
            unknown.append({'path': name, 'bytes': state['bytes'], 'reason': 'BINARY_CONTENT_NOT_SEARCHED'}); continue
        if (scanned >= limits.max_search_files or scanned_bytes + state['bytes'] > limits.max_search_bytes
                or time.process_time() - started > limits.cpu_seconds
                or (text_file and state['bytes'] > limits.max_text_file_bytes)):
            unknown.append({'path': name, 'bytes': state['bytes'], 'reason': 'SEARCH_RESOURCE_LIMIT'}); continue
        found = set(path_matches); h = hashlib.sha256(); carry = b''
        try:
            _safe(p)
            with p.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
                    if text_file:
                        joined = carry + block; found |= _matches(joined, p, ids, gt_hashes); carry = joined[-128:]
            descriptor = {'bytes': state['bytes'], 'sha256': h.hexdigest()}
            if name in inventory and inventory[name] != descriptor: raise RestAuditError('source changed during exposure search')
            inventory[name] = descriptor
        except OSError:
            unknown.append({'path': name, 'bytes': state['bytes'], 'reason': 'READ_ERROR'}); continue
        scanned += 1; scanned_bytes += state['bytes']
        for sid, match_type in sorted(found):
            if len(hits) >= limits.max_evidence_rows:
                raise RestAuditError('exposure evidence limit; no truncated positive report permitted')
            hits.append({'scene_id': sid, 'path': name, **descriptor, **_classification(p, match_type)})
    after, after_excluded = _files(scope, limits.max_search_files)
    if before != after or excluded != after_excluded: raise RestAuditError('search inventory changed during audit')
    for omitted in unknown + excluded:
        omitted['evidence_status'] = 'NOT_VERIFIED'
    return {'search_roots': scope, 'candidate_files': len(before), 'files_content_or_path_hashed': scanned,
            'bytes_hashed': scanned_bytes, 'hits': hits, 'not_searched': unknown, 'excluded': excluded,
            'scope_listing_sha256': _digest(before), 'scope_listing_before_after_unchanged': True,
            'status': 'SEARCH_COMPLETED_WITH_EXPLICIT_LIMITATIONS',
            'truly_unseen_anywhere': 'NOT_VERIFIED',
            'absence_scope': 'Only listed roots and searchable representations; no global or inaccessible history claim.'}


def _description_request(path, provider, description, files, inventory, *, root, limits):
    """Bind saved request image selections without claiming independently logged sends.

    The sibling request convention comes from the frozen description generator;
    image names must resolve inside the GT scene and its already-hashed inventory.
    No raw prompts, description text or image pixels are returned.
    """
    request_path = _safe(path.with_name('description.request.json'), root)
    result = {'description_request_path': str(request_path), 'description_request_sha256': None,
              'request_view_count_status': 'NOT_VERIFIED_SAVED_REQUEST_ABSENT',
              'actual_description_request_view_count': None, 'request_images_ordered': [],
              'request_submission_independently_verified': False,
              'request_evidence_caveat': 'Saved request record and image bytes only; not independent transport or visibility proof.'}
    if not request_path.exists(): return result
    record = _record(request_path, inventory, max_bytes=limits.max_description_bytes)
    request = json.loads(request_path.read_text(encoding='utf-8'))
    gt_path = next(Path(p) for p, value in files.items() if value['kind'] == 'GT')
    scene_dir = gt_path.parent.parent
    if (not isinstance(request, dict) or not isinstance(description, dict)
            or request.get('building') != scene_dir.name or description.get('building') != scene_dir.name
            or request.get('provider') != provider or description.get('provider') != provider
            or not isinstance(request.get('model'), str) or not request['model']
            or request['model'] != description.get('model')):
        raise RestAuditError('saved description/request scene-provider-model binding differs')
    meta_path = next(Path(p) for p, value in files.items() if value['kind'] == 'scene_metadata')
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    views = metadata.get('views', [])
    if (not isinstance(views, list) or len(views) != 12
            or not all(isinstance(v, dict) and isinstance(v.get('path'), str) for v in views)
            or len({v['path'] for v in views}) != len(views)):
        raise RestAuditError('unique twelve metadata image references required')
    by_image = {}
    for index, view in enumerate(views):
        image_path = _relative(scene_dir, view['path'])
        bound = files.get(str(image_path))
        if bound is None or bound['kind'] != 'image_reference':
            raise RestAuditError('metadata image absent from frozen source inventory')
        by_image[view['path']] = {'meta_view_index': index, 'path': str(image_path),
                                'sha256': bound['sha256'], 'bytes': bound['bytes']}
    images = request.get('images')
    if (not isinstance(images, list) or not images or not all(isinstance(p, str) for p in images)
            or len(set(images)) != len(images)):
        raise RestAuditError('unique saved request image paths required')
    for image_path in images:
        _relative(scene_dir, image_path)
        if image_path not in by_image: raise RestAuditError('request image absent from frozen metadata')
    result.update(description_request_sha256=record['sha256'],
                  request_view_count_status='VERIFIED_SAVED_REQUEST_IMAGE_BINDINGS',
                  actual_description_request_view_count=len(images),
                  request_images_ordered=[dict(by_image[p], request_order=i) for i, p in enumerate(images)],
                  description_request_model_id=request['model'])
    return result


def audit_rest30(package, *, gate, root=ROOT, selected_description_provider='openai', limits=RestLimits()):
    _gate(gate)
    root = _safe(root)
    if gate['input_kind'] == 'verified_sources' and root != ROOT:
        raise RestAuditError('actual audit requires existing repository root')
    if selected_description_provider not in PROVIDERS: raise RestAuditError('fixed description provider required')
    package_before = _digest(package); cohort = package['cohort']; inventory = {}
    identity = package['full_manifest']['identity']; cp = _safe(identity['cohort_path'])
    # Validate all traversal roots before opening even the cohort file.
    scope = search_scope(package, root=root)
    descriptor = _record(cp, inventory)
    if descriptor['sha256'] != identity['cohort_file_sha256'] or json.loads(cp.read_text()) != cohort:
        raise RestAuditError('cohort does not match frozen full manifest')
    if _digest({k: v for k, v in cohort.items() if k != 'cohort_sha256'}) != cohort['cohort_sha256']:
        raise RestAuditError('cohort content digest mismatch')
    rest = validate_split(cohort)
    if set(identity['evaluation_scene_ids']) != {s['scene_id'] for s in cohort['scenes'] if s['split'] == 'evaluation'}:
        raise RestAuditError('evaluation manifest disagrees with original split')
    known_eval = []
    for label, plan in sorted(package.get('plans', {}).items()):
        seen = sorted({r['scene_id'] for r in plan['identity']['schedule']} & {r['scene_id'] for r in rest})
        known_eval.append({'source_label': label, 'rest_scene_ids': seen, 'rest_scenes_in_fixed_plan': len(seen)})
    sources = []
    for row in rest:
        if _digest(row['source_files']) != row['source_manifest_sha256']:
            raise RestAuditError('scene source manifest changed')
        kinds = Counter(s['kind'] for s in row['source_files'])
        if any(kinds[k] != n for k, n in [('GT', 1), ('GT_bbox', 1), ('scene_metadata', 1), ('image_reference', 12)]):
            raise RestAuditError('expected GT/bbox/meta/12 frozen images')
        files = {}
        for source in row['source_files']:
            p = _relative(root, source['path'])
            if (_excluded(p) or source['kind'] not in ('GT', 'GT_bbox', 'scene_metadata', 'image_reference', 'saved_description')
                    or (source['kind'] == 'GT' and p.name != 'voxels.npy')
                    or (source['kind'] == 'GT_bbox' and p.name != 'bbox.json')
                    or (source['kind'] == 'scene_metadata' and p.name != 'meta.json')
                    or (source['kind'] == 'saved_description' and p.name != 'description.json')):
                raise RestAuditError('unexpected or excluded frozen source role/path')
            if str(p) in files: raise RestAuditError('duplicate frozen source file')
            files[str(p)] = {**_record(p, inventory, source, max_bytes=limits.max_source_file_bytes), 'kind': source['kind']}
        gt = next(s for s in row['source_files'] if s['kind'] == 'GT')
        bbox = next(s for s in row['source_files'] if s['kind'] == 'GT_bbox')
        images = [{k: s[k] for k in ('path', 'sha256', 'size_bytes')} for s in row['source_files'] if s['kind'] == 'image_reference']
        for provider in PROVIDERS:
            source = row['description_sources'].get(provider, {})
            item = {'scene_id': row['scene_id'], 'split': 'rest', 'size_rank_stratum': row['size_rank_stratum'],
                    'rank': row['rank'], 'description_provider': provider, 'selected_for_prospective_design': provider == selected_description_provider,
                    'scene_source_manifest_sha256': row['source_manifest_sha256'], 'cohort_file_sha256': descriptor['sha256'],
                    'gt_path': str(_relative(root, gt['path'])), 'gt_sha256': gt['sha256'],
                    'bbox_path': str(_relative(root, bbox['path'])), 'bbox_sha256': bbox['sha256'],
                    'saved_image_count': len(images), 'images': images, 'actual_description_request_view_count': None,
                    'request_view_count_status': 'NOT_VERIFIED_BY_REST_SOURCE_INVENTORY',
                    'source_status': 'NOT_AVAILABLE', 'input_path': None, 'input_sha256': None,
                    'input_characters': None, 'input_utf8_bytes': None, 'typed_tree_sha256': None,
                    'future_output_length_status': 'NOT_VERIFIED_NO_MODEL_OUTPUT_OR_PROVIDER_TOKENIZATION',
                    'future_context_fit_status': 'NOT_VERIFIED', 'future_output_completion_guaranteed': False,
                    'frozen_legacy_reference_actions_json_bytes': row.get('control', {}).get('reference_actions_json_bytes'),
                    'reference_length_caveat': 'Saved legacy CPU reference serialization, not future O prediction length or current-schema token estimate.',
                    'research_api_requests_sent': 0, 'token_count_api_requests': 0}
            if source.get('status') == 'prepared' and source.get('path'):
                p = _relative(root, source['path'])
                if str(p) not in files or files[str(p)]['kind'] != 'saved_description':
                    raise RestAuditError('description not included in frozen scene inventory')
                if files[str(p)]['sha256'] != source['sha256'] or files[str(p)]['bytes'] != source['size_bytes']:
                    raise RestAuditError('description descriptor disagrees with frozen source inventory')
                raw = p.read_bytes()
                if len(raw) > limits.max_description_bytes: raise RestAuditError('description exceeds frozen byte limit')
                text = raw.decode('utf-8'); _, proof = canonicalize_saved_text(text, limits.max_description_bytes)
                if source.get('typed_tree_equal') is not True or source.get('typed_tree_sha256') != proof['typed_tree_sha256']:
                    raise RestAuditError('saved description typed-tree binding differs')
                item.update(source_status='VERIFIED_FROZEN_O_SOURCE', input_path=str(p), input_sha256=source['sha256'],
                            input_characters=len(text), input_utf8_bytes=len(raw), typed_tree_sha256=proof['typed_tree_sha256'])
                item.update(_description_request(p, provider, json.loads(text), files, inventory,
                                                 root=root, limits=limits))
            sources.append(item)
    exposures = audit_exposures(scope, rest, inventory, limits=limits)
    by_scene = defaultdict(list)
    for evidence in exposures['hits']: by_scene[evidence['scene_id']].append(evidence)
    for item in sources:
        hits = by_scene[item['scene_id']]
        item['prior_exposure_evidence_sha256'] = _digest(hits)
        item['prior_exposure_evidence_location'] = 'REST30_FEASIBILITY.prior_exposure_audit.hits filtered by scene_id'
        item['prior_exposure_evidence_count'] = len(hits)
        item['exact_v4_prior_builder_artifact_references'] = sum(h['references_exact_v4_scene_identifier'] and h['kind'] == 'PRIOR_BUILDER_OUTPUT_ARTIFACT_NOT_REVALIDATED' for h in hits)
        item['ambiguous_identity_references'] = sum(not h['references_exact_v4_scene_identifier'] for h in hits)
        item['prior_use_status'] = 'EXPOSURE_REFERENCES_FOUND' if hits else 'NO_REFERENCE_FOUND_IN_SEARCHED_SCOPE'
        item['truly_unseen_anywhere'] = 'NOT_VERIFIED'
    verify_inventory(inventory)
    if _digest(package) != package_before: raise RestAuditError('input package mutated')
    selected = [r for r in sources if r['selected_for_prospective_design']]
    overlap = sorted({s for row in known_eval for s in row['rest_scene_ids']})
    complete_plans = {'parent160', 'original781', 'supplement3', 'supplement17'} <= set(package.get('plans', {}))
    summary = {'schema': 'goal011_rest30_readonly_feasibility_v1', 'analysis_label': 'POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC',
        'status': 'AUDIT_COMPLETE_READINESS_NOT_VERIFIED', 'independent_rest_buildings': 30,
        'evaluation_buildings': 60, 'smoke_buildings': 10, 'rest_per_size_quintile': {str(q): 6 for q in range(5)},
        'rest_evaluation_overlap': [], 'rest_smoke_overlap': [], 'split_replay': 'PASS',
        'description_rows': len(sources), 'selected_description_provider': selected_description_provider,
        'verified_selected_inputs': sum(x['source_status'] == 'VERIFIED_FROZEN_O_SOURCE' for x in selected),
        'provider_source_counts': {p: sum(x['description_provider'] == p and x['source_status'] == 'VERIFIED_FROZEN_O_SOURCE' for x in sources) for p in PROVIDERS},
        'verified_saved_request_image_binding_counts': {p: sum(x['description_provider'] == p and x['request_view_count_status'] == 'VERIFIED_SAVED_REQUEST_IMAGE_BINDINGS' for x in sources) for p in PROVIDERS},
        'frozen_goal005_010_plan_overlap': known_eval,
        'verified_fixed_plan_scope_complete': complete_plans,
        'not_evaluated_in_verified_fixed_plans': not overlap if complete_plans else None,
        'rest_scenes_found_in_verified_fixed_plans': overlap,
        'prior_exposure_audit': exposures,
        'truly_unseen_anywhere': 'NOT_VERIFIED', 'new_dataset_distribution': False,
        'cohort_already_gt_ranked_and_cpu_prepared': True,
        'future_protocol_status': 'UNAPPROVED_NOT_ENABLED', 'pricing_status': 'REQUIRES_CURRENT_PRICE_VERIFICATION',
        'output_length_feasibility': 'LOCAL_BYTES_CHARACTERS_ONLY; provider tokens and future completion NOT_VERIFIED',
        'source_inventory_sha256': _digest(inventory), 'source_inventory_file_count': len(inventory),
        'source_before_after_unchanged': True, 'source_inventory': inventory,
        'gate': {k: gate[k] for k in ('input_kind', 'question_register_sha256', 'analysis_plan_sha256', 'code_commit')},
        'resource_limits': asdict(limits), 'limitations': [
            'No reference found is bounded absence evidence, not proof of never being observed or used.',
            'Other dataset versions and bare building numbers are ambiguous until source identity is independently proved.',
            'A saved output or human-study record is exposure evidence, not proof of valid generation or completed human study.',
            'Binary figure/image content is not visually interpreted and inaccessible histories are not searched.',
            'Saved twelve image candidates do not prove a particular description request sent twelve views.',
            'Saved request image selections are source-bound records, not independent transport logs or proof of visible structure.',
            'Legacy reference bytes do not establish future provider context/output token fit.'],
        'research_api_requests_sent': 0, 'model_queries': 0, 'token_count_api_requests': 0, 'api_key_queries': 0,
        'source_writes': 0, 'future_approvals_ledgers_or_launchers_created': 0}
    return {'feasibility': summary, 'sources': sources, 'inventory': inventory}
