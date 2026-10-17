"""Read-only summary/image source binding; no model, GT grid or outcome loading.

Existing manifests contain provenance for multiple research stages. This module
uses only original O input identity, saved requests and image descriptors, and
returns an explicit allowlist. It never scores, imports a model, queries keys,
downloads data, or opens a Builder result, GT voxel array or GT bbox file.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[3]
LABEL = 'POST_HOC_EXPLORATORY / GT_INDEPENDENT_SUMMARY_ALIGNMENT'
PROVIDERS = ('openai', 'anthropic', 'gemini')
MODELS = {'openai': 'gpt-5-mini', 'anthropic': 'claude-haiku-4-5-20251001', 'gemini': 'gemini-3.1-pro-preview'}
DESCRIPTION_DIRS = {'openai': 'description_openai_gpt_5_mini',
                    'anthropic': 'description_anthropic_claude_haiku_4_5_20251001',
                    'gemini': 'description_gemini_gemini_3_1_pro_preview'}
REQUEST_FIELDS = ('max_tokens', 'temperature', 'top_p', 'top_k', 'seed', 'llm_seed')
MAX_JSON_BYTES = 32 * 1024**2
MAX_SOURCE_BYTES = 128 * 1024**2
ANCHORS = {
    'g7_manifest': ('reports/server_only/goal007/MANIFEST.json', 'ef05c2ed93247802e9a68a00d700d39a8934d6231c7787315ab37b01d2467714'),
    'g7_gate': ('reports/server_only/goal007/audit/audit_gate.json', 'b7b40449e83095ca871a01cda51e8277d45ff238eaa571fd89dae087eff2c4c7'),
    'g8_manifest': ('reports/server_only/goal008/MANIFEST.json', '20e4433b3a56132f6031cd4ac08da14566f69795ad412a47d174503b0c053df6'),
    'g8_gate': ('reports/server_only/goal008/audit/source_gate.json', '9a9c866e3ef4595ec578338606c888125942c53a683b339e29bc230f352715e4'),
    'source_map': ('reports/server_only/goal008/phase1/source_map_180.csv', '5ca5bc10d3f92ea7cbcda3032eb5d7ba2dea762b1330b90e602e139e8cd775ce'),
    'protocol_manifest': ('reports/server_only/goal010/analysis/phase0_protocol/PROTOCOL_MANIFEST.json', '42d5285b0d8ed03c5ba87d7f4cfd7232973e781e49881a15193f925bc5c7b6e2'),
    'request_bindings': ('reports/server_only/goal010/analysis/phase0_protocol/SOURCE_BINDINGS.json', 'cf6f1002328a2665cff12355bb4e7f3fee5ef105e820c504b49e870083caf3d7'),
    'request_audit': ('reports/server_only/goal010/analysis/phase0_protocol/view_and_generation_request_audit.json', '34aa34534ded18be41a09a6b4e37f3194f361a5b9f4605eea373d59973b81987'),
    'g11_manifest': ('reports/server_only/goal011/MANIFEST.json', '0eedbb8ba74cd0fcee589ce5aa79d5a6940e8c25a3015bd1964a5e49d1b1dec7'),
    'g11_source_receipt': ('reports/server_only/goal011/receipts/0_source_RECEIPT.json', '11ed46deaf938a84c814e75fb3627367c1f90f9826eca802218cbe07215a0eaf'),
    'g12_sources': ('reports/server_only/goal012/SOURCES.json', '6f45e2225981039be13670e5848c3a461e8c10d08c741afacfc2cb03cadd2def'),
    'g12_integrity': ('reports/server_only/goal012/INTEGRITY.json', '7f4283b21acf4200358a070c7bbe52f1879013c0148dbc5d7fbb913db2c37292'),
    'g12_rq_gate': ('reports/server_only/goal012/RQ_EVIDENCE_GATES.md', '5677773a4c1df5b8a2144e36a51bc8ee5c7be5a18e901870c5fc4e03515160ac'),
    'g12_decision': ('reports/server_only/goal012/EXPERIMENT_DECISION_JA.md', 'a6053782b92a74502849629d59c763c3b2725ba39a3581a03968b05203c59d01'),
}
FULL_MANIFEST = Path('/root/minecraft-runs/goal005-batch1-20261013T213420Z/full_manifest.json')
FULL_MANIFEST_SHA = 'ad8f5d130351a1027ea9b3929bec357d3c2fd7ba6f5810993bcbaa0b901b4116'
GOAL011_GATE_SHA = 'c3868fc8391f2ddd230c0d18ccba048d980a9ed4a3561caa49cd4ec1526fc773'
BLOCKED_PARTS = frozenset(('.git', '.aws', '.ssh', '.codex', '.agents', 'gt'))


class SourceBindingError(ValueError):
    """An immutable binding failed; stop without substitutes or partial scores."""


def require(condition, message):
    if not condition: raise SourceBindingError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def frozen_object_hash(value):
    """Exact server_v2.generation.object_hash format, without importing its code."""
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode('utf-8')
    return digest(raw)


def strict_json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'duplicate JSON key')
            value[key] = item
        return value
    def constant(value):
        raise SourceBindingError('non-finite JSON value')
    def finite_float(value):
        number = float(value)
        require(math.isfinite(number), 'non-finite JSON value')
        return number
    try:
        return json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant, parse_float=finite_float)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise SourceBindingError('invalid source JSON') from exc


def safe(path, root=None):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'absolute source path without traversal required')
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink source rejected')
    require(not any(p in BLOCKED_PARTS or p.startswith('.env') for p in path.parts), 'excluded source path')
    if root is not None: require(path.is_relative_to(root), 'source path escaped bound root')
    return path


def relative(root, name):
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts, 'unsafe relative image/source path')
    return safe(root/path, root)


def record(path, inventory, expected=None):
    path = safe(path)
    require(path.is_file(), 'bound source missing: '+str(path))
    size = path.stat().st_size
    require(size <= MAX_SOURCE_BYTES, 'source file byte limit')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): h.update(block)
    actual = {'bytes': size, 'sha256': h.hexdigest()}
    if expected is not None:
        require(actual['sha256'] == expected['sha256'], 'source hash changed: '+str(path))
        if expected.get('bytes') is not None: require(size == expected['bytes'], 'source byte count changed')
    require(str(path) not in inventory or inventory[str(path)] == actual, 'source changed during binding')
    inventory[str(path)] = actual
    return actual


def read_json(path):
    path = safe(path)
    require(path.stat().st_size <= MAX_JSON_BYTES, 'JSON byte limit')
    return strict_json(path.read_bytes())


def verify_inventory(inventory):
    check = {}
    for name, expected in inventory.items(): record(Path(name), check, expected)


def summary_metadata(description):
    present = 'summary' in description
    value = description.get('summary')
    kind = ('missing' if not present else 'null' if value is None else 'string' if isinstance(value, str)
            else 'boolean' if type(value) is bool else 'number' if type(value) in (int, float)
            else 'array' if isinstance(value, list) else 'object')
    valid = isinstance(value, str) and bool(value.strip())
    status = ('VALID_ORIGINAL_SUMMARY' if valid else 'MISSING_SUMMARY_FIELD' if not present
              else 'MISSING_EMPTY_SUMMARY' if isinstance(value, str) else 'MISSING_NON_STRING_SUMMARY')
    return {'summary_field_present': present, 'summary_json_type': kind, 'summary_status': status,
            'summary_sha256': digest(value.encode('utf-8')) if isinstance(value, str) else None,
            'summary_json_value_sha256': digest(json_bytes(value)) if present else None,
            'summary_characters': len(value) if isinstance(value, str) else None,
            'summary_utf8_bytes': len(value.encode('utf-8')) if isinstance(value, str) else None,
            'summary_text_modified': False, 'summary_text_exported': False}


def _indexed(rows, expected_keys, label):
    values = {(r['scene_id'], r['description_provider']): r for r in rows}
    require(len(values) == len(rows) and set(values) == expected_keys, label+' identity scope changed')
    return values


def description_source(root, scene, provider, name):
    """Allow only the frozen description role, never arbitrary repository JSON."""
    require(provider in PROVIDERS and re.fullmatch(r'buildings_100_v4_building_\d{3}', scene), 'unsupported source identity')
    building = scene.removeprefix('buildings_100_v4_')
    base = root/'datasets' if provider == 'gemini' else root/'outputs/i2t2b'
    expected = base/'buildings_100_v4'/building/DESCRIPTION_DIRS[provider]/'description.json'
    path = safe(name, root)
    require(path == expected, 'description outside frozen source allowlist')
    return path


def request_image_paths(root, scene, images):
    require(isinstance(images, list) and len(images) == 6 and all(isinstance(p, str) for p in images)
            and len(set(images)) == 6, 'six distinct actual request images required')
    scene_root = root/'datasets/buildings_100_v4'/scene.removeprefix('buildings_100_v4_')
    paths = [relative(scene_root, p) for p in images]
    require(all(p.parent == scene_root/'images' and p.suffix == '.png' for p in paths), 'image outside frozen PNG allowlist')
    return paths


def bind_records(source_rows, manifest, request_bindings, request_audit, *, root, inventory, expected_scenes=60, fixture=False):
    """Audit only. Result rows exclude raw text, GT fields and Builder outcomes."""
    require(fixture or (root == ROOT and expected_scenes == 60), 'actual source audit requires fixed repository and60 scenes')
    scenes = manifest['identity']['evaluation_scene_ids']
    require(len(scenes) == expected_scenes and len(set(scenes)) == expected_scenes, 'fixed scene scope changed')
    require(all(re.fullmatch(r'buildings_100_v4_building_\d{3}', s) for s in scenes), 'exact v4 IDs required')
    keys = {(s, p) for s in scenes for p in PROVIDERS}
    mapped = _indexed(source_rows, keys, 'source map')
    bound = _indexed(request_bindings['bindings'], keys, 'request bindings')
    requests = _indexed(request_audit['requests'], keys, 'request audit')
    require(request_bindings['status'] == 'PASS' and request_audit['summary']['status'] == 'PASS', 'prior request gates must pass')
    original = defaultdict(list)
    # Non-O input texts and all GT/output fields are deliberately never retained.
    for r in manifest['identity']['full_schedule']:
        if r['condition'] == 'O': original[(r['scene_id'], r['description_model'])].append(r)
    require(set(original) == keys, 'saved O input scope changed')
    old_inventory = request_bindings['inventory']
    results = []; shared = {}
    for key in sorted(keys):
        source = mapped[key]; binding = bound[key]; prior = requests[key]; scene, provider = key
        description_path = description_source(root, scene, provider, source['source_path'])
        for field in ('input_id', 'input_sha256'):
            require(source[field] == binding[field] == prior[field], 'saved input identity differs')
        require(str(description_path) == binding['description_source_path'] == prior['description_source_path'], 'description path binding differs')
        require(source['source_sha256'] == binding['description_source_sha256'] == prior['description_source_sha256'], 'description source digest differs')
        descriptor = record(description_path, inventory, old_inventory[str(description_path)])
        require(descriptor['sha256'] == source['source_sha256'] == source['input_sha256'], 'original O content hash differs')
        raw = description_path.read_bytes(); description = strict_json(raw)
        require(isinstance(description, dict), 'description must be a JSON object')
        rows = original[key]
        require(len(rows) == 2 and {r['provider'] for r in rows} == {'openai', 'anthropic'}, 'both Builder O input bindings required')
        for row in rows:
            require(row['input_id'] == source['input_id'] and row['input_sha256'] == source['input_sha256']
                    and row['source_sha256'] == source['source_sha256'] and row['input_text'].encode('utf-8') == raw,
                    'saved O text differs from original description')
        request_path = safe(binding['request_path'], root)
        require(request_path == description_path.with_name('description.request.json') == Path(prior['request_path']), 'request sibling binding differs')
        request_descriptor = record(request_path, inventory, old_inventory[str(request_path)])
        require(request_descriptor['sha256'] == binding['request_sha256'] == prior['request_sha256'], 'request hash differs')
        request = read_json(request_path)
        building = scene.removeprefix('buildings_100_v4_')
        require(description.get('building') == request.get('building') == building
                and description.get('provider') == request.get('provider') == provider
                and description.get('model') == request.get('model') == source['description_model_id'] == prior['description_model'] == MODELS[provider],
                'description/request scene-provider-model identity differs')
        for field in ('prompt', 'system_prompt'):
            require(isinstance(request.get(field), str) and digest(request[field].encode('utf-8')) == prior[field+'_sha256'], 'request prompt hash differs')
        for field in REQUEST_FIELDS:
            require(request.get(field) == prior.get(field) and (field in request) == prior.get(field+'_recorded'), 'recorded request settings differ')
        require(request.get('max_tokens') == (4096 if provider == 'gemini' else 1800), 'frozen provider output cap changed')
        require(request.get('created_at') == prior['request_record_created_at_utc']
                and description.get('created_at') == prior['description_record_created_at_utc'], 'saved creation-time binding differs')
        meta_path = safe(binding['meta_path'], root)
        require(meta_path == root/'datasets/buildings_100_v4'/building/'meta.json' == Path(prior['meta_path']), 'scene metadata path differs')
        meta_descriptor = record(meta_path, inventory, old_inventory[str(meta_path)])
        require(meta_descriptor['sha256'] == binding['meta_sha256'] == prior['meta_sha256'], 'scene metadata hash differs')
        meta = read_json(meta_path); views = meta.get('views')
        require(isinstance(views, list) and len(views) == 12 and all(isinstance(v, dict) and isinstance(v.get('path'), str) for v in views), 'twelve saved candidate views required')
        view_paths = [v['path'] for v in views]
        require(len(set(view_paths)) == 12, 'duplicate candidate image references')
        images = request.get('images')
        paths = request_image_paths(root, scene, images)
        require([str(p) for p in paths] == binding['actual_image_paths_ordered'] == prior['actual_image_paths_ordered'], 'request image order/path differs')
        require(all(p in view_paths for p in images), 'request image not in saved metadata')
        indices = [view_paths.index(p) for p in images]
        require(indices == prior['actual_meta_view_indices_ordered'] == [0, 2, 4, 6, 10, 11], 'request metadata-view order differs')
        image_rows = [dict(path=str(p), **record(p, inventory, old_inventory[str(p)])) for p in paths]
        hashes = [p['sha256'] for p in image_rows]
        require(hashes == binding['actual_image_hashes_ordered'] == prior['actual_image_hashes_ordered'], 'request image bytes differ')
        signature = ([str(p) for p in paths], hashes, prior['prompt_sha256'], prior['system_prompt_sha256'])
        require(scene not in shared or shared[scene] == signature, 'three providers do not share ordered image bytes/prompts')
        shared[scene] = signature
        results.append({'scene_id': scene, 'description_provider': provider, 'description_model_id': source['description_model_id'],
            'description_source_path': str(description_path), 'description_source_sha256': descriptor['sha256'], 'description_source_bytes': descriptor['bytes'],
            'O_input_id': source['input_id'], 'O_input_sha256': source['input_sha256'], 'O_text_bytes_equal_original_description': True,
            'O_input_utf8_bytes': len(raw), 'O_input_verified_builder_bindings': ['anthropic', 'openai'],
            'saved_request_path': str(request_path), 'saved_request_sha256': request_descriptor['sha256'], 'saved_request_bytes': request_descriptor['bytes'],
            'prompt_sha256': prior['prompt_sha256'], 'system_prompt_sha256': prior['system_prompt_sha256'],
            'description_created_at_utc': description.get('created_at'), 'request_record_created_at_utc': request.get('created_at'),
            'recorded_description_max_tokens': request['max_tokens'], 'recorded_description_temperature': request.get('temperature'),
            'recorded_description_top_p': request.get('top_p'), 'recorded_description_seed': request.get('seed'), 'recorded_description_llm_seed': request.get('llm_seed'),
            'recorded_description_top_k': request.get('top_k'),
            'description_request_fields_recorded': {field: field in request for field in REQUEST_FIELDS},
            'scene_metadata_path': str(meta_path), 'scene_metadata_sha256': meta_descriptor['sha256'], 'scene_metadata_bytes': meta_descriptor['bytes'],
            'saved_candidate_view_count': 12, 'actual_request_image_count': 6, 'actual_meta_view_indices_ordered': indices,
            'images_ordered': image_rows, **summary_metadata(description),
            'binding_status': 'PASS', 'score_status': 'COMPUTATION_NOT_PERFORMED', 'score': None,
            'transport_submission_independently_verified': False, 'GT_or_Builder_outcome_in_scoring_payload': False})
    return results


def audit_sources():
    """Production audit bound to prior committed sources; no scoring or writes."""
    started = time.monotonic(); inventory = {}; anchors = {}
    for label, (name, expected) in ANCHORS.items():
        p = safe(ROOT/name, ROOT); record(p, inventory, {'sha256': expected})
        anchors[label] = read_json(p) if p.suffix == '.json' else p.read_text(encoding='utf-8')
    g7 = anchors['g7_manifest']; g8 = anchors['g8_gate']; g8m = anchors['g8_manifest']
    require(anchors['g7_gate']['status'] == 'PASS' and anchors['g7_gate']['immutable_inventories_equal'] is True,
            'Goal007 prior integrity gate failed')
    require(g7['audit_gate_sha256'] == ANCHORS['g7_gate'][1] and g8['goal007_manifest_sha256'] == ANCHORS['g7_manifest'][1], 'Goal007 gate chain differs')
    require(g8['status'] == 'PASS' and g8['description_sources_count'] == 180 and g8m['source_gate_sha256'] == ANCHORS['g8_gate'][1], 'Goal008 gate chain differs')
    require(g8m['products']['phase1/source_map_180.csv']['sha256'] == ANCHORS['source_map'][1], 'source map manifest binding differs')
    require(anchors['g11_source_receipt']['status'] == 'PASS' and anchors['g11_source_receipt']['source_before_after_unchanged'] is True, 'Goal011 prior source gate failed')
    g11_receipt = anchors['g11_source_receipt']
    require(anchors['g11_manifest']['run_root'] == g11_receipt['run_root'], 'Goal011 canonical run binding differs')
    g11_gate_path = safe(Path(anchors['g11_manifest']['run_root'])/'0_source/SOURCE_GATE.json')
    g11_gate_descriptor = g11_receipt['products']['0_source/SOURCE_GATE.json']
    require(g11_gate_descriptor['sha256'] == GOAL011_GATE_SHA, 'Goal011 canonical source gate hash differs')
    record(g11_gate_path, inventory, g11_gate_descriptor)
    g11_gate = read_json(g11_gate_path)
    require(g11_gate['status'] == 'PASS' and g11_gate['phase0_binding'] == 'PASS'
            and g11_gate['definitions_frozen'] is True and g11_gate['input_kind'] == 'verified_sources',
            'Goal011 canonical source gate failed')
    require(g11_gate['goal007_manifest_sha256'] == ANCHORS['g7_manifest'][1]
            and g11_gate['source_package_sha256'] == g11_receipt['source_package_sha256'], 'Goal011 source gate chain differs')
    require(anchors['g12_integrity']['all_protected_original_bytes_unchanged'] is True, 'Goal012 prior integrity gate failed')
    require(anchors['g12_integrity']['status'] == 'PASS', 'Goal012 prior integrity status failed')
    require(anchors['protocol_manifest']['status'] == 'PASS' and anchors['protocol_manifest']['source_inventory_unchanged'] is True,
            'Goal010 prior protocol source gate failed')
    g12 = {s['source_id']: s for s in anchors['g12_sources']['internal_sources']}
    require(g12['g8_source_map']['sha256'] == ANCHORS['source_map'][1], 'Goal012 source map binding differs')
    for label, name in [('request_bindings', 'SOURCE_BINDINGS.json'), ('request_audit', 'view_and_generation_request_audit.json')]:
        require(anchors['protocol_manifest']['products'][name]['sha256'] == ANCHORS[label][1], 'protocol product binding differs')
    require(Path(g7['source_manifest']['path']) == FULL_MANIFEST and g7['source_manifest']['sha256'] == FULL_MANIFEST_SHA, 'fixed full manifest binding differs')
    record(FULL_MANIFEST, inventory, {'sha256': FULL_MANIFEST_SHA}); manifest = read_json(FULL_MANIFEST)
    require(frozen_object_hash(manifest['identity']) == manifest['identity_hash'] == g7['source_manifest']['identity_hash'], 'full manifest identity digest differs')
    source_rows = list(csv.DictReader(io.StringIO(anchors['source_map'])))
    rows = bind_records(source_rows, manifest, anchors['request_bindings'], anchors['request_audit'], root=ROOT, inventory=inventory)
    verify_inventory(inventory)
    receipt = {'schema': 'goal013_summary_source_binding_v1', 'analysis_label': LABEL, 'status': 'PASS',
        'description_records': len(rows), 'independent_buildings': len({r['scene_id'] for r in rows}),
        'description_provider_counts': dict(Counter(r['description_provider'] for r in rows)),
        'summary_status_counts': dict(Counter(r['summary_status'] for r in rows)),
        'unique_selected_image_paths': len({i['path'] for r in rows for i in r['images_ordered']}),
        'image_text_pair_capacity': len(rows)*6, 'all_three_providers_share_ordered_image_bytes_and_prompts': True,
        'source_inventory': inventory, 'source_inventory_sha256': digest(json_bytes(inventory)), 'source_inventory_file_count': len(inventory),
        'source_before_after_unchanged': True, 'anchor_sha256': {k: v[1] for k, v in ANCHORS.items()},
        'canonical_prior_gate_sha256': {str(g11_gate_path): GOAL011_GATE_SHA},
        'full_manifest_sha256': FULL_MANIFEST_SHA, 'scoring_data_allowlist': ['summary', 'images_ordered'],
        'summary_selection': 'original JSON string only; exact Unicode preserved; missing/non-string/whitespace-only is missing',
        'summary_json_value_hash_encoding': 'UTF-8 JSON parsed value, ensure_ascii=False, sort_keys=True, separators=(comma,colon); missing field has no value/hash',
        'score_status': 'COMPUTATION_NOT_PERFORMED', 'scored_descriptions': 0,
        'GT_grid_files_opened': 0, 'GT_bbox_files_opened': 0, 'Builder_outcome_files_opened': 0,
        'research_api_requests_sent': 0, 'model_queries': 0, 'token_count_api_requests': 0, 'api_key_queries': 0,
        'new_weights_or_packages_downloaded': 0, 'source_writes': 0, 'elapsed_seconds': time.monotonic()-started,
        'limitations': ['Only summary is a future scorer input, not the full O JSON record.',
                       'Saved requests bind six images among twelve candidates; no independent transport or visibility proof.',
                       'Description model/version, creation dates,1800 versus4096 output caps and sampling remain confounded.',
                       'The full manifest contains historical GT provenance; it is used solely to validate saved O text identity and is not passed to scoring.',
                       'This source PASS does not validate or enable a scorer and does not change the old RQ2 gate.']}
    return {'rows': rows, 'receipt': receipt}


def load_scoring_record(row, *, root=ROOT):
    """Return only exact summary and six image paths after revalidating hashes."""
    inventory = {}; scene = row['scene_id']; provider = row['description_provider']
    require(row['binding_status'] == 'PASS', 'unbound row must not be scored')
    path = description_source(root, scene, provider, row['description_source_path'])
    record(path, inventory, {'sha256': row['description_source_sha256'], 'bytes': row['description_source_bytes']})
    description = read_json(path); metadata = summary_metadata(description)
    require(all(metadata[k] == row[k] for k in metadata), 'summary metadata/hash changed')
    require(metadata['summary_status'] == 'VALID_ORIGINAL_SUMMARY', 'missing summary must not be scored')
    require(description.get('building') == scene.removeprefix('buildings_100_v4_') and description.get('provider') == provider
            and description.get('model') == row['description_model_id'] == MODELS[provider], 'scoring source identity differs')
    request_path = safe(row['saved_request_path'], root)
    require(request_path == path.with_name('description.request.json'), 'scoring request sibling differs')
    record(request_path, inventory, {'sha256': row['saved_request_sha256'], 'bytes': row['saved_request_bytes']})
    request = read_json(request_path)
    require(all(request.get(field) == description.get(field) for field in ('building', 'provider', 'model')), 'scoring request identity differs')
    expected_paths = request_image_paths(root, scene, request.get('images'))
    require(len(row['images_ordered']) == 6, 'scoring requires six saved images')
    require([str(p) for p in expected_paths] == [i['path'] for i in row['images_ordered']], 'scoring image order differs from saved request')
    paths = []
    for image in row['images_ordered']:
        p = safe(image['path'], root); record(p, inventory, image); paths.append(str(p))
    require(len(set(paths)) == 6, 'duplicate scoring image')
    verify_inventory(inventory)
    return {'summary': description['summary'], 'images_ordered': paths}


def write_outputs(result, destination):
    out = safe(Path(destination))
    require(out.is_relative_to('/tmp') or re.fullmatch(r'/root/minecraft-runs/goal013-\d{8}T\d{6}Z/phase0', str(out)),
            'new fixture directory or canonical Goal013 phase0 output directory required')
    out.mkdir(parents=True, exist_ok=False)
    products = {}
    for name, value in [('SOURCE_BINDING_180.json', result['rows']), ('SOURCE_BINDING_RECEIPT.json', result['receipt'])]:
        with (out/name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
        products[name] = {'sha256': digest((out/name).read_bytes()), 'bytes': (out/name).stat().st_size}
    name = 'SOURCE_BINDING_180.csv'
    with (out/name).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['rows'][0]), lineterminator='\n'); writer.writeheader()
        for row in result['rows']:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    products[name] = {'sha256': digest((out/name).read_bytes()), 'bytes': (out/name).stat().st_size}
    return products


if __name__ == '__main__':
    raise SystemExit('Direct execution is disabled. Use the committed Goal013 run_preflight entry point.')
