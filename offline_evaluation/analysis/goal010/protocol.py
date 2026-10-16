"""Read-only description-request provenance and conservative visibility audit.

No baseline, metric, projection, raycasting, inference, network or key access is
performed. The caller supplies the independently verified Goal007/008 package
and holds the existing source locks and Python network blockade.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import resource
import time

from ...server_v2.cohort_v4 import freeze_ranked_split
from ...server_v2.generation import utc_now

ROOT = Path(__file__).resolve().parents[3]
PROVIDERS = ('openai', 'anthropic', 'gemini')
SOURCE_CODE = ('tools/generate_building_descriptions.py', 'tools/capture_one_building.py',
               'tools/capture_building_collection.py', 'offline_evaluation/server_v2/cohort_v4.py',
               'offline_evaluation/server_v2/data.py', 'offline_evaluation/server_v2/core.py')
CAMERA_FIELDS = ('x', 'y', 'z', 'actual_x', 'actual_y', 'actual_z', 'yaw', 'pitch',
                 'actual_yaw', 'actual_pitch', 'fov', 'width', 'height')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def regular(path, root):
    path, root = Path(path), Path(root).resolve()
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError('source must be a regular file inside its verified root')
    return path.resolve()


def relative_file(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('unsafe relative source path')
    return regular(Path(root) / path, root)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _text_hash(value):
    if not isinstance(value, str): raise ValueError('recorded prompt text required for hashing')
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _record(path, inventory):
    path = Path(path)
    descriptor = {'sha256': sha256(path), 'bytes': path.stat().st_size}
    previous = inventory.setdefault(str(path), descriptor)
    if previous != descriptor: raise ValueError('source changed during request audit')
    return descriptor


def verify_inventory(inventory):
    for name, descriptor in inventory.items():
        path = Path(name)
        if path.is_symlink() or not path.is_file() or path.stat().st_size != descriptor['bytes'] or sha256(path) != descriptor['sha256']:
            raise ValueError('request/image/source-code inventory changed')


def audit_selection(cohort):
    rows = cohort['scenes']
    checked = freeze_ranked_split(rows, seed=cohort['seed'])
    if (cohort['population_scenes'] != 100 or cohort['split_counts'] != {'evaluation': 60, 'smoke': 10, 'rest': 30}
            or cohort.get('selection_uses_model_results') is not False):
        raise ValueError('fixed 100-scene selection specification mismatch')
    expected = {r['scene_id']: r for r in checked['scenes']}
    for row in rows:
        for field in ('rank', 'size_rank_stratum', 'stratum_draw_order', 'split', 'gt_occupied'):
            if row[field] != expected[row['scene_id']][field]:
                raise ValueError('fixed size-stratified selection replay mismatch')
    return {'population_buildings': 100, 'independent_evaluation_buildings': 60,
        'split_counts': cohort['split_counts'], 'seed': cohort['seed'], 'rank_order': checked['rank_order'],
        'rng_implementation': checked['rng_implementation'], 'selection_uses_model_results': False,
        'per_rank_quintile': {'population': 20, 'evaluation': 12, 'smoke': 2, 'rest': 6},
        'replay_exact': True, 'scenes': [{k: row[k] for k in ('scene_id', 'rank', 'gt_occupied', 'size_rank_stratum', 'stratum_draw_order', 'split')} for row in rows]}


def _camera(view):
    result = {}
    for field in CAMERA_FIELDS:
        value = view.get(field)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError('nonfinite or nonnumeric saved camera metadata')
        result[field] = value
    return result


def bind_selected_slots(sources, slots):
    if len(slots) != 960 or {r['global_schedule_index'] for r in slots} != set(range(960)):
        raise ValueError('canonical fixed 960 selected slots required')
    index = {(r['scene_id'], r['description_model'], r['provider'], r['condition']): r for r in slots}
    if len(index) != 960: raise ValueError('duplicate selected slot identity')
    bindings = {}
    fields = ('global_schedule_index', 'attempt_id', 'original_attempt_id', 'selected_attempt_id',
              'source_attempt_id', 'source_plan_path', 'source_plan_identity', 'source_entry_hash',
              'source_gt_descriptor_hash', 'source_run', 'source_label', 'status', 'input_id', 'input_sha256')
    for source in sources:
        if source.get('O_J_typed_tree_equal') is not True: raise ValueError('O/J typed JSON tree binding mismatch')
        bound = {}
        for builder in ('openai', 'anthropic'):
            bound[builder] = {}
            for condition in ('O', 'J'):
                row = index[(source['scene_id'], source['description_provider'], builder, condition)]
                prefix = '' if condition == 'O' else 'j_'
                if (row['input_id'] != source[prefix + 'input_id'] or row['input_sha256'] != source[prefix + 'input_sha256']
                        or row['source_gt_descriptor_hash'] != source['gt_descriptor_sha256']
                        or row['gt_grid_sha256'] != source['gt_descriptor']['grid_sha256']):
                    raise ValueError('selected O/J slot input/GT binding mismatch')
                if condition == 'O':
                    for field in ('attempt_id', 'global_schedule_index', 'status', 'source_plan_identity'):
                        if row[field] != source['builder_observations'][builder][field]:
                            raise ValueError('selected O observation binding mismatch')
                bound[builder][condition] = {field: row.get(field) for field in fields}
        bindings[(source['scene_id'], source['description_provider'])] = bound
    return bindings


def audit_description(source, *, root, inventory, scene_cache):
    """Return allowlisted metadata only, never raw prompts/responses/base64."""
    path = regular(source['source_path'], root)
    if _record(path, inventory)['sha256'] != source['source_sha256'] or source['input_sha256'] != source['source_sha256']:
        raise ValueError('original O description hash binding mismatch')
    description = read_json(path)
    scene = source['scene_id']; provider = source['description_provider']
    if provider not in PROVIDERS: raise ValueError('unsupported description provider')
    # This sibling convention is verified in generate_building_descriptions.py;
    # the parent originates in the already verified description source map.
    request_path = relative_file(path.parent, 'description.request.json')
    request_descriptor = _record(request_path, inventory); request = read_json(request_path)
    building_dir = regular(source['gt_voxels_path'], root).parent.parent
    for field, expected in [('gt_voxels_path', source['gt_descriptor']['voxels_sha256']),
                            ('gt_bbox_path', source['gt_descriptor']['bbox_sha256'])]:
        if _record(regular(source[field], root), inventory)['sha256'] != expected:
            raise ValueError('GT source descriptor changed')
    if (request.get('building') != building_dir.name or description.get('building') != building_dir.name
            or request.get('provider') != provider or description.get('provider') != provider
            or request.get('model') != source['description_model_id'] or description.get('model') != source['description_model_id']):
        raise ValueError('description/request scene-provider-model identity mismatch')
    if scene not in scene_cache:
        from PIL import Image
        meta_path = relative_file(building_dir, 'meta.json')
        meta_descriptor = _record(meta_path, inventory); meta = read_json(meta_path)
        views = meta.get('views')
        if not isinstance(views, list) or not views or len({v['path'] for v in views}) != len(views):
            raise ValueError('unique recorded candidate views required')
        saved = []
        for number, view in enumerate(views):
            image_path = relative_file(building_dir, view['path'])
            descriptor = _record(image_path, inventory)
            with Image.open(image_path) as image:
                dimensions = list(image.size); image_format = image.format; image.verify()
            if image_format != 'PNG' or dimensions != [view.get('width'), view.get('height')]:
                raise ValueError('actual image dimensions/format differ from recorded view')
            saved.append({'scene_id': scene, 'meta_view_index': number, 'relative_image_path': view['path'],
                'image_path': str(image_path), 'image_sha256': descriptor['sha256'], 'image_bytes': descriptor['bytes'],
                'actual_image_width': dimensions[0], 'actual_image_height': dimensions[1],
                'scene_meta_created_at_utc': meta.get('created_at'), 'per_image_capture_time_utc': view.get('created_at'),
                **_camera(view)})
        scene_cache[scene] = {'meta_path': str(meta_path), 'meta_sha256': meta_descriptor['sha256'],
            'views': saved, 'generation_origin': meta.get('generation', {}).get('origin'),
            'storage_bbox': read_json(source['gt_bbox_path']), 'meta_bbox': meta.get('bbox'),
            'camera_planner': {k: meta.get('camera_planner', {}).get(k) for k in ('radius', 'target', 'fov')},
            'scene_meta_created_at_utc': meta.get('created_at')}
    cached = scene_cache[scene]
    images = request.get('images')
    if not isinstance(images, list) or not images or len(set(images)) != len(images) or not all(isinstance(x, str) for x in images):
        raise ValueError('unique actual request image paths required')
    saved_by_path = {view['relative_image_path']: view for view in cached['views']}
    sent = []
    for position, image_path in enumerate(images):
        relative_file(building_dir, image_path)
        if image_path not in saved_by_path: raise ValueError('request image absent from saved view metadata')
        sent.append(dict(saved_by_path[image_path], description_provider=provider, request_image_order=position,
                         request_sha256=request_descriptor['sha256']))
    parameters = {}
    for field in ('temperature', 'top_p', 'top_k', 'max_tokens', 'seed', 'llm_seed'):
        value = request.get(field)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError('numeric request setting required')
        parameters[field] = value
        parameters[field + '_recorded'] = field in request
    usage_path = path.with_name('llm_usage.json')
    usage_created = None
    if usage_path.is_file():
        _record(regular(usage_path, root), inventory)
        usage_created = read_json(usage_path).get('created_at')
    row = {'scene_id': scene, 'description_provider': provider, 'description_model': request['model'],
        'description_source_path': str(path), 'description_source_sha256': source['source_sha256'],
        'input_id': source['input_id'], 'input_sha256': source['input_sha256'], 'j_input_id': source['j_input_id'],
        'j_input_sha256': source['j_input_sha256'], 'typed_tree_sha256': source['typed_tree_sha256'],
        'gt_descriptor_sha256': source['gt_descriptor_sha256'], 'gt_voxels_sha256': source['gt_descriptor']['voxels_sha256'],
        'gt_bbox_sha256': source['gt_descriptor']['bbox_sha256'],
        'request_path': str(request_path), 'request_sha256': request_descriptor['sha256'],
        'request_record_created_at_utc': request.get('created_at'), 'description_record_created_at_utc': description.get('created_at'),
        'usage_record_created_at_utc': usage_created, 'api_request_start_utc': None, 'api_request_end_utc': None,
        'timestamp_interpretation': 'saved record timestamps; actual submission/start/end timestamps are not present',
        'prompt_sha256': _text_hash(request.get('prompt')), 'system_prompt_sha256': _text_hash(request.get('system_prompt')),
        'actual_request_image_count': len(sent), 'saved_candidate_view_count': len(cached['views']),
        'meta_path': cached['meta_path'], 'meta_sha256': cached['meta_sha256'], **parameters,
        'actual_image_hashes_ordered': [v['image_sha256'] for v in sent],
        'actual_image_paths_ordered': [v['image_path'] for v in sent],
        'actual_meta_view_indices_ordered': [v['meta_view_index'] for v in sent],
        'builder_observations': source['builder_observations']}
    return row, sent


def _write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')


def _write_csv(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n'); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True, allow_nan=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def audit_protocol(package, destination, *, root=ROOT):
    """Phase0 only. Call within root's verified-source locks/network blockade."""
    started = time.monotonic(); out = Path(destination)
    if out.exists(): raise FileExistsError('new protocol audit destination required')
    sources, cohort = package['descriptions'], package['cohort']
    selection = audit_selection(cohort)
    evaluation = {row['scene_id'] for row in cohort['scenes'] if row['split'] == 'evaluation'}
    if len(sources) != 180 or {(r['scene_id'], r['description_provider']) for r in sources} != {(s, p) for s in evaluation for p in PROVIDERS}:
        raise ValueError('all fixed 60-by-3 source-bound descriptions required')
    selected_bindings = bind_selected_slots(sources, package['slots'])
    inventory, scene_cache, requests, image_rows = {}, {}, [], []
    for name in SOURCE_CODE: _record(relative_file(root, name), inventory)
    for source in sources:
        row, images = audit_description(source, root=root, inventory=inventory, scene_cache=scene_cache)
        row['selected_builder_slot_bindings'] = selected_bindings[(source['scene_id'], source['description_provider'])]
        requests.append(row); image_rows.extend(images)
    by_scene = {scene: [row for row in requests if row['scene_id'] == scene] for scene in evaluation}
    image_equal = all(len({tuple(row['actual_image_hashes_ordered']) for row in rows}) == 1 for rows in by_scene.values())
    prompt_equal = all(len({(row['prompt_sha256'], row['system_prompt_sha256']) for row in rows}) == 1 for rows in by_scene.values())
    summary = {'status': 'PASS', 'analysis_label': 'CPU_DIAGNOSTIC', 'request_records': len(requests),
        'independent_buildings': len(evaluation), 'request_image_references': len(image_rows),
        'unique_saved_images': sum(len(row['views']) for row in scene_cache.values()),
        'actual_request_image_count_distribution': dict(Counter(str(r['actual_request_image_count']) for r in requests)),
        'saved_candidate_view_count_distribution': dict(Counter(str(r['saved_candidate_view_count']) for r in requests)),
        'all_three_providers_share_ordered_image_bytes_per_scene': image_equal,
        'all_three_providers_share_prompt_hashes_per_scene': prompt_equal,
        'provider_settings': {p: {'models': sorted({r['description_model'] for r in requests if r['description_provider'] == p}),
             **{field: sorted({r[field] for r in requests if r['description_provider'] == p}, key=lambda x: str(x))
                for field in ('temperature', 'max_tokens', 'top_p', 'seed', 'llm_seed')},
             'request_record_created_at_min_utc': min(r['request_record_created_at_utc'] for r in requests if r['description_provider'] == p),
             'request_record_created_at_max_utc': max(r['request_record_created_at_utc'] for r in requests if r['description_provider'] == p)} for p in PROVIDERS},
        'causal_model_ability_comparison_supported': False,
        'confounds': ['description-generation model/version and provider', 'recorded output-token budgets may differ',
                     'description generation occurred at different recorded times', 'single saved output, no controlled decoding replication'],
        'source_inventory_files': len(inventory), 'source_inventory_unchanged': True,
        'research_api_requests_sent': 0, 'api_key_queries': 0, 'token_count_api_requests': 0}
    verify_inventory(inventory)
    out.mkdir(parents=True, exist_ok=False)
    _write_json(out / 'view_and_generation_request_audit.json', {'summary': summary, 'requests': requests,
        'scene_metadata': scene_cache, 'selection_audit': selection})
    _write_csv(out / 'view_and_generation_request_audit.csv', requests)
    _write_csv(out / 'image_request_view_audit.csv', image_rows)
    _write_json(out / 'SOURCE_BINDINGS.json', {'status': 'PASS', 'inventory': inventory,
        'bindings': [{k: r[k] for k in ('scene_id', 'description_provider', 'description_source_path', 'description_source_sha256',
            'input_id', 'input_sha256', 'j_input_id', 'j_input_sha256', 'gt_descriptor_sha256', 'gt_voxels_sha256', 'gt_bbox_sha256',
            'request_path', 'request_sha256', 'meta_path', 'meta_sha256', 'actual_image_paths_ordered', 'actual_image_hashes_ordered', 'builder_observations', 'selected_builder_slot_bindings')} for r in requests]})
    settings = '\n'.join(f"- {p}: models={v['models']}, temperature={v['temperature']}, max_tokens={v['max_tokens']}; recorded UTC {v['request_record_created_at_min_utc']}–{v['request_record_created_at_max_utc']}" for p, v in summary['provider_settings'].items())
    text = f"""# Goal010 プロトコル監査（CPU_DIAGNOSTIC）

180 request記録と実参照画像を全数照合した。入力画像枚数分布は {summary['actual_request_image_count_distribution']}、保存候補view数分布は {summary['saved_candidate_view_count_distribution']}。meta.viewsは保存候補であり、送信入力はrequest.imagesで特定した。同一sceneの3 providerで画像bytes・順序の一致={image_equal}、prompt hashes一致={prompt_equal}。

{settings}

created_atは保存記録のUTCで、実API送信開始・終了時刻は未記録。producerは応答取得後にdescription/request記録を保存する。記述モデルversion、max_tokens、実施時期等の設定込みの観測であり、モデル固有能力の因果比較を支持しない。

100建築をGT occupied数とscene IDで順位付けし、5層（各20建築）内の固定乱択で評価12・smoke2・残り6を選んだ。seed={cohort['seed']}、総数60/10/30の再生が一致。独立評価単位は60建築であり、180記述・360Builder観測ではない。

Oは保存JSON説明レコード、Jは同じtyped JSON treeの再シリアライズである。P/Eの実入力形式・Builder sampling・strict evaluatorはrootの共通source auditと固定manifest/codeに束縛する。meta.generation.origin、storage bbox、camera positionsは別概念として保存し、occupied bboxや光学中心の代用にしない。GTの読み込みは既存YXZ→XYZ readerを使用する。

保存camera yaw/pitch/FOVはplanned/requested metadataであり、実際の投影を独立検証した証拠ではない。actual XYZがあってもactual yaw/pitch、カメラ光学中心・eye offset、effective projection、深度/遮蔽の画像対応を保証しない。可視性判断はPhase4へ分離し、未検証なら割合を作らない。

元request・prompt・response本文やbase64、credentialsは出力していない。出典path/hash、設定のallowlist、画像の実寸法、カメラ記録だけを保存。研究API送信・キー照会・token-count APIは0。
"""
    with (out / 'PROTOCOL_AUDIT.md').open('x', encoding='utf-8') as stream: stream.write(text)
    products = {p.name: {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in sorted(out.iterdir())}
    result = dict(summary, version='goal010_protocol_v1', created_at_utc=utc_now(), products=products,
        performance={'seconds': time.monotonic() - started, 'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    _write_json(out / 'PROTOCOL_MANIFEST.json', result)
    return result


def visibility_feasibility(protocol_manifest_path, destination):
    """Phase4: audit measured metadata only; never invent visible percentages."""
    manifest_path = Path(protocol_manifest_path); manifest = read_json(manifest_path)
    if manifest.get('version') != 'goal010_protocol_v1' or manifest.get('status') != 'PASS':
        raise ValueError('verified complete Phase0 protocol audit required')
    for name, descriptor in manifest['products'].items():
        path = relative_file(manifest_path.parent, name)
        if sha256(path) != descriptor['sha256'] or path.stat().st_size != descriptor['bytes']:
            raise ValueError('frozen protocol output changed')
    bindings = read_json(manifest_path.parent / 'SOURCE_BINDINGS.json'); verify_inventory(bindings['inventory'])
    audit = read_json(manifest_path.parent / 'view_and_generation_request_audit.json')
    saved = [view for scene in audit['scene_metadata'].values() for view in scene['views']]
    out = Path(destination)
    if out.exists(): raise FileExistsError('fresh visibility output directory required')
    result = {'status': 'NOT_VALIDATED_NO_VISIBILITY_CONCLUSION', 'analysis_label': 'CPU_DIAGNOSTIC',
        'protocol_manifest_path': str(manifest_path.resolve()), 'protocol_manifest_sha256': sha256(manifest_path),
        'source_binding': 'PASS', 'request_records': manifest['request_records'],
        'unique_saved_views': len(saved), 'actual_request_image_count_distribution': manifest['actual_request_image_count_distribution'],
        'recorded_camera_field_counts': {field: sum(v.get(field) is not None for v in saved) for field in CAMERA_FIELDS},
        'recorded_fov_values': sorted({v['fov'] for v in saved if v['fov'] is not None}),
        'actual_image_resolution_counts': dict(Counter(f"{v['actual_image_width']}x{v['actual_image_height']}" for v in saved)),
        'independent_pixel_projection_validation_performed': False, 'independent_occlusion_validation_performed': False,
        'reasons': ['Saved yaw/pitch and FOV describe planned/requested poses; effective rendered projection is not recorded.',
            'Saved actual XYZ is not a validated optical camera center or eye-height convention.',
            'Actual yaw/pitch, effective intrinsics and synchronized depth/voxel correspondence are not validated against these RGB images.',
            'No independently verified pixel reprojection/occlusion residuals are available; full-cube raycasting would assume block geometry/transparency.'],
        'visible_voxel_fraction': None, 'invisible_voxel_fraction': None, 'six_vs_twelve_visibility_difference': None,
        'claims_not_supported': ['invisible regions are the main reconstruction-error cause', 'six-view/12-view voxel visibility percentages',
            'visible surface material determines all interior occupied voxels or air cells'],
        'next_validation_required': ['Recover actual renderer projection, optical center and pose conventions.',
            'Verify projected GT surface landmarks against immutable RGB on a fixed sample with stated pixel residual tolerance.',
            'Independently validate occlusion and block shape/transparency before six-vs-twelve visible-surface comparisons.'],
        'research_api_requests_sent': 0, 'api_key_queries': 0, 'image_modifications': 0, 'synthetic_images_presented_as_observations': 0}
    out.mkdir(parents=True, exist_ok=False)
    _write_json(out / 'VISIBILITY_FEASIBILITY.json', result)
    text = """# 可視性の検証可能性

判定：NOT_VALIDATED_NO_VISIBILITY_CONCLUSION。

実requestの画像参照・画像hash・保存viewの角度/解像度/座標は監査できた。しかし保存FOVは撮影計画のrequested値であり、実画像の投影パラメータと同じだと実証されていない。actual XYZも、光学中心や目の高さ、actual yaw/pitchまで確定する記録ではない。元の撮影コードはRGB VideoProducerを使い、同期深度を保存していない。

したがってピクセル投影・遮蔽・非立方体block形状/透過性を実RGBと独立照合するまで、可視/不可視voxel割合や6対12枚の可視率差を算出しない。未検証は可視率0を意味しない。『不可視領域が誤差の主因』とも結論しない。可視外表面、内部占有、空気セル、材質の推定可能性は別概念である。

次の最低条件は実カメラ規約の回収、固定サンプルのGT面ランドマークと元RGBの再投影誤差による検証、独立した遮蔽/透過性検証である。本Goalでは画像の生成・加工、API、モデル照会、人手実験を行っていない。
"""
    with (out / 'VISIBILITY_LIMITATIONS_JA.md').open('x', encoding='utf-8') as stream: stream.write(text)
    return result
