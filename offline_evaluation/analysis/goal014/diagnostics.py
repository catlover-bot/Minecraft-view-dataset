"""Fixed-gallery Goal014 contrasts from hash-bound normalized CLIP features.

No model loading, inference, outcome access, network or original-file writes.
The unchanged Goal013A adapter supplies float32 normalized features upstream.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from ..goal013a import scorer

PROVIDERS = ('anthropic', 'gemini', 'openai')
ATOL = RTOL = 1e-5


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _hash(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'explicit SHA256 required')
    return value


def _key(row):
    return row['scene_id'], row['description_provider']


def build_feature_metadata(score_rows, config, *, score_sha256, config_sha256,
                           source_binding_sha256, fixture=False):
    """Describe the exact ordered feature inputs without copying text or tokens."""
    require(isinstance(score_rows, list) and score_rows, 'planned score rows required')
    rows = sorted(score_rows, key=_key)
    require(len({_key(r) for r in rows}) == len(rows), 'duplicate source row')
    scenes = sorted({r['scene_id'] for r in rows})
    require(set(map(_key, rows)) == {(s, p) for s in scenes for p in PROVIDERS}, 'complete planned provider frame required')
    require(fixture or (len(rows) == 180 and len(scenes) == 60), 'production requires 180 rows and 60 scenes')
    require(scenes == config['planned_scene_ids'], 'source and config scene order differ')
    require(config['score_id'] == scorer.SCORE_ID and config['transform'] == scorer.TRANSFORM
            and config['tolerances'] == scorer.TOLERANCES, 'fixed scorer method differs')
    transform_sha = scorer.json_hash(scorer.TRANSFORM)
    require(config['preprocess_sha256'] == transform_sha, 'fixed transform hash differs')
    clip_root = Path(config['clip_source_root']) / 'clip'
    tokenizer_sha = scorer.json_hash({name: config['file_hashes'][str(clip_root/name)]['sha256']
                                      for name in ('simple_tokenizer.py', 'bpe_simple_vocab_16e6.txt.gz')})
    require(config['file_hashes'][config['weight_path']]['sha256'] == scorer.CHECKPOINT_SHA256,
            'fixed checkpoint hash differs')
    image_rows, text_rows = [], []
    for scene in scenes:
        members = [r for r in rows if r['scene_id'] == scene]
        images = members[0]['images_ordered']
        require(len(images) == 6 and len({im['path'] for im in images}) == 6, 'six distinct ordered source images required')
        require(all(r['images_ordered'] == images for r in members), 'provider image source/order mismatch')
        for view, im in enumerate(images):
            require(type(im['bytes']) is int and im['bytes'] > 0, 'image bytes required')
            image_rows.append({'scene_id': scene, 'view_index': view, 'actual_path': im['path'],
                               'path': im['path'], 'sha256': _hash(im['sha256']), 'bytes': im['bytes']})
    require(len({im['path'] for im in image_rows}) == len(image_rows), 'image path shared between scenes')
    for row in rows:
        require(row['score_id'] == scorer.SCORE_ID and row['binding_status'] == 'PASS'
                and row['source_binding_status'] == 'PASS', 'source/scorer gate differs')
        require(row['weight_sha256'] == scorer.CHECKPOINT_SHA256
                and row['transform_sha256'] == transform_sha and row['tokenizer_sha256'] == tokenizer_sha,
                'source model/transform/tokenizer hash mismatch')
        if row['score_status'] != 'COMPUTATION_VALID':
            require(row['score_status'] == 'MISSING_SUMMARY' and row['score'] is None
                    and row['summary_status'] != 'VALID_ORIGINAL_SUMMARY', 'unexpected invalid score row')
            continue
        token = row['token_diagnostics']
        require(row['summary_status'] == 'VALID_ORIGINAL_SUMMARY' and row['token_coverage_status'] == 'VERIFIED'
                and token['token_coverage_status'] == 'VERIFIED' and token['phase2_token_eligible'] is True,
                'valid original summary/token coverage required')
        require(token['prefix'] == scorer.PREFIX and token['context_length'] == 77
                and token['summary_sha256'] == row['summary_sha256'], 'summary/token identity mismatch')
        require(math.isfinite(row['score']) and len(row['view_scores']) == 6
                and all(math.isfinite(q) for q in row['view_scores']), 'finite original score and six views required')
        text_rows.append({'scene_id': row['scene_id'], 'description_provider': row['description_provider'],
                          'summary_sha256': _hash(row['summary_sha256']),
                          'description_source_path': row['description_source_path'],
                          'description_source_sha256': _hash(row['description_source_sha256']),
                          'O_input_id': row['O_input_id'], 'O_input_sha256': _hash(row['O_input_sha256']),
                          'prefixed_text_sha256': _hash(token['prefixed_text_sha256']),
                          'token_ids_sha256': _hash(token['token_ids_sha256'])})
    require(fixture or len(text_rows) == 178, 'production requires 178 nonempty summaries')
    return {'schema': 'goal014_normalized_feature_identity_v1', 'score_id': scorer.SCORE_ID,
            'device': 'cpu', 'dtype': 'float32', 'normalized': True, 'prefix': scorer.PREFIX,
            'weight_sha256': scorer.CHECKPOINT_SHA256, 'transform_sha256': transform_sha,
            'tokenizer_sha256': tokenizer_sha, 'source_file_hashes': config['file_hashes'],
            'installed_distributions': config['installed_distributions'],
            'score_sha256': _hash(score_sha256), 'config_sha256': _hash(config_sha256),
            'source_binding_sha256': _hash(source_binding_sha256),
            'candidate_scene_ids': scenes, 'image_rows': image_rows, 'text_rows': text_rows,
            'planned_row_keys': [{'scene_id': r['scene_id'], 'description_provider': r['description_provider']} for r in rows]}


def validate_features(values, expected_rows, *, fixture=False):
    values = np.asarray(values)
    require(values.dtype == np.float32 and values.ndim == 2 and values.shape[0] == expected_rows
            and values.shape[1] > 0, 'ordered float32 feature matrix shape mismatch')
    require(fixture or values.shape[1] == 512, 'ViT-B/32 feature dimension must be 512')
    require(np.isfinite(values).all(), 'nonfinite feature')
    norm_squared = np.sum(values**2, axis=1)
    require((norm_squared > 0).all(), 'zero norm feature')
    require(np.allclose(norm_squared, 1, atol=2e-5, rtol=2e-5), 'normalized features required; renormalization forbidden')
    return values


def contrast_row(candidate_scores, candidate_scene_ids, matched_scene_id):
    """Exact floating ties; original scene excluded from every other statistic."""
    values = [float(v) for v in candidate_scores]
    require(len(values) == len(candidate_scene_ids) >= 2 and len(set(candidate_scene_ids)) == len(values),
            'unique candidate scene identities required')
    require(all(math.isfinite(v) for v in values) and matched_scene_id in candidate_scene_ids,
            'finite candidate scores and matched source scene required')
    index = candidate_scene_ids.index(matched_scene_id)
    matched = values[index]
    other = [value for j, value in enumerate(values) if j != index]
    mean = math.fsum(other) / len(other)
    ties = sum(matched == value for value in other)
    wins = sum(matched > value for value in other)
    return {'matching_score': matched, 'nonmatching_mean': mean,
            'nonmatching_sd_population': math.sqrt(math.fsum((v-mean)**2 for v in other)/len(other)),
            'D_i': matched-mean, 'W_i': (wins+0.5*ties)/len(other), 'ties': ties,
            'wins': wins, 'losses': len(other)-wins-ties, 'nonmatching_candidates': len(other)}


def compute_diagnostics(image_features, text_features, metadata, score_rows, config, *,
                        score_sha256, config_sha256, source_binding_sha256, fixture=False):
    expected = build_feature_metadata(score_rows, config, score_sha256=score_sha256,
                                     config_sha256=config_sha256, source_binding_sha256=source_binding_sha256,
                                     fixture=fixture)
    require(metadata == expected, 'feature metadata/order/source/token/config mismatch')
    images = validate_features(image_features, len(metadata['image_rows']), fixture=fixture)
    texts = validate_features(text_features, len(metadata['text_rows']), fixture=fixture)
    require(images.shape[1] == texts.shape[1], 'image/text feature dimensions differ')
    scenes = metadata['candidate_scene_ids']
    source = {_key(row): row for row in score_rows}
    matrix = np.empty((len(texts), len(scenes)), dtype=np.float64)
    diagnostics = {}
    view_errors, mean_errors, scalar_errors = [], [], []
    for i, (text, identity) in enumerate(zip(texts, metadata['text_rows'])):
        # Preserve the original float32 elementwise reduction, per-view clamp,
        # and fsum mean6. BLAS matrix products may use different reductions.
        views = np.float32(2.5)*np.clip(np.sum(images*text[None, :], axis=1, dtype=np.float32), 0, None)
        for j in range(len(scenes)):
            matrix[i, j] = math.fsum(float(v) for v in views[6*j:6*j+6])/6
        original = source[_key(identity)]
        matched = scenes.index(identity['scene_id'])
        new_views = views[6*matched:6*matched+6].astype(np.float64)
        old_views = np.asarray(original['view_scores'], dtype=np.float64)
        require(np.allclose(new_views, old_views, atol=ATOL, rtol=RTOL), 'original per-view score replay mismatch')
        require(np.isclose(matrix[i, matched], original['score'], atol=ATOL, rtol=RTOL), 'original mean6 score replay mismatch')
        view_errors.extend(abs(new_views-old_views).tolist())
        mean_errors.append(abs(matrix[i, matched]-original['score']))
        # Independent float64 scalar cosine check over matched inputs.
        for image, q in zip(images[6*matched:6*matched+6], new_views):
            denom = math.sqrt(math.fsum(float(v)**2 for v in image))*math.sqrt(math.fsum(float(v)**2 for v in text))
            independent = 2.5*max(math.fsum(float(a)*float(b) for a, b in zip(image, text))/denom, 0.)
            require(math.isclose(independent, q, rel_tol=RTOL, abs_tol=ATOL), 'independent scalar replay mismatch')
            scalar_errors.append(abs(independent-q))
        diagnostics[_key(identity)] = {'scene_id': identity['scene_id'], 'description_provider': identity['description_provider'],
            'status': 'COMPUTATION_VALID', 'null_reason': None, 'matrix_row_index': i,
            'original_fixed_score': original['score'], 'matching_view_scores': new_views.tolist(),
            **contrast_row(matrix[i], scenes, identity['scene_id'])}
    rows = []
    for identity in metadata['planned_row_keys']:
        key = _key(identity)
        rows.append(diagnostics.get(key, {**identity, 'status': 'MISSING_SUMMARY',
            'null_reason': source[key]['null_reason'], 'matrix_row_index': None,
            'original_fixed_score': None, 'matching_view_scores': None,
            **{name: None for name in ('matching_score', 'nonmatching_mean', 'nonmatching_sd_population',
                                      'D_i', 'W_i', 'ties', 'wins', 'losses')},
            'nonmatching_candidates': len(scenes)-1}))
    return {'matrix': matrix, 'rows': rows, 'metadata': metadata,
            'replay': {'status': 'PASS', 'per_view_comparisons': len(view_errors),
                       'summary_mean_comparisons': len(mean_errors), 'atol': ATOL, 'rtol': RTOL,
                       'max_per_view_absolute_difference': max(view_errors),
                       'max_mean6_absolute_difference': max(mean_errors),
                       'max_independent_scalar_absolute_difference': max(scalar_errors),
                       'original_scores_modified': False, 'feature_pairs': len(texts)*len(images),
                       'candidate_gallery_fixed': True, 'tie_rule': 'exact floating equality; half credit'}}


def _new_output(path, fixture):
    path = Path(path).absolute()
    require(not path.exists() and not any(p.is_symlink() for p in (path, *path.parents)), 'exclusive nonsymlink output required')
    if not fixture:
        relative = path.relative_to('/root/minecraft-runs')
        require(len(relative.parts) >= 2 and relative.parts[0].startswith('goal014-'), 'new Goal014 run output required')
    return path


def save_numeric_array(path, values, *, fixture=False):
    """Exclusive numeric NPY persistence; Python object/pickle payloads forbidden."""
    path = _new_output(path, fixture)
    values = np.asarray(values)
    require(values.dtype.kind in 'fiu b'.replace(' ', '') and values.dtype.hasobject is False
            and np.isfinite(values).all(), 'finite numeric array required; no pickle')
    with path.open('xb') as stream:
        np.save(stream, values, allow_pickle=False)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size, 'shape': list(values.shape), 'dtype': str(values.dtype), 'allow_pickle': False}


def save_feature_bundle(directory, image_features, text_features, metadata, *, fixture=False):
    directory = Path(directory)
    images = validate_features(image_features, len(metadata['image_rows']), fixture=fixture)
    texts = validate_features(text_features, len(metadata['text_rows']), fixture=fixture)
    output = _new_output(directory/'FEATURE_MANIFEST.json', fixture)
    manifest = {'metadata': metadata,
                'images': save_numeric_array(directory/'IMAGE_FEATURES.npy', images, fixture=fixture),
                'texts': save_numeric_array(directory/'TEXT_FEATURES.npy', texts, fixture=fixture)}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return manifest


def load_feature_bundle(manifest_path, expected_metadata, *, fixture=False):
    manifest = json.loads(Path(manifest_path).read_text())
    require(manifest['metadata'] == expected_metadata, 'saved feature metadata mismatch')
    arrays = []
    for kind in ('images', 'texts'):
        desc = manifest[kind]; path = Path(desc['path'])
        require(path.is_file() and not path.is_symlink(), 'feature array missing or symlink')
        raw = path.read_bytes()
        require(len(raw) == desc['bytes'] and hashlib.sha256(raw).hexdigest() == desc['sha256'], 'feature bytes/hash mismatch')
        with path.open('rb') as stream:
            values = np.load(stream, allow_pickle=False)
        require(list(values.shape) == desc['shape'] and str(values.dtype) == desc['dtype']
                and desc['allow_pickle'] is False, 'feature array descriptor mismatch')
        arrays.append(validate_features(values, len(expected_metadata['image_rows' if kind == 'images' else 'text_rows']), fixture=fixture))
    return (*arrays, manifest['metadata'])
