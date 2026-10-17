"""Execute hash-bound original CLIPScore AST definitions for CPU parity.

No downloaded source is copied into this repository. Original MIT copyright and
license remain alongside the hash-bound source; extraction metadata records the
exact definitions. Unrelated reference-caption imports and the CLI are not run.
"""
from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np

from . import scorer as s

DEFINITIONS = ('CLIPCapDataset', 'CLIPImageDataset', 'extract_all_captions',
               'extract_all_images', 'get_clip_score')


def extract_reference(source_path, expected_sha256, namespace):
    """Compile unchanged selected top-level AST nodes, not reconstructed copies."""
    path = Path(source_path)
    s.require(s.file_hash(path) == expected_sha256, 'author source hash mismatch')
    raw = path.read_bytes(); tree = ast.parse(raw, filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in DEFINITIONS]
    s.require(len(nodes) == len(DEFINITIONS) and {n.name for n in nodes} == set(DEFINITIONS), 'reference definition set differs')
    extracted = ast.Module(body=nodes, type_ignores=[])
    ns = dict(namespace)
    exec(compile(extracted, str(path), 'exec'), ns)
    metadata = {
        'source_path': str(path), 'source_sha256': hashlib.sha256(raw).hexdigest(),
        'source_revision': s.CLIPSCORE_REVISION,
        'definitions': [{'name': node.name, 'first_line': node.lineno, 'last_line': node.end_lineno,
                         'ast_sha256': hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()} for node in nodes],
        'omitted_top_level_imports': [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))],
        'original_function_bodies_modified': False,
        'excluded_entrypoints': ['parse_args', 'get_refonlyclipscore', 'main', '__main__ block'],
    }
    return ns, metadata


def load_reference(session):
    from PIL import Image
    from packaging import version
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    s.require(version.parse(np.__version__) >= version.parse('1.21'), 'fixed NumPy >=1.21 reference branch required')
    path = str(Path(session.config['clipscore_source_path']))
    ns, metadata = extract_reference(path, session.files[path]['sha256'], {
        'torch': session.torch, 'clip': session.clip, 'Image': Image, 'np': np,
        'Compose': Compose, 'Resize': Resize, 'CenterCrop': CenterCrop,
        'ToTensor': ToTensor, 'Normalize': Normalize, 'version': version,
        'warnings': warnings, 'tqdm': SimpleNamespace(tqdm=lambda values: values),
    })
    original_images = ns['extract_all_images']
    original_captions = ns['extract_all_captions']
    calls = {'extract_all_images': 0, 'extract_all_captions': 0, 'get_clip_score': 0}

    def images(data, model, device, batch_size=8, num_workers=0):
        s.require(device == 'cpu' and 1 <= batch_size <= 8 and num_workers == 0, 'reference execution budget differs')
        calls['extract_all_images'] += 1
        return original_images(data, model, device, batch_size=batch_size, num_workers=0)

    def captions(data, model, device, batch_size=8, num_workers=0):
        s.require(device == 'cpu' and 1 <= batch_size <= 8 and num_workers == 0, 'reference execution budget differs')
        calls['extract_all_captions'] += 1
        return original_captions(data, model, device, batch_size=batch_size, num_workers=0)

    ns['extract_all_images'] = images
    ns['extract_all_captions'] = captions
    metadata['execution_overrides'] = {
        'DataLoader': 'batch_size=8 and num_workers=0 passed by wrappers; unchanged original extractors execute',
        'tqdm': 'progress display disabled; iterable unchanged',
        'numpy_branch': '>=1.21 explicit float32 L2 normalization',
        'unused_imports_not_required': ['sklearn', 'generation_eval_utils', 'reference-caption metrics and CLI imports'],
    }
    return ns, metadata, calls


def _comparison(left, right, atol, rtol):
    a = np.asarray(left); b = np.asarray(right)
    s.require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), 'parity array shape/finite gate failed')
    difference = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))) if a.size else 0.0
    return {'pass': bool(np.allclose(a, b, atol=atol, rtol=rtol)),
            'max_absolute_difference': difference, 'atol': atol, 'rtol': rtol, 'shape': list(a.shape)}


def independent_scalar_scores(image_features, text_features):
    """Separate float64 scalar arithmetic check; no adapter normalization call."""
    result = []
    for image, text in zip(image_features, text_features):
        norm_i = math.sqrt(math.fsum(float(v)**2 for v in image))
        norm_t = math.sqrt(math.fsum(float(v)**2 for v in text))
        s.require(norm_i > 0 and norm_t > 0, 'independent zero norm')
        cosine = math.fsum(float(a)*float(b) for a, b in zip(image, text))/(norm_i*norm_t)
        result.append(2.5*max(cosine, 0.0))
    return result


def run_reference_parity(session, images, summaries):
    """Real model execution, only after the root runner's committed-spec gate.

    Inputs are fixed ordered image paths and original summaries, never outcomes.
    A returned PASS requires execution of the original get_clip_score function.
    """
    s.require(1 <= len(images) == len(summaries) <= 8, 'reference parity requires 1..8 paired inputs')
    s.require(all(isinstance(x, str) and x.strip() for x in summaries), 'reference summaries must be nonempty')
    reference, extraction, calls = load_reference(session)
    token_checks = []
    transform_checks = []
    reference_caps = reference['CLIPCapDataset'](summaries)
    reference_images = reference['CLIPImageDataset'](images)
    for i, (path, summary) in enumerate(zip(images, summaries)):
        diagnostic = session.token_diagnostics(summary)
        tokens = reference_caps[i]['caption'].tolist()
        token_checks.append(tokens == diagnostic['token_ids'])
        transform_checks.append(_comparison(reference_images[i]['image'].numpy(), session.image_tensor(path).numpy(),
                                            s.TOLERANCES['preprocess_atol'], s.TOLERANCES['preprocess_rtol']))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        raw_images = reference['extract_all_images'](images, session.model, 'cpu')
        calls['get_clip_score'] += 1
        reference_mean, reference_per, reference_text = reference['get_clip_score'](session.model, raw_images, summaries, 'cpu')
    reference_norm_images = raw_images / np.sqrt(np.sum(raw_images**2, axis=1, keepdims=True))
    adapter_images = session.encode_images(images, batch_size=8, use_cache=False)
    adapter_text = session.encode_texts(summaries, batch_size=8, use_cache=False)
    adapter_per = np.float32(2.5)*np.clip(np.sum(adapter_images*adapter_text, axis=1), 0, None)
    f_atol = s.TOLERANCES['normalized_feature_atol']; f_rtol = s.TOLERANCES['normalized_feature_rtol']
    q_atol = s.TOLERANCES['score_atol']; q_rtol = s.TOLERANCES['score_rtol']
    checks = {
        'normalized_image_features': _comparison(reference_norm_images, adapter_images, f_atol, f_rtol),
        'normalized_text_features': _comparison(reference_text, adapter_text, f_atol, f_rtol),
        'per_image_q': _comparison(reference_per, adapter_per, q_atol, q_rtol),
        'independent_scalar_formula': _comparison(reference_per, independent_scalar_scores(raw_images, reference_text), q_atol, q_rtol),
        'batch_one_images': _comparison(adapter_images, session.encode_images(images, batch_size=1, use_cache=False), f_atol, f_rtol),
        'batch_one_texts': _comparison(adapter_text, session.encode_texts(summaries, batch_size=1, use_cache=False), f_atol, f_rtol),
    }
    first_image_cache = session.encode_images(images, batch_size=8, use_cache=True)
    first_text_cache = session.encode_texts(summaries, batch_size=8, use_cache=True)
    before = dict(session.cache_counts)
    second_image_cache = session.encode_images(images, batch_size=8, use_cache=True)
    second_text_cache = session.encode_texts(summaries, batch_size=8, use_cache=True)
    cache_no_new_encoding = all(session.cache_counts[k] == before[k] for k in ('image_encoded', 'text_encoded'))
    checks['cache_images'] = _comparison(adapter_images, first_image_cache, f_atol, f_rtol)
    checks['cache_texts'] = _comparison(adapter_text, first_text_cache, f_atol, f_rtol)
    checks['cache_replay_images'] = _comparison(first_image_cache, second_image_cache, 0, 0)
    checks['cache_replay_texts'] = _comparison(first_text_cache, second_text_cache, 0, 0)
    view_mean_check = None
    if len(adapter_per) == 6:
        view_mean_check = s.aggregate_six(adapter_per)['score'] == s.aggregate_six(adapter_per[::-1])['score']
    passed = all(token_checks) and all(x['pass'] for x in transform_checks) and all(x['pass'] for x in checks.values())
    passed = passed and cache_no_new_encoding and view_mean_check is not False and calls['get_clip_score'] == 1
    return {'status': 'PASS' if passed else 'REFERENCE_PARITY_FAILED',
            'real_reference_function_executed': True, 'original_function_calls': calls,
            'paired_inputs': len(images), 'score_id': s.SCORE_ID, 'dtype': 'float32', 'device': 'cpu',
            'token_rows_exact': token_checks, 'preprocess_checks': transform_checks, 'numeric_checks': checks,
            'cache_replay_without_encoding': cache_no_new_encoding, 'six_view_order_mean_exact': view_mean_check,
            'reference_mean': float(reference_mean), 'tolerances': dict(s.TOLERANCES),
            'source_extraction': extraction, 'reference_warning_categories': sorted({type(w.message).__name__ for w in caught}),
            'source_hashes': session.files, 'raw_summary_exported': False,
            'gpu_2021_fp16_bit_parity_claimed': False, 'human_quality_validity': 'NOT_VALIDATED'}
