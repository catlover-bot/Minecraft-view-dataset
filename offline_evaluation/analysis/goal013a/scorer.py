"""One fixed, offline CPU CLIPScore adapter; no dataset or outcome imports.

The model is loaded lazily, only from a verified local checkpoint. Pure token,
aggregation and geometry helpers can be checked without a model installation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys

import numpy as np

SCORE_ID = 'MEAN6_CLIPSCORE_VITB32_CPU_FP32_SUMMARY_ONLY_GOAL013A'
PREFIX = 'A photo depicts '
CONTEXT_LENGTH = 77
PAYLOAD_LIMIT = 75
CHECKPOINT_SHA256 = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
CLIP_REVISION = 'd05afc436d78f1c48dc0dbf8e5980a9d471f35f6'
CLIPSCORE_REVISION = '1036465276513621f77f1c2208d742e4a430781f'
TOLERANCES = {'token_ids': 'EXACT', 'preprocess_atol': 0.0, 'preprocess_rtol': 0.0,
              'normalized_feature_atol': 1e-5, 'normalized_feature_rtol': 1e-5,
              'score_atol': 1e-5, 'score_rtol': 1e-5}
TRANSFORM = {'resize_short_edge': 224, 'interpolation': 'PIL_BICUBIC',
             'center_crop': [224, 224], 'rgb_conversion': 'AFTER_CROP',
             'tensor_scale': 'UINT8_DIV_255',
             'mean': [0.48145466, 0.4578275, 0.40821073],
             'std': [0.26862954, 0.26130258, 0.27577711],
             'exif_transpose': False, 'padding_before_transform': False}


class ScorerError(ValueError):
    """A fixed-method, provenance or computational-validity gate failed."""


def require(condition, message):
    if not condition:
        raise ScorerError(message)


def json_hash(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file(), 'absolute local regular file required')
    require('..' not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)), 'symlink/traversal refused')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def verify_files(config):
    root = Path(config['clip_source_root'])
    required = {root/'clip'/name for name in ('__init__.py', 'clip.py', 'model.py', 'simple_tokenizer.py', 'bpe_simple_vocab_16e6.txt.gz')}
    required |= {root/'LICENSE', Path(config['clipscore_source_path']), Path(config['clipscore_license_path']), Path(config['weight_path'])}
    descriptors = config['file_hashes']
    require(all(str(path) in descriptors for path in required), 'required source/tokenizer/license/checkpoint hash absent')
    actual = {}
    for name, descriptor in descriptors.items():
        path = Path(name)
        expected = descriptor if isinstance(descriptor, str) else descriptor['sha256']
        digest = file_hash(path)
        require(digest == expected, 'fixed source/checkpoint hash mismatch: ' + str(path))
        if isinstance(descriptor, dict) and 'bytes' in descriptor:
            require(path.stat().st_size == descriptor['bytes'], 'fixed source byte count mismatch')
        actual[str(path)] = {'sha256': digest, 'bytes': path.stat().st_size}
    require(actual[str(Path(config['weight_path']))]['sha256'] == CHECKPOINT_SHA256, 'only official ViT-B/32 checkpoint is permitted')
    return actual


def normalize_features(features):
    values = np.asarray(features)
    require(values.dtype == np.float32 and values.ndim == 2 and values.shape[0] > 0 and values.shape[1] > 0,
            'nonempty float32 feature matrix required')
    require(bool(np.isfinite(values).all()), 'nonfinite feature')
    norms = np.sqrt(np.sum(values**2, axis=1, keepdims=True))
    require(bool(np.isfinite(norms).all()) and bool((norms > 0).all()), 'zero/nonfinite feature norm')
    return values / norms


def clipscore_pairs(image_features, text_features):
    """Raw-feature reference formula, using the author's NumPy float32 path."""
    image = normalize_features(image_features)
    text = normalize_features(text_features)
    require(image.shape == text.shape, 'one text feature per image is required')
    return np.float32(2.5) * np.clip(np.sum(image * text, axis=1), 0, None)


def aggregate_six(values):
    result = {'score_id': SCORE_ID, 'score_status': 'COMPUTATION_INVALID', 'score': None,
              'view_scores': None, 'view_min': None, 'view_max': None, 'view_sd_population': None}
    if len(values) != 6:
        return {**result, 'null_reason': 'SIX_VIEWS_REQUIRED'}
    if any(v is None or not math.isfinite(float(v)) for v in values):
        return {**result, 'null_reason': 'MISSING_OR_NONFINITE_VIEW'}
    numbers = [float(v) for v in values]
    mean = math.fsum(numbers) / 6
    return {**result, 'score_status': 'COMPUTATION_VALID', 'score': mean,
            'view_scores': numbers, 'view_min': min(numbers), 'view_max': max(numbers),
            'view_sd_population': math.sqrt(math.fsum((v-mean)**2 for v in numbers) / 6), 'null_reason': None}


def score_six(image_features, text_feature):
    """Score normalized session features, without a second normalization pass."""
    image = np.asarray(image_features)
    text = np.asarray(text_feature)
    if text.ndim == 2 and text.shape[0] == 1:
        text = text[0]
    require(image.dtype == text.dtype == np.float32 and image.ndim == 2 and image.shape[0] == 6,
            'six float32 image features required')
    require(text.ndim == 1 and image.shape[1] == text.shape[0], 'text feature dimension mismatch')
    if not np.isfinite(image).all() or not np.isfinite(text).all():
        return {**aggregate_six([None]*6), 'null_reason': 'NONFINITE_FEATURE'}
    require(np.allclose(np.sum(image**2, axis=1), 1, atol=2e-5, rtol=2e-5)
            and np.isclose(np.sum(text**2), 1, atol=2e-5, rtol=2e-5), 'normalized features required')
    return aggregate_six(np.float32(2.5) * np.clip(np.sum(image * text[None, :], axis=1), 0, None))


def token_diagnostics(summary, tokenizer, tokenize):
    """Preserve actual 77 IDs; only verified payload coverage can enter Phase 2.

    The full token list belongs to the canonical run, not a public text export.
    Hash encoding is compact JSON integer array in UTF-8.
    """
    base = {'token_coverage_status': 'MISSING_SUMMARY', 'summary_token_count_untruncated': None,
            'summary_retained_tokens': None, 'summary_dropped_tokens': None, 'summary_dropped_fraction': None,
            'token_ids': None, 'token_ids_sha256': None, 'context_length': CONTEXT_LENGTH,
            'payload_limit': PAYLOAD_LIMIT, 'prefix': PREFIX, 'phase2_token_eligible': False}
    if not isinstance(summary, str) or not summary.strip():
        return base
    prefix = list(tokenizer.encode(PREFIX))
    summary_ids = list(tokenizer.encode(summary))
    combined = list(tokenizer.encode(PREFIX + summary))
    sot = tokenizer.encoder['<|startoftext|>']; eot = tokenizer.encoder['<|endoftext|>']
    require(type(sot) is int and type(eot) is int and sot != eot, 'invalid fixed special tokens')
    require(all(type(t) is int and 0 <= t <= max(sot, eot) for t in prefix + summary_ids + combined), 'invalid tokenizer IDs')
    raw_tokens = [sot] + combined + [eot]
    expected = raw_tokens[:CONTEXT_LENGTH]
    if len(raw_tokens) > CONTEXT_LENGTH:
        expected[-1] = eot
    expected += [0] * (CONTEXT_LENGTH-len(expected))
    tensor = tokenize(PREFIX + summary, context_length=CONTEXT_LENGTH, truncate=True)
    actual = tensor.tolist()
    if len(actual) == 1 and isinstance(actual[0], list):
        actual = actual[0]
    require(actual == expected, 'actual CLIP token sequence differs from verified truncation')
    reserved = any(t in (sot, eot) for t in combined)
    literal_reserved = any(x in summary.casefold() for x in ('<|startoftext|>', '<|endoftext|>'))
    concat = combined == prefix + summary_ids
    reasons = []
    if reserved or literal_reserved:
        reasons.append('RESERVED_SPECIAL_TOKEN')
    if not summary_ids:
        reasons.append('EMPTY_AFTER_TOKENIZER_CLEANUP')
    if not concat:
        reasons.append('PREFIX_SUMMARY_TOKEN_BOUNDARY_NOT_VERIFIED')
    if len(prefix) > PAYLOAD_LIMIT:
        reasons.append('PREFIX_EXCEEDS_PAYLOAD')
    verified = not reasons
    retained = min(len(summary_ids), max(0, PAYLOAD_LIMIT-len(prefix))) if verified else None
    dropped = len(summary_ids)-retained if verified else None
    active_len = min(len(raw_tokens), CONTEXT_LENGTH)
    special_mask = [i == 0 or i == active_len-1 or (i < active_len and t in (sot, eot)) for i, t in enumerate(actual)]
    return {**base, 'token_coverage_status': 'VERIFIED' if verified else 'TOKEN_COVERAGE_NOT_VERIFIED',
            'summary_token_count_untruncated': len(summary_ids), 'prefix_token_count': len(prefix),
            'combined_payload_token_count_untruncated': len(combined), 'full_token_count_untruncated': len(raw_tokens),
            'prefix_summary_concatenation_verified': concat, 'summary_retained_tokens': retained,
            'summary_dropped_tokens': dropped, 'summary_dropped_fraction': dropped/len(summary_ids) if verified else None,
            'truncated': len(raw_tokens) > CONTEXT_LENGTH,
            'low_text_coverage': dropped/len(summary_ids) > 0.5 if verified else None,
            'special_token_ids': {'SOT': sot, 'EOT': eot}, 'special_token_mask': special_mask,
            'padding_mask': [i >= active_len for i in range(CONTEXT_LENGTH)],
            'actual_nonpadding_positions': active_len, 'actual_nonzero_id_count': sum(t != 0 for t in actual),
            'model_eot_argmax_position': actual.index(max(actual)),
            'reserved_special_token_present': reserved or literal_reserved, 'token_ids': actual,
            'token_ids_sha256': json_hash(actual), 'prefix_payload_sha256': json_hash(prefix),
            'summary_payload_sha256': json_hash(summary_ids), 'combined_payload_sha256': json_hash(combined),
            'summary_sha256': hashlib.sha256(summary.encode('utf-8')).hexdigest(),
            'prefixed_text_sha256': hashlib.sha256((PREFIX+summary).encode('utf-8')).hexdigest(),
            'coverage_null_reasons': reasons, 'phase2_token_eligible': verified}


def coverage_gate(rows):
    """Check the frozen majority criterion; unknown coverage is never imputed."""
    groups = {'ALL': rows}
    for provider in ('openai', 'anthropic', 'gemini'):
        groups[provider] = [r for r in rows if r['description_provider'] == provider]
    details = {}
    for name, group in groups.items():
        nonempty = [r for r in group if r['summary_status'] == 'VALID_ORIGINAL_SUMMARY']
        diagnostics = [r.get('token_diagnostics', r) for r in nonempty]
        verified = [d for d in diagnostics if d['token_coverage_status'] == 'VERIFIED']
        low = sum(d['summary_dropped_fraction'] > 0.5 for d in verified)
        unknown = len(nonempty)-len(verified)
        status = ('NOT_ASSESSABLE_NO_NONEMPTY_SUMMARY' if not nonempty else
                  'MEASURE_NOT_REPRESENTATIVE_OF_SUMMARY' if low > len(nonempty)/2 else
                  'INDETERMINATE_TOKEN_COVERAGE' if low+unknown > len(nonempty)/2 else 'PASS')
        details[name] = {'status': status, 'nonempty_summaries': len(nonempty), 'verified': len(verified),
                         'coverage_unverified': unknown, 'unread_fraction_over_half': low}
    passed = all(d['status'] == 'PASS' for d in details.values())
    return {'status': 'PASS' if passed else 'INSUFFICIENT_MEASURE_VALIDITY',
            'phase2_allowed': passed, 'groups': details, 'overall': details['ALL'],
            'providers': {p: details[p] for p in ('openai', 'anthropic', 'gemini')},
            'human_quality_validity': 'NOT_VALIDATED'}


def crop_metadata(size):
    width, height = size
    require(type(width) is int and type(height) is int and width > 0 and height > 0, 'positive integer image size required')
    resized = (224, int(224*height/width)) if width <= height else (int(224*width/height), 224)
    left = int(round((resized[0]-224)/2.0)); top = int(round((resized[1]-224)/2.0))
    return {'original_size_wh': [width, height], 'resized_size_wh': list(resized),
            'crop_rectangle_resized_xyxy_half_open': [left, top, left+224, top+224],
            'crop_rectangle_original_xyxy_half_open': [left*width/resized[0], top*height/resized[1],
                                                      (left+224)*width/resized[0], (top+224)*height/resized[1]],
            'retained_area_fraction': 224**2/(resized[0]*resized[1]),
            'contains_complete_building': 'NOT_VERIFIED', 'transform_sha256': json_hash(TRANSFORM)}


def fixed_crop_examples(rows):
    by_scene = {}
    for row in sorted(rows, key=lambda r: (r['scene_id'], r['description_provider'])):
        by_scene.setdefault(row['scene_id'], row['images_ordered'][0])
    require(len(by_scene) >= 3, 'three fixed source scenes required')
    return [{'scene_id': scene, **by_scene[scene]} for scene in sorted(by_scene)[:3]]


class OfflineCLIPScorer:
    def __init__(self, config):
        self.config = dict(config)
        self.files = verify_files(config)
        import torch
        from PIL import Image
        self.torch = torch; self.Image = Image
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        root = Path(config['clip_source_root'])/'clip'
        package = '_goal013a_fixed_openai_clip'
        require(package not in sys.modules, 'use one fixed CLIP session per process')
        spec = importlib.util.spec_from_file_location(package, root/'__init__.py', submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec); sys.modules[package] = module
        spec.loader.exec_module(module)
        self.clip = module; self.clip_impl = sys.modules[package+'.clip']
        def no_download(*args, **kwargs):
            raise ScorerError('model download is disabled during inference')
        self.clip_impl._download = no_download
        self.model, self.preprocess = module.load(str(Path(config['weight_path'])), device='cpu', jit=False)
        self.model = self.model.float().eval()
        require(self.model.context_length == 77 and self.model.visual.input_resolution == 224,
                'fixed CLIP architecture mismatch')
        require(all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in self.model.parameters()), 'CPU float32 model required')
        self.tokenizer = self.clip_impl._tokenizer
        self.tokenizer_identity = json_hash({name: self.files[str(root/name)]['sha256']
                                            for name in ('simple_tokenizer.py', 'bpe_simple_vocab_16e6.txt.gz')})
        self.image_cache = {}; self.text_cache = {}
        self.cache_counts = {'image_hits': 0, 'text_hits': 0, 'image_encoded': 0, 'text_encoded': 0}

    def token_diagnostics(self, summary):
        return token_diagnostics(summary, self.tokenizer, self.clip.tokenize)

    def image_tensor(self, path):
        raw = Path(path).read_bytes()
        with self.Image.open(io.BytesIO(raw)) as image:
            return self.preprocess(image)

    def crop_preview(self, path):
        with self.Image.open(io.BytesIO(Path(path).read_bytes())) as image:
            metadata = crop_metadata(image.size)
            resized = image.resize(tuple(metadata['resized_size_wh']), resample=self.Image.Resampling.BICUBIC)
            preview = resized.crop(tuple(metadata['crop_rectangle_resized_xyxy_half_open'])).convert('RGB')
        return preview, metadata

    def _raw_images(self, paths, batch_size=8):
        require(type(batch_size) is int and 1 <= batch_size <= 8 and len(paths) > 0, 'batch size must be 1..8')
        pieces = []
        with self.torch.inference_mode():
            for start in range(0, len(paths), batch_size):
                tensor = self.torch.stack([self.image_tensor(p) for p in paths[start:start+batch_size]])
                require(tensor.dtype == self.torch.float32, 'float32 preprocessing required')
                pieces.append(self.model.encode_image(tensor).cpu().numpy().copy())
        return np.vstack(pieces)

    def _raw_texts(self, summaries, batch_size=8):
        require(type(batch_size) is int and 1 <= batch_size <= 8 and len(summaries) > 0, 'batch size must be 1..8')
        require(all(isinstance(s, str) and s.strip() for s in summaries), 'missing summary must not be encoded')
        pieces = []
        with self.torch.inference_mode():
            for start in range(0, len(summaries), batch_size):
                tokens = self.clip.tokenize([PREFIX+s for s in summaries[start:start+batch_size]], truncate=True)
                pieces.append(self.model.encode_text(tokens).cpu().numpy().copy())
        return np.vstack(pieces)

    def encode_images(self, paths, batch_size=8, use_cache=True):
        return self._cached(paths, self.image_cache, self._raw_images, 'image', batch_size, use_cache,
                            lambda p: (file_hash(Path(p)), json_hash(TRANSFORM), CHECKPOINT_SHA256))

    def encode_texts(self, summaries, batch_size=8, use_cache=True):
        require(all(isinstance(value, str) and value.strip() for value in summaries), 'missing summary must not be encoded')
        return self._cached(summaries, self.text_cache, self._raw_texts, 'text', batch_size, use_cache,
                            lambda s: (hashlib.sha256((PREFIX+s).encode('utf-8')).hexdigest(), self.tokenizer_identity, CHECKPOINT_SHA256))

    def _cached(self, items, cache, encode, label, batch_size, use_cache, key):
        require(type(batch_size) is int and 1 <= batch_size <= 8 and len(items) > 0, 'batch size must be 1..8')
        if not use_cache:
            return normalize_features(encode(items, batch_size))
        keys = [key(x) for x in items]; pending = {}
        for item, identity in zip(items, keys):
            if identity in cache:
                self.cache_counts[label+'_hits'] += 1
            else:
                pending.setdefault(identity, item)
        if pending:
            features = normalize_features(encode(list(pending.values()), batch_size))
            self.cache_counts[label+'_encoded'] += len(pending)
            for identity, vector in zip(pending, features):
                cache[identity] = vector.copy()
        return np.stack([cache[k].copy() for k in keys])


def load_model(config):
    return OfflineCLIPScorer(config)
