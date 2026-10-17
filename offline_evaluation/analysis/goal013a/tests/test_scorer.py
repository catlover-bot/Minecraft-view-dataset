"""Synthetic fixtures only; no torch import, real checkpoint or source dataset."""
import hashlib
import itertools

import numpy as np
import pytest

from offline_evaluation.analysis.goal013a import scorer as s
from offline_evaluation.analysis.goal013a.reference_parity import independent_scalar_scores


class TinyTokenizer:
    encoder = {'<|startoftext|>': 98, '<|endoftext|>': 99}

    def encode(self, text):
        result = []
        for word in text.lower().split():
            if word == 'clean_empty':
                continue
            result.append({'a': 1, 'photo': 2, 'depicts': 3, '!': 0,
                           '<|startoftext|>': 98, '<|endoftext|>': 99,
                           'joined': 8}.get(word, 4))
        if text == s.PREFIX + 'JOINED':
            result[-1] = 7
        return result

    def tokenize(self, text, context_length=77, truncate=True):
        values = [98] + self.encode(text) + [99]
        if len(values) > context_length:
            assert truncate
            values = values[:context_length]; values[-1] = 99
        return np.array([values + [0] * (context_length-len(values))], dtype=np.int32)


def diagnostic(text):
    tokenizer = TinyTokenizer()
    return s.token_diagnostics(text, tokenizer, tokenizer.tokenize)


def test_exact_summary_bytes_prefix_and_id_zero_are_preserved():
    text = '  ! 日本語  '
    d = diagnostic(text)
    assert d['token_coverage_status'] == 'VERIFIED'
    assert d['summary_sha256'] == hashlib.sha256(text.encode()).hexdigest()
    assert d['prefix_token_count'] == 3 and d['summary_token_count_untruncated'] == 2
    assert d['token_ids'][:7] == [98, 1, 2, 3, 0, 4, 99]
    assert d['actual_nonpadding_positions'] == 7 and d['actual_nonzero_id_count'] == 6
    assert d['padding_mask'][4] is False and d['special_token_mask'][4] is False
    assert d['summary_retained_tokens'] == 2 and d['summary_dropped_tokens'] == 0


@pytest.mark.parametrize('length,dropped,low', [(72, 0, False), (73, 1, False), (144, 72, False), (145, 73, True)])
def test_payload_prefix_budget_and_strict_half_threshold(length, dropped, low):
    d = diagnostic(' '.join(['token']*length))
    assert d['summary_retained_tokens'] == min(length, 72)
    assert d['summary_dropped_tokens'] == dropped and d['low_text_coverage'] is low
    assert len(d['token_ids']) == 77 and d['token_ids'][-1] == 99
    assert d['truncated'] is (length > 72)


@pytest.mark.parametrize('text', [None, '', ' \t\n', 23])
def test_missing_summary_never_becomes_zero_score_or_token_count(text):
    d = diagnostic(text)
    assert d['token_coverage_status'] == 'MISSING_SUMMARY'
    assert d['summary_token_count_untruncated'] is None and d['token_ids'] is None


@pytest.mark.parametrize('text,reason', [
    ('<|endoftext|> building', 'RESERVED_SPECIAL_TOKEN'),
    ('CLEAN_EMPTY', 'EMPTY_AFTER_TOKENIZER_CLEANUP'),
    ('JOINED', 'PREFIX_SUMMARY_TOKEN_BOUNDARY_NOT_VERIFIED'),
])
def test_unverified_token_coverage_is_not_association_eligible(text, reason):
    d = diagnostic(text)
    assert d['token_coverage_status'] == 'TOKEN_COVERAGE_NOT_VERIFIED'
    assert d['phase2_token_eligible'] is False and d['summary_dropped_fraction'] is None
    assert reason in d['coverage_null_reasons']


def test_actual_token_sequence_mismatch_is_fatal():
    tokenizer = TinyTokenizer()
    with pytest.raises(s.ScorerError, match='actual CLIP token'):
        s.token_diagnostics('building', tokenizer, lambda *args, **kwargs: np.zeros((1, 77), dtype=np.int32))


def test_reference_formula_has_no_logit_scale_upper_one_clip_or_percent():
    images = np.array([[3, 4], [1, 0], [-1, 0]], dtype=np.float32)
    texts = np.array([[3, 4], [0, 1], [1, 0]], dtype=np.float32)
    q = s.clipscore_pairs(images, texts)
    np.testing.assert_allclose(q, [2.5, 0, 0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(q, independent_scalar_scores(images, texts), rtol=1e-5, atol=1e-5)
    assert q.dtype == np.float32


@pytest.mark.parametrize('values', [np.zeros((1, 2), dtype=np.float32), np.array([[np.nan, 1]], dtype=np.float32), np.ones((1, 2), dtype=np.float64)])
def test_bad_features_are_rejected(values):
    with pytest.raises(s.ScorerError):
        s.normalize_features(values)


def test_mean6_population_sd_and_permutation_invariance():
    values = [0., .2, .4, .6, .8, 1.]
    result = s.aggregate_six(values)
    assert result['score'] == .5 and result['score_status'] == 'COMPUTATION_VALID'
    assert result['view_sd_population'] == pytest.approx(np.std(values, ddof=0))
    for permutation in itertools.permutations(values):
        assert s.aggregate_six(permutation)['score'] == result['score']
    image = np.repeat(np.array([[1, 0]], dtype=np.float32), 6, axis=0)
    assert s.score_six(image, np.array([1, 0], dtype=np.float32))['score'] == 2.5


@pytest.mark.parametrize('values', [[1.]*5, [1.]*5+[None], [1.]*5+[float('nan')], [1.]*5+[float('inf')]])
def test_one_missing_or_nonfinite_view_refuses_partial_average(values):
    result = s.aggregate_six(values)
    assert result['score'] is None and result['score_status'] == 'COMPUTATION_INVALID'


def test_crop_is_recorded_in_resized_and_original_frames():
    d = s.crop_metadata((800, 400))
    assert d['resized_size_wh'] == [448, 224]
    assert d['crop_rectangle_resized_xyxy_half_open'] == [112, 0, 336, 224]
    assert d['crop_rectangle_original_xyxy_half_open'] == [200., 0., 600., 400.]
    assert d['retained_area_fraction'] == .5
    assert d['contains_complete_building'] == 'NOT_VERIFIED'
    assert s.crop_metadata((400, 800))['crop_rectangle_resized_xyxy_half_open'] == [0, 112, 224, 336]


def test_crop_examples_choose_first_three_scene_ids_and_first_requested_view():
    rows = [{'scene_id': name, 'description_provider': provider,
             'images_ordered': [{'path': name+'/first.png', 'sha256': 'a'*64}]} for name in ['004','002','003','001'] for provider in ['openai','gemini']]
    examples = s.fixed_crop_examples(rows)
    assert [r['scene_id'] for r in examples] == ['001','002','003']
    assert all(r['path'].endswith('/first.png') for r in examples)


def test_coverage_gate_checks_each_provider_and_never_fills_unknowns():
    rows = [{'description_provider': provider, 'summary_status': 'VALID_ORIGINAL_SUMMARY',
             'token_diagnostics': diagnostic('short')} for provider in ('openai','anthropic','gemini') for _ in range(2)]
    assert s.coverage_gate(rows)['phase2_allowed'] is True
    rows[-1]['token_diagnostics'] = diagnostic(' '.join(['long']*145))
    assert s.coverage_gate(rows)['phase2_allowed'] is True  # exactly half is not a majority
    rows[-2]['token_diagnostics'] = diagnostic(' '.join(['long']*145))
    result = s.coverage_gate(rows)
    assert result['phase2_allowed'] is False
    assert result['groups']['gemini']['status'] == 'MEASURE_NOT_REPRESENTATIVE_OF_SUMMARY'


def test_hash_mismatch_rejected_before_any_model_import(tmp_path):
    source = tmp_path/'fixed.py'; source.write_bytes(b'fixture')
    config = {'clip_source_root': str(tmp_path), 'clipscore_source_path': str(source),
              'clipscore_license_path': str(source), 'weight_path': str(source), 'file_hashes': {}}
    with pytest.raises(s.ScorerError, match='hash absent'):
        s.load_model(config)


@pytest.mark.parametrize('mutation', ['source_hash', 'wrong_checkpoint'])
def test_complete_but_untrusted_source_or_checkpoint_rejected(tmp_path, mutation):
    paths = [tmp_path/'clip'/name for name in ('__init__.py','clip.py','model.py','simple_tokenizer.py','bpe_simple_vocab_16e6.txt.gz')]
    paths += [tmp_path/'LICENSE', tmp_path/'clipscore.py', tmp_path/'clipscore-LICENSE', tmp_path/'weight.pt']
    descriptors = {}
    for path in paths:
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'synthetic '+path.name.encode())
        descriptors[str(path)] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    if mutation == 'source_hash':
        descriptors[str(paths[0])]['sha256'] = '0'*64
    config = {'clip_source_root': str(tmp_path), 'clipscore_source_path': str(tmp_path/'clipscore.py'),
              'clipscore_license_path': str(tmp_path/'clipscore-LICENSE'), 'weight_path': str(tmp_path/'weight.pt'),
              'file_hashes': descriptors}
    with pytest.raises(s.ScorerError, match='hash mismatch|official ViT-B/32'):
        s.load_model(config)


def test_cache_uses_image_bytes_and_returns_independent_copies(tmp_path):
    session = object.__new__(s.OfflineCLIPScorer)
    session.image_cache = {}; session.cache_counts = {'image_hits': 0, 'image_encoded': 0}
    calls = []
    def encode(paths, batch_size):
        calls.append(tuple(paths))
        return np.tile(np.array([[3.,4.]], dtype=np.float32), (len(paths),1))
    session._raw_images = encode
    a = tmp_path/'a.png'; b = tmp_path/'b.png'
    a.write_bytes(b'same'); b.write_bytes(b'same')
    first = session.encode_images([a,b]); first[0,0] = -99
    second = session.encode_images([b,a])
    assert len(calls) == 1 and session.cache_counts['image_encoded'] == 1
    np.testing.assert_allclose(second, [[.6,.8],[.6,.8]], atol=1e-6)
    a.write_bytes(b'changed'); session.encode_images([a])
    assert len(calls) == 2
