"""Small synthetic fixtures; no source text, real model or production matrix."""
import copy
import hashlib
import json

import numpy as np
import pytest

from offline_evaluation.analysis.goal014 import diagnostics as d
from offline_evaluation.analysis.goal013a import scorer


HASHES = dict(score_sha256='1'*64, config_sha256='2'*64, source_binding_sha256='3'*64)


def fixture_data(missing=True):
    config = {'planned_scene_ids': ['A', 'B'], 'score_id': scorer.SCORE_ID,
              'transform': copy.deepcopy(scorer.TRANSFORM), 'tolerances': dict(scorer.TOLERANCES),
              'preprocess_sha256': scorer.json_hash(scorer.TRANSFORM), 'clip_source_root': '/fixture/clip',
              'weight_path': '/fixture/weight.pt', 'file_hashes': {
                  '/fixture/weight.pt': {'sha256': scorer.CHECKPOINT_SHA256},
                  '/fixture/clip/clip/simple_tokenizer.py': {'sha256': 'a'*64},
                  '/fixture/clip/clip/bpe_simple_vocab_16e6.txt.gz': {'sha256': 'b'*64}},
              'installed_distributions': {'numpy': 'synthetic'}}
    tokenizer = scorer.json_hash({'simple_tokenizer.py': 'a'*64, 'bpe_simple_vocab_16e6.txt.gz': 'b'*64})
    images = np.array([[1, 0]]*3+[[-1, 0]]*3+[[0, 1]]*6, dtype=np.float32)
    rows = []
    for scene in ['A', 'B']:
        for provider in d.PROVIDERS:
            absent = missing and scene == 'B' and provider == 'gemini'
            q = [2.5]*3+[0.]*3 if scene == 'A' else [0.]*6
            row = {'scene_id': scene, 'description_provider': provider,
                   'images_ordered': [{'path': '/fixture/'+scene+str(i)+'.png', 'sha256': 'c'*64, 'bytes': 30} for i in range(6)],
                   'score_id': scorer.SCORE_ID, 'binding_status': 'PASS', 'source_binding_status': 'PASS',
                   'weight_sha256': scorer.CHECKPOINT_SHA256, 'transform_sha256': config['preprocess_sha256'],
                   'tokenizer_sha256': tokenizer, 'score_status': 'MISSING_SUMMARY' if absent else 'COMPUTATION_VALID',
                   'summary_status': 'EMPTY_SUMMARY' if absent else 'VALID_ORIGINAL_SUMMARY',
                   'score': None if absent else sum(q)/6, 'view_scores': None if absent else q,
                   'null_reason': 'EMPTY_SUMMARY' if absent else None, 'summary_sha256': 'd'*64,
                   'description_source_path': '/fixture/'+scene+provider+'.json', 'description_source_sha256': 'e'*64,
                   'O_input_id': scene+provider, 'O_input_sha256': 'e'*64,
                   'token_coverage_status': 'MISSING_SUMMARY' if absent else 'VERIFIED',
                   'token_diagnostics': {'token_coverage_status': 'VERIFIED', 'phase2_token_eligible': True,
                        'prefix': scorer.PREFIX, 'context_length': 77, 'summary_sha256': 'd'*64,
                        'prefixed_text_sha256': 'f'*64, 'token_ids_sha256': '0'*64}}
            rows.append(row)
    texts = np.array([[1, 0]]*sum(r['score_status'] == 'COMPUTATION_VALID' for r in rows), dtype=np.float32)
    metadata = d.build_feature_metadata(rows, config, **HASHES, fixture=True)
    return rows, config, images, texts, metadata


def compute(data):
    rows, config, images, texts, metadata = data
    return d.compute_diagnostics(images, texts, metadata, rows, config, **HASHES, fixture=True)


def test_matching_replay_clip_before_mean_and_missing_rows():
    result = compute(fixture_data())
    assert result['matrix'].shape == (5, 2)
    assert result['matrix'].dtype == np.float64
    assert result['replay']['per_view_comparisons'] == 30
    assert result['replay']['summary_mean_comparisons'] == 5
    assert result['replay']['feature_pairs'] == 60
    assert result['rows'][0]['matching_score'] == 1.25
    assert result['rows'][0]['D_i'] == 1.25 and result['rows'][0]['W_i'] == 1
    assert result['rows'][3]['D_i'] == -1.25 and result['rows'][3]['W_i'] == 0
    assert result['rows'][4]['status'] == 'MISSING_SUMMARY'
    assert result['rows'][4]['D_i'] is None and result['rows'][4]['W_i'] is None
    assert result['rows'][4]['null_reason'] == 'EMPTY_SUMMARY'
    assert len(result['rows']) == 6


@pytest.mark.parametrize('target', ['image_order', 'text_order', 'token', 'config', 'source'])
def test_metadata_mismatch_fails_closed(target):
    rows, config, images, texts, metadata = fixture_data()
    if target == 'image_order':
        metadata['image_rows'][0], metadata['image_rows'][1] = metadata['image_rows'][1], metadata['image_rows'][0]
    elif target == 'text_order':
        metadata['text_rows'][0], metadata['text_rows'][1] = metadata['text_rows'][1], metadata['text_rows'][0]
    elif target == 'token':
        metadata['text_rows'][0]['token_ids_sha256'] = '9'*64
    elif target == 'config':
        metadata['config_sha256'] = '9'*64
    else:
        metadata['source_binding_sha256'] = '9'*64
    with pytest.raises(ValueError, match='metadata/order/source/token/config'):
        compute((rows, config, images, texts, metadata))


@pytest.mark.parametrize('case', ['nonfinite', 'zero', 'unnormalized', 'dtype', 'shape'])
def test_invalid_features_rejected(case):
    rows, config, images, texts, metadata = fixture_data()
    if case == 'nonfinite':
        images[0, 0] = np.nan
    elif case == 'zero':
        images[0] = 0
    elif case == 'unnormalized':
        images[0] *= 2
    elif case == 'dtype':
        images = images.astype(np.float64)
    else:
        images = images[:-1]
    with pytest.raises(ValueError):
        compute((rows, config, images, texts, metadata))


@pytest.mark.parametrize('case', ['view', 'mean', 'array_order'])
def test_original_score_or_embedding_order_difference_stops(case):
    data = fixture_data()
    rows, config, images, texts, metadata = data
    if case == 'view':
        rows[0]['view_scores'][0] = 2.4
    elif case == 'mean':
        rows[0]['score'] = 1.3
    else:
        images[[0, 6]] = images[[6, 0]]
    with pytest.raises(ValueError, match='replay mismatch'):
        compute(data)


def test_exact_ties_half_credit_and_candidate_permutation():
    scores = [1., 1., 1.-1e-12, 1.+1e-12]
    result = d.contrast_row(scores, ['A', 'B', 'C', 'D'], 'A')
    assert result['nonmatching_candidates'] == 3
    assert result['ties'] == result['wins'] == result['losses'] == 1
    assert result['W_i'] == .5
    perm = [2, 0, 3, 1]
    actual = d.contrast_row([scores[i] for i in perm], [['A', 'B', 'C', 'D'][i] for i in perm], 'A')
    assert actual == result
    assert d.contrast_row([1.]*60, [str(i) for i in range(60)], '0')['ties'] == 59


def test_matching_candidate_is_excluded_from_other59():
    result = d.contrast_row([2.]+[1.]*59, [str(i) for i in range(60)], '0')
    assert result['nonmatching_mean'] == 1 and result['nonmatching_sd_population'] == 0
    assert result['D_i'] == 1 and result['W_i'] == 1 and result['nonmatching_candidates'] == 59


def test_saved_bundle_numeric_only_identity_and_exclusive(tmp_path):
    rows, config, images, texts, metadata = fixture_data()
    manifest = d.save_feature_bundle(tmp_path, images, texts, metadata, fixture=True)
    loaded_images, loaded_texts, loaded_meta = d.load_feature_bundle(tmp_path/'FEATURE_MANIFEST.json', metadata, fixture=True)
    np.testing.assert_array_equal(images, loaded_images)
    np.testing.assert_array_equal(texts, loaded_texts)
    assert loaded_meta == metadata and manifest['images']['allow_pickle'] is False
    with pytest.raises(ValueError, match='exclusive'):
        d.save_feature_bundle(tmp_path, images, texts, metadata, fixture=True)
    with pytest.raises(ValueError, match='no pickle'):
        d.save_numeric_array(tmp_path/'bad.npy', np.array([{}], dtype=object), fixture=True)
    with (tmp_path/'IMAGE_FEATURES.npy').open('ab') as stream:
        stream.write(b'x')
    with pytest.raises(ValueError, match='bytes/hash'):
        d.load_feature_bundle(tmp_path/'FEATURE_MANIFEST.json', metadata, fixture=True)


def test_source_hashes_and_original_score_records_not_mutated():
    data = fixture_data()
    before = json.dumps(data[0], sort_keys=True)
    compute(data)
    assert json.dumps(data[0], sort_keys=True) == before
    data[0][0]['transform_sha256'] = '9'*64
    with pytest.raises(ValueError, match='model/transform/tokenizer'):
        d.build_feature_metadata(data[0], data[1], **HASHES, fixture=True)


def test_production_output_guard_refuses_old_run(tmp_path):
    with pytest.raises(ValueError):
        d.save_numeric_array('/root/minecraft-runs/goal013a-not-real/new.npy', np.ones(2))
