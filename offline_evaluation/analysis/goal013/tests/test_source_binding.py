"""Small independent synthetic sources; never load the fixed research cohort."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from offline_evaluation.analysis.goal013 import source_binding as sb


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value, inventory):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()
    path.write_bytes(raw)
    inventory[str(path)] = {'bytes': len(raw), 'sha256': sha(raw)}
    return raw


@pytest.fixture
def sources(tmp_path):
    root = tmp_path/'repo'
    scene = 'buildings_100_v4_building_001'
    building = 'building_001'
    scene_root = root/'datasets/buildings_100_v4'/building
    paths = [f'images/view{i:02d}.png' for i in range(12)]
    selected = [paths[i] for i in (0, 2, 4, 6, 10, 11)]
    inventory = {}
    for i, name in enumerate(paths):
        path = scene_root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = b'fixture-image-bytes:' + str(i).encode()
        path.write_bytes(raw)
        inventory[str(path)] = {'bytes': len(raw), 'sha256': sha(raw)}
    meta_path = scene_root/'meta.json'
    save(meta_path, {'views': [{'path': name} for name in paths], 'gt_path': 'DO_NOT_READ.npy'}, inventory)
    rows = []; bindings = []; requests = []; schedule = []
    models = {'openai': ('gpt-5-mini', 'description_openai_gpt_5_mini'),
              'anthropic': ('claude-haiku-4-5-20251001', 'description_anthropic_claude_haiku_4_5_20251001'),
              'gemini': ('gemini-3.1-pro-preview', 'description_gemini_gemini_3_1_pro_preview')}
    image_paths = [str(scene_root/p) for p in selected]
    image_hashes = [inventory[p]['sha256'] for p in image_paths]
    for provider, (model, dirname) in models.items():
        prefix = root/'datasets' if provider == 'gemini' else root/'outputs/i2t2b'
        desc_path = prefix/'buildings_100_v4'/building/dirname/'description.json'
        description = {'building': building, 'provider': provider, 'model': model,
                       'created_at': '2026-01-01T00:00:00+00:00',
                       'summary': '  屋根\ne\u0301 🧱  ', 'dimensions': 'must not enter scoring payload'}
        raw = save(desc_path, description, inventory)
        input_id = 'fixture-original-' + provider
        rows.append({'scene_id': scene, 'description_provider': provider, 'source_path': str(desc_path),
                     'source_sha256': sha(raw), 'input_sha256': sha(raw), 'input_id': input_id, 'description_model_id': model})
        for builder in ('openai', 'anthropic'):
            schedule.append({'condition': 'O', 'scene_id': scene, 'description_model': provider, 'provider': builder,
                             'input_id': input_id, 'input_sha256': sha(raw), 'source_sha256': sha(raw), 'input_text': raw.decode(),
                             'gt_path': str(scene_root/'DO_NOT_READ.npy'), 'outcome_path': str(root/'DO_NOT_READ.json')})
        request = {'building': building, 'provider': provider, 'model': model, 'prompt': 'same prompt',
                   'system_prompt': 'same system', 'images': selected,
                   'created_at': '2026-01-01T00:00:01+00:00', 'temperature': .2,
                   'max_tokens': 4096 if provider == 'gemini' else 1800}
        if provider == 'gemini': request['llm_seed'] = -1
        request_path = desc_path.with_name('description.request.json')
        save(request_path, request, inventory)
        binding = {'scene_id': scene, 'description_provider': provider, 'input_id': input_id, 'input_sha256': sha(raw),
                   'description_source_path': str(desc_path), 'description_source_sha256': sha(raw),
                   'request_path': str(request_path), 'request_sha256': inventory[str(request_path)]['sha256'],
                   'meta_path': str(meta_path), 'meta_sha256': inventory[str(meta_path)]['sha256'],
                   'actual_image_paths_ordered': image_paths, 'actual_image_hashes_ordered': image_hashes,
                   'builder_observations': {'ignored': 'MUST NOT READ'}, 'gt_voxels_path': 'MUST NOT READ'}
        prior = {k: v for k, v in binding.items() if k not in ('builder_observations', 'gt_voxels_path')}
        prior.update(description_model=model, prompt_sha256=sha(b'same prompt'), system_prompt_sha256=sha(b'same system'),
                     description_record_created_at_utc=description['created_at'], request_record_created_at_utc=request['created_at'],
                     actual_meta_view_indices_ordered=[0, 2, 4, 6, 10, 11])
        for field in ('max_tokens', 'temperature', 'top_p', 'top_k', 'seed', 'llm_seed'):
            prior[field] = request.get(field); prior[field+'_recorded'] = field in request
        bindings.append(binding); requests.append(prior)
    schedule.append({'condition': 'J', 'input_text': 'DO NOT USE'})
    return {'root': root, 'source_rows': rows, 'manifest': {'identity': {'evaluation_scene_ids': [scene], 'full_schedule': schedule}},
            'request_bindings': {'status': 'PASS', 'bindings': bindings, 'inventory': inventory},
            'request_audit': {'summary': {'status': 'PASS'}, 'requests': requests}}


def bind(sources):
    inventory = {}
    rows = sb.bind_records(**sources, inventory=inventory, expected_scenes=1, fixture=True)
    sb.verify_inventory(inventory)
    return rows, inventory


def test_frozen_identity_hash_uses_historical_pretty_json_and_newline():
    # Golden text is independent of the implementation serializer.
    expected = '{\n  "a": "屋根",\n  "b": [\n    1,\n    false\n  ]\n}\n'.encode()
    assert sb.frozen_object_hash({'b': [1, False], 'a': '屋根'}) == sha(expected)
    assert sb.frozen_object_hash({'b': [1, False], 'a': '屋根'}) != sha(expected.rstrip())


def test_complete_synthetic_binding_exports_hashes_not_text(sources, tmp_path, monkeypatch):
    opened = []
    original_open = Path.open
    def tracked_open(path, *args, **kwargs):
        assert 'DO_NOT_READ' not in str(path)
        opened.append(str(path))
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', tracked_open)
    rows, inventory = bind(sources)
    assert len(rows) == 3
    assert len(inventory) == 13  # Three descriptions + three requests + meta + six selected image files.
    assert set(opened) == set(inventory)
    for row in rows:
        assert row['binding_status'] == 'PASS'
        assert row['saved_candidate_view_count'] == 12
        assert row['actual_request_image_count'] == 6
        assert row['O_input_utf8_bytes'] == row['description_source_bytes']
        assert row['summary_sha256'] == sha('  屋根\ne\u0301 🧱  '.encode())
        assert row['score'] is None and row['score_status'] == 'COMPUTATION_NOT_PERFORMED'
        payload = sb.load_scoring_record(row, root=sources['root'])
        assert set(payload) == {'summary', 'images_ordered'}
        assert payload['summary'] == '  屋根\ne\u0301 🧱  '
        assert payload['images_ordered'] == [i['path'] for i in row['images_ordered']]
    products = sb.write_outputs({'rows': rows, 'receipt': {'status': 'PASS'}}, tmp_path/'result')
    assert set(products) == {'SOURCE_BINDING_180.csv', 'SOURCE_BINDING_180.json', 'SOURCE_BINDING_RECEIPT.json'}
    exported = (tmp_path/'result/SOURCE_BINDING_180.json').read_text()
    assert '屋根' not in exported and 'must not enter' not in exported and 'DO_NOT_READ' not in exported
    with pytest.raises(FileExistsError):
        sb.write_outputs({'rows': rows, 'receipt': {}}, tmp_path/'result')


@pytest.mark.parametrize('description,kind,status', [
    ({}, 'missing', 'MISSING_SUMMARY_FIELD'),
    ({'summary': None}, 'null', 'MISSING_NON_STRING_SUMMARY'),
    ({'summary': False}, 'boolean', 'MISSING_NON_STRING_SUMMARY'),
    ({'summary': 7}, 'number', 'MISSING_NON_STRING_SUMMARY'),
    ({'summary': []}, 'array', 'MISSING_NON_STRING_SUMMARY'),
    ({'summary': {}}, 'object', 'MISSING_NON_STRING_SUMMARY'),
    ({'summary': ''}, 'string', 'MISSING_EMPTY_SUMMARY'),
    ({'summary': ' \n\t'}, 'string', 'MISSING_EMPTY_SUMMARY'),
])
def test_missing_types_do_not_become_text(description, kind, status):
    value = sb.summary_metadata(description)
    assert value['summary_json_type'] == kind
    assert value['summary_status'] == status
    assert value['summary_field_present'] == ('summary' in description)
    if kind not in ('missing', 'string'):
        assert value['summary_sha256'] is None
        assert value['summary_json_value_sha256'] is not None


@pytest.mark.parametrize('mutation,message', [
    ('saved_O', 'saved O text'), ('image_bytes', 'source hash changed'),
    ('request_order', 'source hash changed'), ('description_bytes', 'source hash changed'),
    ('scope', 'identity scope'), ('prior_cap', 'recorded request settings'), ('prior_gate', 'prior request gates'),
])
def test_binding_rejects_changed_source_or_binding(sources, mutation, message):
    if mutation == 'saved_O': sources['manifest']['identity']['full_schedule'][0]['input_text'] += ' '
    elif mutation == 'image_bytes': Path(sources['request_bindings']['bindings'][0]['actual_image_paths_ordered'][0]).write_bytes(b'changed')
    elif mutation == 'description_bytes': Path(sources['source_rows'][0]['source_path']).write_text('{}')
    elif mutation == 'request_order':
        path = Path(sources['request_bindings']['bindings'][0]['request_path'])
        request = json.loads(path.read_text()); request['images'].reverse(); path.write_text(json.dumps(request))
    elif mutation == 'scope': sources['source_rows'].pop()
    elif mutation == 'prior_cap': sources['request_audit']['requests'][0]['max_tokens'] = 1
    elif mutation == 'prior_gate': sources['request_bindings']['status'] = 'FAIL'
    with pytest.raises(sb.SourceBindingError, match=message): bind(sources)


def test_summary_and_image_revalidation_prevents_scoring_mutations(sources):
    rows, _ = bind(sources)
    row = next(r for r in rows if r['description_provider'] == 'openai')
    substituted = copy.deepcopy(row)
    substituted['images_ordered'].reverse()
    with pytest.raises(sb.SourceBindingError, match='image order'):
        sb.load_scoring_record(substituted, root=sources['root'])
    wrong_role = copy.deepcopy(row)
    wrong_role['images_ordered'][0]['path'] = str(sources['root']/'GT.npy')
    with pytest.raises(sb.SourceBindingError, match='image order'):
        sb.load_scoring_record(wrong_role, root=sources['root'])
    path = Path(row['description_source_path'])
    description = json.loads(path.read_text()); description['summary'] += '!'
    path.write_text(json.dumps(description))
    with pytest.raises(sb.SourceBindingError, match='source hash changed'):
        sb.load_scoring_record(row, root=sources['root'])


def test_source_role_and_symlink_rejected_before_open(sources, tmp_path):
    with pytest.raises(sb.SourceBindingError, match='source allowlist'):
        sb.description_source(sources['root'], 'buildings_100_v4_building_001', 'openai', sources['root']/'GT.json')
    with pytest.raises(sb.SourceBindingError, match='unsafe relative'):
        sb.request_image_paths(sources['root'], 'buildings_100_v4_building_001', ['../GT.npy'] + [f'images/{i}.png' for i in range(5)])
    target = tmp_path/'target'; target.write_text('x')
    link = tmp_path/'link'; link.symlink_to(target)
    with pytest.raises(sb.SourceBindingError, match='symlink'): sb.record(link, {})


def test_strict_json_and_final_inventory_fail_closed(sources):
    for raw in (b'{"summary":"a","summary":"b"}', b'{"summary":NaN}', b'{"summary":1e999}'):
        with pytest.raises(sb.SourceBindingError): sb.strict_json(raw)
    _, inventory = bind(sources)
    Path(sources['source_rows'][0]['source_path']).write_bytes(b'changed after binding')
    with pytest.raises(sb.SourceBindingError, match='source hash changed'): sb.verify_inventory(inventory)
