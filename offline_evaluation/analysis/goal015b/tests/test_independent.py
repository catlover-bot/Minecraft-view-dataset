"""Small independent executor and real Goal015 extraction/evaluation fixtures."""
from copy import deepcopy
import hashlib
import json
import random

import pytest

from offline_evaluation.analysis.goal015b.independent import (
    execute_independent, grid_sha256, IndependentLimits, IndependentExecutionError)
from offline_evaluation.server_v2 import core
from offline_evaluation.experiments.goal015 import adapter, evaluate


def set_block(x=0, y=0, z=0, block='stone'):
    return dict(op='set', x=x, y=y, z=z, block=block)


def cuboid(op='fill', first=(0, 0, 0), second=(1, 1, 1), block='stone'):
    row = {'op': op, **{axis + '1': value for axis, value in zip('xyz', first)},
           **{axis + '2': value for axis, value in zip('xyz', second)}}
    if op != 'carve': row['block'] = block
    return row


FIXTURES = [
    ('empty_array', [], 0),
    ('stone_set', [set_block()], 1),
    ('fill_2x2x2', [cuboid()], 8),
    ('reversed_endpoints', [cuboid(first=(1, 1, 1), second=(0, 0, 0))], 8),
    ('air_delete', [cuboid(), set_block(block='minecraft:air')], 7),
    ('carve_delete', [cuboid(), cuboid(op='carve', second=(0, 0, 0))], 7),
    ('place_delete_all', [cuboid(), cuboid(op='carve')], 0),
    ('material_only', [set_block(), set_block(block='minecraft:dirt')], 1),
]


@pytest.mark.parametrize('name,operations,count', FIXTURES, ids=[r[0] for r in FIXTURES])
def test_eight_requested_fixtures_match_existing_executor(name, operations, count):
    before = deepcopy(operations)
    actual, trace = execute_independent(operations)
    assert actual == core.execute(operations).voxels and len(actual) == count
    assert operations == before and trace['operation_count'] == len(operations)
    assert trace['final_occupied_voxels'] == count and trace['final_grid_sha256'] == grid_sha256(actual)
    if name == 'place_delete_all':
        assert trace['first_occupied_operation_index'] == trace['max_occupied_operation_index'] == 0
        assert trace['max_occupied_voxels'] == 8 and trace['last_zero_transition_operation_index'] == 1
        assert trace['non_air_write_visits'] == trace['air_write_visits'] == trace['deleted_occupied_voxels'] == 8
        assert trace['empty_classification'] == 'PLACED_THEN_DELETED_ALL'
    if name == 'material_only':
        assert actual[(0, 0, 0)] == 'dirt' and trace['material_replacements'] == 1
        assert trace['new_occupied_voxels'] == 1 and trace['non_air_write_visits'] == 2


@pytest.mark.parametrize('block', ['air', 'cave_air', 'void_air', 'minecraft:cave_air', 'minecraft:void_air'])
def test_all_air_names_are_absence_and_noop_visits_count(block):
    actual, trace = execute_independent([set_block(block=block), cuboid(block=block)])
    assert actual == {} and trace['air_write_visits'] == 9 and trace['deleted_occupied_voxels'] == 0
    assert trace['first_occupied_operation_index'] is None and trace['last_zero_transition_operation_index'] is None
    assert trace['empty_classification'] == 'AIR_OR_CARVE_ONLY'


def test_xyz_and_last_write_preserve_exact_coordinates_and_materials():
    ops = [set_block(3, -2, 5, 'minecraft:stone'), set_block(3, -2, 5, 'custom/path-block'),
           set_block(-7, 8, 9, 'dirt')]
    grid, trace = execute_independent(ops)
    assert grid == {(3, -2, 5): 'custom/path-block', (-7, 8, 9): 'dirt'}
    assert trace['material_replacements'] == 1
    encoded = json.dumps([[-7, 8, 9, 'dirt'], [3, -2, 5, 'custom/path-block']], separators=(',', ':')).encode()
    assert grid_sha256(grid) == hashlib.sha256(encoded).hexdigest()


def test_independent_executor_does_not_delegate_to_old_core(monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('old core must not execute')
    monkeypatch.setattr(core, 'execute', forbidden)
    monkeypatch.setattr(core, 'validate_actions', forbidden)
    monkeypatch.setattr(core, 'material_id', forbidden)
    grid, trace = execute_independent([cuboid()], core.Limits())
    assert len(grid) == trace['max_occupied_voxels'] == 8


def test_independent_default_bounds_match_frozen_execution_bounds():
    limits = core.Limits()
    assert all(getattr(IndependentLimits(), key) == getattr(limits, key)
               for key in IndependentLimits.__dataclass_fields__)


@pytest.mark.parametrize('operations,limits,index,phase', [
    ([dict(set_block(), x=True)], core.Limits(), 0, 'validation'),
    ([dict(set_block(), x=1.0)], core.Limits(), 0, 'validation'),
    ([set_block(block='Stone')], core.Limits(), 0, 'validation'),
    ([set_block(block='stone[axis=y]')], core.Limits(), 0, 'validation'),
    ([dict(set_block(), extra=1)], core.Limits(), 0, 'validation'),
    ([set_block(x=100001)], core.Limits(), 0, 'validation'),
    ([set_block()], core.Limits(max_operations=0), None, 'validation'),
    ([cuboid()], core.Limits(max_touched_voxels=7), 0, 'validation'),
    ([cuboid()], core.Limits(max_voxels=7), 0, 'validation'),
    ([set_block(), set_block(x=1)], core.Limits(max_voxels=1), 1, 'execution'),
])
def test_original_rejections_and_bounds_are_not_relaxed(operations, limits, index, phase):
    with pytest.raises(core.ValidationError): core.execute(operations, limits)
    with pytest.raises(IndependentExecutionError) as caught: execute_independent(operations, limits)
    assert caught.value.operation_index == index and caught.value.phase == phase


def test_deletion_releases_live_capacity_but_repeated_visits_consume_touch_budget():
    ops = [set_block(), set_block(block='air'), set_block(x=1)]
    grid, trace = execute_independent(ops, core.Limits(max_voxels=1, max_touched_voxels=3))
    assert grid == {(1, 0, 0): 'stone'} and trace['touched_voxel_visits'] == 3
    with pytest.raises(IndependentExecutionError):
        execute_independent(ops, core.Limits(max_voxels=1, max_touched_voxels=2))


def test_fixed_small_random_programs_match_without_gt_or_repairs():
    rng = random.Random(20261020)
    for _ in range(40):
        operations = []
        for _ in range(12):
            op = rng.choice(('set', 'fill', 'carve')); a = tuple(rng.randint(-2, 2) for _ in range(3))
            b = tuple(rng.randint(-2, 2) for _ in range(3)); block = rng.choice(('stone', 'dirt', 'air', 'minecraft:void_air'))
            operations.append(set_block(*a, block) if op == 'set' else cuboid(op, a, b, block))
        grid, trace = execute_independent(operations)
        assert grid == core.execute(operations).voxels
        assert trace['touched_voxel_visits'] == trace['air_write_visits'] + trace['non_air_write_visits']


def _envelope(provider, operations):
    text = json.dumps({'operations': operations})
    if provider == 'openai':
        data = {'id': 'synthetic-openai-positive', 'model': adapter.MODELS[provider], 'status': 'completed',
                'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': text}]}],
                'usage': {'input_tokens': 1, 'output_tokens': 1}}
    else:
        data = {'id': 'synthetic-anthropic-positive', 'model': adapter.MODELS[provider], 'stop_reason': 'end_turn',
                'content': [{'type': 'text', 'text': text}], 'usage': {'input_tokens': 1, 'output_tokens': 1}}
    return json.dumps(data).encode(), text


@pytest.mark.parametrize('provider', ['openai', 'anthropic'])
@pytest.mark.parametrize('name,operations,count', [row for row in FIXTURES if row[2] > 0],
                         ids=[row[0] for row in FIXTURES if row[2] > 0])
def test_nonempty_examples_survive_real_goal015_extraction_to_evaluation(provider, name, operations, count, monkeypatch):
    # A hard guard ensures even an accidental transport call is an immediate failure.
    def no_send(*args, **kwargs): raise AssertionError('research API calls forbidden in diagnostic fixtures')
    monkeypatch.setattr(adapter, 'send_request', no_send)
    raw, text = _envelope(provider, operations)
    parsed = adapter.parse_response(provider, raw)
    assert parsed.complete and parsed.failure_kind is None and parsed.text == text
    validated = adapter.validate_output(parsed.text)
    independent_grid, _ = execute_independent(validated)
    record = {'call_id': 'goal015b-fixture-' + name, 'scene_id': 'synthetic-only', 'builder': provider,
              'condition': 'FULL_RECORD', 'repeat_index': 1, 'paired_block_id': 'synthetic-only-pair'}
    observation = dict(record, state='SETTLED', send_status='POST_RETURNED', response_received=True,
        completed_response=parsed.complete, usage_known=True, model_output_failure=False,
        failure_kind=parsed.failure_kind, response_text=parsed.text, response_model=parsed.model, http_status=200)
    result = evaluate.evaluate_one(record, observation, core.Grid(independent_grid))
    assert result['status'] == 'PROTOCOL_VALID' and result['evaluation_status'] == 'PASS'
    assert result['nonempty'] and not result['empty_output'] and result['final_voxel_count'] == count
    assert core.payload_to_grid(result['voxel_payload']).voxels == independent_grid
    assert result['metrics']['values']['aligned_occupancy_iou'] == 1.0
    assert result['metrics']['values']['aligned_surface_f1_at_1'] == 1.0
