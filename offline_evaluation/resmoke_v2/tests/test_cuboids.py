import copy
import json
from dataclasses import replace

import pytest

from offline_evaluation.resmoke_v2.cuboids import (
    decode_exact, encode_exact, exact_builder_text, reference_actions,
)
from offline_evaluation.server_v2.core import Grid, Limits, ValidationError, execute, parse_actions
from offline_evaluation.server_v2.oracle import decode_exact_spec, encode_exact_spec


def fixture_grid():
    cells = {(x, y, z): "stone" for x in range(-1, 3) for y in range(2, 5) for z in range(-3, 2)}
    del cells[(0, 3, -1)]  # An internal hole must remain air.
    cells[(1, 3, 0)] = "glass"
    cells[(5, 2, -2)] = "oak_planks"  # A disconnected component cannot bridge the gap.
    return Grid(cells, origin=(-5, 1, -8))


def test_single_solid_cuboid_and_inclusive_endpoints():
    grid = Grid({(x, y, z): "stone" for x in range(-1, 3) for y in range(2, 5) for z in range(-3, 2)})
    spec = encode_exact(grid)
    assert spec["cuboids"] == [[-1, 2, -3, 2, 4, 1, 0]]
    assert decode_exact(spec).to_dict() == grid.to_dict()
    assert len(execute(reference_actions(spec)["operations"]).voxels) == 60


def test_holes_material_boundaries_and_disconnected_components_roundtrip():
    grid = fixture_grid()
    spec = encode_exact(grid)
    assert decode_exact(json.dumps(spec)).to_dict() == grid.to_dict()
    actions = parse_actions(json.dumps(reference_actions(spec)))
    actual = execute(actions, origin=grid.origin)
    assert actual.to_dict() == grid.to_dict()
    assert (0, 3, -1) not in actual.voxels
    assert (4, 2, -2) not in actual.voxels
    assert actual.voxels[(1, 3, 0)] == "glass"
    assert encode_exact(Grid(dict(reversed(list(grid.voxels.items()))), origin=grid.origin)) == spec
    assert encode_exact(decode_exact(spec)) == spec


def test_fixed_axis_growth_rule_is_z_then_x_then_y():
    grid = Grid({(0, 0, 0): "stone", (0, 0, 1): "stone", (1, 0, 0): "stone",
                 (0, 1, 0): "stone", (0, 1, 1): "stone"})
    assert encode_exact(grid)["cuboids"] == [[0, 0, 0, 0, 1, 1, 0], [1, 0, 0, 1, 0, 0, 0]]


@pytest.mark.parametrize("frame", ["world", "local"])
def test_nonzero_origin_is_applied_only_for_local_coordinates(frame):
    grid = Grid({(0, 0, 0): "stone", (0, 0, 1): "stone"}, frame, (-10, 8, 21))
    spec = encode_exact(grid)
    assert decode_exact(spec).to_dict() == grid.to_dict()
    result = execute(reference_actions(spec)["operations"], origin=grid.origin)
    assert result.to_dict() == grid.as_world().to_dict()


def test_empty_spec_remains_empty_and_no_control_is_called_a_model_output():
    spec = encode_exact(Grid())
    assert spec["palette"] == spec["cuboids"] == []
    assert decode_exact(spec).voxels == {}
    assert reference_actions(spec) == {"operations": []}
    body = exact_builder_text(spec)
    assert "discrete-specification-to-actions diagnostic" in body
    assert "operations" not in body  # No pre-generated reference action list is provided.


@pytest.mark.parametrize("second_material", [0, 1])
def test_overlap_rejected_even_if_material_matches(second_material):
    spec = encode_exact(Grid({(0, 0, 0): "glass", (3, 0, 0): "stone"}))
    spec["cuboids"] = [[0, 0, 0, 2, 0, 0, 0], [1, 0, 0, 3, 0, 0, second_material]]
    with pytest.raises(ValidationError, match="overlapping"):
        decode_exact(spec)


@pytest.mark.parametrize("mutation", [
    lambda s: s["cuboids"][0].__setitem__(0, True),
    lambda s: s["cuboids"][0].__setitem__(6, True),
    lambda s: s["cuboids"][0].__setitem__(6, 500),
    lambda s: s["cuboids"][0].__setitem__(3, -10),
    lambda s: s["cuboids"].reverse(),
    lambda s: s["origin_xyz"].__setitem__(0, False),
    lambda s: s["palette"].append("air"),
    lambda s: s["palette"].append(s["palette"][0]),
    lambda s: s["palette"].__setitem__(0, "minecraft:glass"),
    lambda s: s.__setitem__("endpoints", "exclusive"),
    lambda s: s.__setitem__("coordinate_frame", "automatic"),
    lambda s: s.__setitem__("extra", "ignored"),
])
def test_invalid_or_ambiguous_specs_rejected(mutation):
    spec = copy.deepcopy(encode_exact(fixture_grid()))
    mutation(spec)
    with pytest.raises(ValidationError):
        decode_exact(spec)


def test_duplicate_json_key_not_silently_overwritten():
    raw = json.dumps(encode_exact(Grid()))
    with pytest.raises(ValidationError, match="duplicate JSON key"):
        decode_exact(raw.replace('"cuboids": []', '"cuboids": [], "cuboids": []'))


def test_spec_metadata_mutation_cannot_change_encoder_or_decoder_semantics():
    mutated = encode_exact(Grid())
    mutated["cuboid_fields"][0] = "z1"
    with pytest.raises(ValidationError, match="unsupported"):
        decode_exact(mutated)
    assert encode_exact(Grid())["cuboid_fields"][0] == "x1"


def test_limits_fail_before_oversized_expansion_and_do_not_truncate():
    spec = encode_exact(Grid({(0, 0, 0): "stone"}))
    spec["cuboids"][0][3] = 1000
    with pytest.raises(ValidationError, match="expanded voxels"):
        decode_exact(spec, replace(Limits(), max_voxels=10))
    with pytest.raises(ValidationError, match="operation limit"):
        encode_exact(fixture_grid(), replace(Limits(), max_operations=1))
    with pytest.raises(ValidationError, match="byte limit"):
        encode_exact(fixture_grid(), replace(Limits(), max_json_bytes=20))
    with pytest.raises(ValidationError, match="coordinate bound"):
        reference_actions(encode_exact(Grid({(10, 0, 0): "stone"}, "local", (100000, 0, 0))))


def test_old_z_run_representation_remains_a_separate_unchanged_protocol():
    grid = Grid({(x, y, z): "stone" for x in range(2) for y in range(2) for z in range(3)})
    old_before = encode_exact_spec(grid)
    assert old_before["schema"] == "server_v2_exact_spec_v1"
    assert old_before["runs"] == [[0, 0, 0, 2, 0], [0, 1, 0, 2, 0],
                                   [1, 0, 0, 2, 0], [1, 1, 0, 2, 0]]
    new = encode_exact(grid)
    assert len(new["cuboids"]) == 1
    assert encode_exact_spec(grid) == old_before
    assert decode_exact_spec(old_before).to_dict() == decode_exact(new).to_dict()
