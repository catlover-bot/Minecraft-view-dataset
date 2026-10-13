"""Independent global-search checks; no research API or synthetic success claim."""
from dataclasses import replace
from itertools import product
import random

import numpy as np
import pytest

from offline_evaluation.server_v2.alignment import (
    AlignmentResourceLimit, MODULUS, PRIMITIVE_ROOT, select_translation,
)
from offline_evaluation.server_v2.core import Grid, Limits, ValidationError, evaluate_pair


def brute_force(gt, pred):
    """Independent tiny-example enumerator over the full overlap-support cube."""
    if not gt or not pred:
        return (0, 0, 0), 0
    ranges = [range(min(g[i] for g in gt) - max(p[i] for p in pred),
                    max(g[i] for g in gt) - min(p[i] for p in pred) + 1)
              for i in range(3)]
    results = []
    for shift in product(*ranges):
        shifted = {(p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]) for p in pred}
        overlap = len(gt.intersection(shifted))
        results.append((-overlap, sum(abs(v) for v in shift), shift))
    negative_count, _, shift = min(results)
    return shift, -negative_count


@pytest.mark.parametrize("sparse_pairs", [0, 100_000])
def test_global_matches_independent_exhaustive_asymmetric_random_shapes(sparse_pairs):
    rng = random.Random(90210)
    limits = replace(Limits(), max_alignment_sparse_pairs=sparse_pairs)
    universe = list(product(range(-2, 3), range(-1, 3), range(-2, 2)))
    methods = set()
    for _ in range(45):
        gt = set(rng.sample(universe, rng.randint(1, 8)))
        pred = set(rng.sample(universe, rng.randint(1, 8)))
        expected_shift, expected_overlap = brute_force(gt, pred)
        result = select_translation(gt, pred, limits=limits)
        assert tuple(result["shift_xyz"]) == expected_shift
        assert result["maximum_occupancy_intersection"] == expected_overlap
        assert result["search_complete"] and not result["resource_limit_reached"]
        assert result["globally_optimal_unrestricted"]
        methods.add(result["method"])
    assert ("exact_integer_ntt_correlation" if sparse_pairs == 0
            else "exact_sparse_relative_difference_count") in methods


@pytest.mark.parametrize("translation", [(1, -1, 1), (20, -3, -17), (-100, 7, 80)])
def test_distant_known_translation_uses_verified_unique_shape_certificate(translation):
    gt = Grid({(-2, 3, 0): "stone", (0, 4, 1): "glass", (1, 3, 1): "wood"})
    result = evaluate_pair(gt, gt.shifted(translation), alignment_mode="strict_translation")
    assert result["alignment"]["shift_xyz"] == [-v for v in translation]
    assert result["alignment"]["method"] == "verified_identical_shape_upper_bound"
    assert result["alignment"]["search_complete"]
    assert result["aligned"]["occupancy_iou"] == 1
    assert result["aligned"]["material_aware_iou"] == 1
    assert result["aligned"]["total_normalized_repair_cost"] == 0


def test_far_independent_world_origins_are_restored_from_compact_arrays():
    gt = {(0, 0, 0), (2, 1, 0), (1, 3, 2), (4, 0, 2)}
    pred = {(-1, 0, 0), (1, 1, 0), (0, 3, 2), (7, 2, 1)}
    limits = replace(Limits(), max_alignment_sparse_pairs=0)
    local = select_translation(gt, pred, limits=limits)
    g_origin, p_origin = (10**25, -10**24, 10**23), (-10**26, 10**21, -10**20)
    far_gt = {tuple(c[i] + g_origin[i] for i in range(3)) for c in gt}
    far_pred = {tuple(c[i] + p_origin[i] for i in range(3)) for c in pred}
    far = select_translation(far_gt, far_pred, limits=limits)
    assert far["method"] == local["method"] == "exact_integer_ntt_correlation"
    assert far["maximum_occupancy_intersection"] == local["maximum_occupancy_intersection"]
    assert far["shift_xyz"] == [local["shift_xyz"][i] + g_origin[i] - p_origin[i] for i in range(3)]
    assert far["transform_cells"] == local["transform_cells"]
    assert far["selected_shift_verified"]


@pytest.mark.parametrize("sparse_pairs", [0, 100_000])
def test_translation_tie_is_resolved_in_world_coordinates_after_local_convolution(sparse_pairs):
    gt = {(0, 0, 0), (4, 0, 0)}
    pred = {(2, 0, 0), (9, 0, 0)}
    limits = replace(Limits(), max_alignment_sparse_pairs=sparse_pairs)
    assert select_translation(gt, pred, limits=limits)["shift_xyz"] == [-2, 0, 0]
    gt_far = {(x + 10, y, z) for x, y, z in gt}
    pred_far = {(x + 20, y, z) for x, y, z in pred}
    # The world-coordinate L1 optimum changes among tied local-shape maxima.
    result = select_translation(gt_far, pred_far, limits=limits)
    assert result["shift_xyz"] == [-8, 0, 0]
    assert tuple(result["shift_xyz"]) == brute_force(gt_far, pred_far)[0]


@pytest.mark.parametrize("sparse_pairs", [0, 100_000])
def test_tie_uses_world_L1_then_XYZ_and_never_material_favorable_shift(sparse_pairs):
    gt = Grid({(0, 0, 0): "stone", (2, 0, 0): "glass"})
    pred = Grid({(0, 0, 0): "brick", (1, 0, 0): "stone"})
    result = evaluate_pair(gt, pred, alignment_mode="strict_translation",
                           limits=replace(Limits(), max_alignment_sparse_pairs=sparse_pairs))
    assert result["alignment"]["shift_xyz"] == [0, 0, 0]
    assert result["aligned"]["material_aware_iou"] == 0
    assert result["aligned"]["replace_count"] == 1
    # Equal magnitude winners at -1 and +1 must choose XYZ lexicographic -1.
    tie = select_translation({(-1, 0, 0), (1, 0, 0)}, {(0, 0, 0)},
                             limits=replace(Limits(), max_alignment_sparse_pairs=sparse_pairs))
    assert tie["shift_xyz"] == [-1, 0, 0]


@pytest.mark.parametrize("gt,pred", [(set(), set()), (set(), {(9, 7, 1)}), ({(-4, 1, 0)}, set())])
def test_empty_sets_have_zero_shift_for_entire_unbounded_tie_domain(gt, pred):
    result = select_translation(gt, pred)
    assert result["shift_xyz"] == [0, 0, 0]
    assert result["maximum_occupancy_intersection"] == 0
    assert result["xyz_search_bounds"] is None
    assert result["method"] == "empty_set_exact"
    assert result["all_candidates_resolved"]


def test_partial_containment_at_zero_is_global_upper_bound_certificate():
    result = select_translation({(0, 0, 0), (1, 0, 0), (100, 1, 0)}, {(1, 0, 0)})
    assert result["shift_xyz"] == [0, 0, 0]
    assert result["maximum_occupancy_intersection"] == 1
    assert result["optimality_certificate"] == "minimum_count_upper_bound_at_unique_minimum_L1_shift"


def test_bounded_compatibility_and_difference_from_strict():
    gt = Grid({(0, 0, 0): "stone", (1, 2, 0): "glass"})
    pred = gt.shifted((20, -3, -17))
    bounded = evaluate_pair(gt, pred, alignment_radius=1)
    strict = evaluate_pair(gt, pred, alignment_radius=1, alignment_mode="strict_translation")
    assert bounded["metric_version"] == "server_v2.1"
    assert strict["metric_version"] == "server_v2.2_strict_translation"
    assert bounded["alignment"]["mode"] == "bounded"
    assert bounded["alignment"]["xyz_search_bounds"] == [[-1, 1]] * 3
    assert bounded["alignment"]["candidates_evaluated"] == 27
    assert bounded["alignment"]["all_candidates_processed"]
    assert bounded["aligned"]["occupancy_iou"] == 0
    assert strict["aligned"]["occupancy_iou"] == 1


@pytest.mark.parametrize("updates", [
    {"max_alignment_transform_cells": 1},
    {"max_alignment_transform_work": 1},
    {"max_alignment_memory_bytes": 1},
])
def test_integer_backend_resource_failures_are_explicit_and_incomplete(updates):
    gt = {(0, 0, 0), (2, 1, 0), (1, 3, 2)}
    pred = {(1, -1, 2), (3, 3, 1)}
    limits = replace(Limits(), max_alignment_sparse_pairs=0, **updates)
    with pytest.raises(AlignmentResourceLimit) as caught:
        select_translation(gt, pred, limits=limits)
    assert caught.value.metadata["status"] == "ALIGNMENT_RESOURCE_LIMIT"
    assert caught.value.metadata["mode"] == "strict_translation"
    assert caught.value.metadata["search_complete"] is False
    assert caught.value.metadata["resource_limit_reached"] is True
    assert "shift_xyz" not in caught.value.metadata


def test_sparse_resource_limit_and_coordinate_validation():
    with pytest.raises(AlignmentResourceLimit, match="work"):
        select_translation({(0, 0, 0), (4, 0, 0)}, {(1, 0, 0), (6, 0, 0)},
                           limits=replace(Limits(), max_alignment_work=0))
    for coordinates in [[(True, 0, 0)], [(1, 2)], [(1.0, 2, 3)]]:
        with pytest.raises(ValidationError):
            select_translation(coordinates, {(0, 0, 0)})
    with pytest.raises(ValidationError, match="mode"):
        select_translation(set(), set(), mode="approximate")


def test_huge_separation_with_different_shapes_fails_before_transform_allocation(monkeypatch):
    gt = {(0, 0, 0), (10**12, 0, 0)}
    pred = {(0, 0, 0), (2 * 10**12, 0, 0)}
    monkeypatch.setattr(np, "zeros", lambda *args, **kw: pytest.fail("transform allocated before resource gate"))
    with pytest.raises(AlignmentResourceLimit):
        select_translation(gt, pred, limits=replace(Limits(), max_alignment_sparse_pairs=0))


def test_modular_root_support_and_exact_integer_bound_are_recorded():
    gt = {(0, 0, 0), (2, 1, 0), (1, 3, 2)}
    pred = {(1, -1, 2), (3, 3, 1)}
    result = select_translation(gt, pred, limits=replace(Limits(), max_alignment_sparse_pairs=0))
    n = result["transform_cells"]
    root = pow(PRIMITIVE_ROOT, (MODULUS - 1) // n, MODULUS)
    assert pow(root, n, MODULUS) == 1
    assert pow(root, n // 2, MODULUS) != 1
    assert result["max_possible_overlap"] < result["modular_prime"]
    assert (MODULUS - 1) ** 2 <= np.iinfo(np.int64).max
    assert result["arithmetic"] == "exact_int64_modular_no_floating_point"
    assert result["candidates_evaluated"] == np.prod(result["correlation_output_shape_xyz"])
    assert result["selected_shift_verified"]


def test_large_transform_matches_independent_integer_coordinate_pair_oracle():
    rng = random.Random(271828)
    gt = {tuple(rng.randrange(31) for _ in range(3)) for _ in range(60)}
    pred = {tuple(rng.randrange(41) for _ in range(3)) for _ in range(55)}
    counts = {}
    for g in gt:
        for p in pred:
            shift = (g[0] - p[0], g[1] - p[1], g[2] - p[2])
            counts[shift] = counts.get(shift, 0) + 1
    expected = min(counts, key=lambda s: (-counts[s], abs(s[0]) + abs(s[1]) + abs(s[2]), s))
    result = select_translation(gt, pred, limits=replace(Limits(), max_alignment_sparse_pairs=0))
    assert result["method"] == "exact_integer_ntt_correlation"
    assert result["transform_cells"] == 524_288
    assert result["shift_xyz"] == list(expected)
    assert result["maximum_occupancy_intersection"] == counts[expected]


def test_local_grid_frame_is_transformed_once_before_alignment():
    gt = Grid({(4, 5, 6): "stone", (6, 4, 7): "glass"})
    pred = Grid({(0, 0, 0): "stone", (2, -1, 1): "glass"},
                coordinate_frame="local", origin=(100, 20, -10))
    result = evaluate_pair(gt, pred, alignment_mode="strict_translation")
    assert result["alignment"]["shift_xyz"] == [-96, -15, 16]
    assert result["aligned"]["occupancy_iou"] == 1
    assert result["aligned"]["material_aware_iou"] == 1
