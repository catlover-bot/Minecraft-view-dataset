"""Small exhaustive fixtures for the independent summed-volume checker."""
import itertools
import random
from dataclasses import replace
import unittest
from unittest.mock import patch

from offline_evaluation.analysis.goal010.independent_geometry import check_box, IndependentGeometryResourceLimit
from offline_evaluation.server_v2.core import Grid, Limits, evaluate_pair


def box(origin, dims, shell=False, material='stone'):
    points = itertools.product(*(range(origin[i], origin[i] + dims[i]) for i in range(3)))
    return Grid({c: material for c in points if not shell or any(c[i] in (origin[i], origin[i] + dims[i] - 1) for i in range(3))})


def brute(gt, pred):
    if not gt.voxels or not pred.voxels: return 0, (0, 0, 0)
    lower = [min(c[i] for c in gt.voxels) - max(c[i] for c in pred.voxels) for i in range(3)]
    upper = [max(c[i] for c in gt.voxels) - min(c[i] for c in pred.voxels) for i in range(3)]
    values = []
    for shift in itertools.product(*(range(lower[i], upper[i] + 1) for i in range(3))):
        count = sum(tuple(c[i] + shift[i] for i in range(3)) in gt.voxels for c in pred.voxels)
        values.append((-count, sum(abs(v) for v in shift), shift))
    result = min(values)
    return -result[0], result[2]


class TestIndependentGeometry(unittest.TestCase):
    def assert_exact(self, gt, pred, structure):
        result = check_box(gt, pred, structure)
        overlap, shift = brute(gt, pred)
        self.assertEqual(result['aligned_intersection'], overlap)
        self.assertEqual(tuple(result['shift_xyz']), shift)
        absolute = len(set(gt.voxels) & set(pred.voxels))
        self.assertEqual(result['absolute_intersection'], absolute)
        expected_iou = overlap / (len(gt.voxels) + len(pred.voxels) - overlap) if gt.voxels or pred.voxels else 1.
        self.assertEqual(result['aligned_occupancy_iou'], expected_iou)
        original = evaluate_pair(gt, pred, alignment_mode='strict_translation')
        self.assertEqual(result['shift_xyz'], original['alignment']['shift_xyz'])
        self.assertEqual(result['aligned_occupancy_iou'], original['aligned']['occupancy_iou'])
        self.assertEqual(result['absolute_occupancy_iou'], original['absolute']['occupancy_iou'])
        return result

    def test_filled_box_negative_coordinates_and_distant_origin(self):
        gt = box((-9, 7, -3), (2, 3, 4)); pred = box((12, -8, 21), (2, 3, 4))
        result = self.assert_exact(gt, pred, 'filled_box')
        self.assertEqual(result['shift_xyz'], [-21, 15, -24]); self.assertEqual(result['aligned_occupancy_iou'], 1.)

    def test_shell_outer_minus_inner_is_exact(self):
        gt = Grid({(0, 0, 0): 'glass', (1, 1, 1): 'stone', (2, 2, 2): 'wood', (-1, 1, 1): 'brick'})
        pred = box((0, 0, 0), (3, 3, 3), shell=True)
        result = self.assert_exact(gt, pred, 'six_face_shell')
        self.assertEqual(result['absolute_intersection'], 2)
        self.assertIn('outer_minus_inner', result['method'])

    def test_width_one_or_two_shell_duplicates_are_set_cells(self):
        gt = Grid({(-2, -1, 0): 'stone', (0, 2, 3): 'stone', (1, 1, 1): 'stone'})
        for axis in range(3):
            for width in (1, 2):
                dims = [3, 3, 3]; dims[axis] = width
                shell = box((-1, -2, -3), dims, shell=True)
                self.assertEqual(len(shell.voxels), dims[0] * dims[1] * dims[2])
                self.assert_exact(gt, shell, 'six_face_shell')

    def test_minimum_l1_precedes_xyz_lexicographic(self):
        gt = Grid({(-2, 0, 0): 'stone', (1, 0, 0): 'stone'})
        result = self.assert_exact(gt, box((0, 0, 0), (1, 1, 1)), 'filled_box')
        self.assertEqual(result['shift_xyz'], [1, 0, 0])

    def test_exact_l1_tie_uses_xyz_lexicographic(self):
        gt = Grid({(-1, 0, 0): 'stone', (0, -1, 0): 'stone', (0, 0, -1): 'stone', (1, 0, 0): 'stone'})
        result = self.assert_exact(gt, box((0, 0, 0), (1, 1, 1)), 'filled_box')
        self.assertEqual(result['shift_xyz'], [-1, 0, 0])

    def test_zero_is_selected_among_positive_overlap_ties(self):
        result = self.assert_exact(box((-2, -2, -2), (5, 5, 5)), box((0, 0, 0), (1, 1, 1)), 'filled_box')
        self.assertEqual(result['shift_xyz'], [0, 0, 0])

    def test_empty_sets_have_zero_shift_and_no_dense_allocation(self):
        for gt, pred in [(Grid({}), box((5, 5, 5), (2, 2, 2))), (box((0, 0, 0), (1, 1, 1)), Grid({})), (Grid({}), Grid({}))]:
            with patch('offline_evaluation.analysis.goal010.independent_geometry.np.zeros', side_effect=AssertionError('allocation')):
                result = check_box(gt, pred, 'filled_box')
            self.assertEqual(result['shift_xyz'], [0, 0, 0]); self.assertEqual(result['aligned_intersection'], 0)
            self.assertEqual(result['aligned_occupancy_iou'], 1. if not gt.voxels and not pred.voxels else 0.)

    def test_material_labels_do_not_affect_any_geometry_result(self):
        gt = box((-1, 0, 2), (3, 3, 3), shell=True, material='glass')
        a = check_box(gt, box((0, 0, 0), (2, 3, 2), material='stone'), 'filled_box')
        b = check_box(gt, box((0, 0, 0), (2, 3, 2), material='brick'), 'filled_box')
        self.assertEqual(a, b); self.assertFalse(a['material_labels_used'])

    def test_random_sparse_gt_matches_exhaustive_and_production(self):
        rng = random.Random(20261017)
        universe = list(itertools.product(range(-2, 3), repeat=3))
        for index in range(24):
            gt = Grid({c: 'stone' for c in rng.sample(universe, rng.randint(1, 15))})
            dims = tuple(rng.randint(1, 4) for _ in range(3)); origin = tuple(rng.randint(-3, 3) for _ in range(3))
            shell = index % 2 == 0
            self.assert_exact(gt, box(origin, dims, shell=shell), 'six_face_shell' if shell else 'filled_box')

    def test_filled_prediction_with_missing_voxel_is_rejected(self):
        pred = box((0, 0, 0), (3, 3, 3)); del pred.voxels[(1, 1, 1)]
        with self.assertRaisesRegex(ValueError, 'not the declared complete'): check_box(Grid({}), pred, 'filled_box')

    def test_shell_interior_replacement_is_rejected(self):
        pred = box((0, 0, 0), (3, 3, 3), shell=True); del pred.voxels[(1, 0, 1)]; pred.voxels[(1, 1, 1)] = 'stone'
        with self.assertRaisesRegex(ValueError, 'interior voxel'): check_box(Grid({}), pred, 'six_face_shell')

    def test_local_grid_is_rejected_to_prevent_origin_double_application(self):
        pred = Grid({(0, 0, 0): 'stone'}, 'local', (-4, 3, 2))
        with self.assertRaisesRegex(ValueError, 'world XYZ'): check_box(Grid({}), pred, 'filled_box')

    def test_resource_limits_fail_before_dense_allocation(self):
        gt = Grid({(-4, -4, -4): 'stone', (4, 4, 4): 'stone'}); pred = box((0, 0, 0), (3, 3, 3))
        for limits in [replace(Limits(), max_alignment_transform_cells=1),
                       replace(Limits(), max_alignment_memory_bytes=1),
                       replace(Limits(), max_numpy_bytes=1),
                       replace(Limits(), max_alignment_work=1),
                       replace(Limits(), max_alignment_transform_work=1)]:
            with patch('offline_evaluation.analysis.goal010.independent_geometry.np.zeros', side_effect=AssertionError('allocation')):
                with self.assertRaises(IndependentGeometryResourceLimit) as caught: check_box(gt, pred, 'filled_box', limits)
            self.assertFalse(caught.exception.metadata['search_complete'])
            self.assertFalse(caught.exception.metadata['approximate_result_returned'])

    def test_occupied_count_and_coordinate_caps_are_enforced(self):
        pred = box((0, 0, 0), (2, 1, 1))
        with self.assertRaises(IndependentGeometryResourceLimit): check_box(Grid({}), pred, 'filled_box', replace(Limits(), max_voxels=1))
        with self.assertRaises(IndependentGeometryResourceLimit): check_box(Grid({(100001, 0, 0): 'stone'}), pred, 'filled_box')

    def test_world_origin_metadata_is_not_added_twice(self):
        gt = Grid({(2, 3, 4): 'stone'}, origin=(100, 200, 300))
        pred = Grid({(2, 3, 4): 'stone'}, origin=(-100, -200, -300))
        result = self.assert_exact(gt, pred, 'filled_box')
        self.assertEqual(result['absolute_occupancy_iou'], 1.); self.assertEqual(result['shift_xyz'], [0, 0, 0])


if __name__ == '__main__': unittest.main()
