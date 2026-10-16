"""Synthetic-only request, camera, path, and source-binding guard tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from offline_evaluation.analysis.goal010 import protocol as p


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding='utf-8')


def fixture_source(root, scene='building_000', provider='openai'):
    bdir = root / 'datasets' / scene
    voxel = bdir / 'gt/voxels.npy'; bbox = bdir / 'gt/bbox.json'
    if not voxel.exists():
        voxel.parent.mkdir(parents=True); voxel.write_bytes(b'synthetic-not-a-real-grid')
        write(bbox, {'xmin': -1, 'xmax': 2, 'ymin': 4, 'ymax': 7, 'zmin': 0, 'zmax': 5})
        views = []
        for index in range(12):
            image = bdir / f'images/view{index:02d}.png'; image.parent.mkdir(exist_ok=True)
            Image.new('RGB', (12, 8), (index, 0, 0)).save(image)
            views.append({'path': str(image.relative_to(bdir)), 'x': 10., 'y': 6., 'z': 0.,
                'actual_x': 10., 'actual_y': 6., 'actual_z': 0., 'yaw': index * 30., 'pitch': 0.,
                'fov': 70., 'width': 12, 'height': 8})
        write(bdir / 'meta.json', {'views': views, 'generation': {'origin': {'x': 0, 'y': 4, 'z': 0}},
            'bbox': json.loads(bbox.read_text()), 'created_at': '2026-01-01T00:00:00+00:00',
            'camera_planner': {'fov': 70, 'radius': 10., 'target': {'x': 0, 'y': 4, 'z': 0}}})
    out = root / 'outputs' / scene / provider; description_path = out / 'description.json'
    description = {'building': scene, 'provider': provider, 'model': provider + '-fixture',
                   'summary': 'synthetic', 'created_at': '2026-02-01T00:00:00+00:00'}
    write(description_path, description)
    request = {'building': scene, 'provider': provider, 'model': provider + '-fixture',
        'images': [f'images/view{i:02d}.png' for i in (0, 2, 4, 6, 10, 11)],
        'prompt': 'same synthetic prompt', 'system_prompt': 'same synthetic system', 'temperature': .2,
        'max_tokens': 4096 if provider == 'gemini' else 1800, 'created_at': '2026-02-01T00:00:01+00:00'}
    write(out / 'description.request.json', request)
    digest = p.sha256(description_path)
    return {'scene_id': scene, 'description_provider': provider, 'description_model_id': provider + '-fixture',
        'source_path': str(description_path), 'source_sha256': digest, 'input_id': scene + provider + 'O',
        'input_sha256': digest, 'j_input_id': scene + provider + 'J', 'j_input_sha256': '1' * 64,
        'typed_tree_sha256': '2' * 64, 'O_J_typed_tree_equal': True,
        'gt_voxels_path': str(voxel), 'gt_bbox_path': str(bbox), 'gt_descriptor_sha256': '3' * 64,
        'gt_descriptor': {'voxels_sha256': p.sha256(voxel), 'bbox_sha256': p.sha256(bbox), 'grid_sha256': '4' * 64},
        'builder_observations': {}}


def fixture_package(root):
    raw = [{'scene_id': f'building_{i:03d}', 'gt_occupied': i + 1} for i in range(100)]
    cohort = p.freeze_ranked_split(raw, seed=20261013)
    cohort.update(population_scenes=100, selection_uses_model_results=False)
    sources, slots = [], []
    for scene in [r['scene_id'] for r in cohort['scenes'] if r['split'] == 'evaluation']:
        for provider in p.PROVIDERS:
            source = fixture_source(root, scene, provider); sources.append(source)
            for builder in ('openai', 'anthropic'):
                for condition in ('O', 'J'):
                    prefix = '' if condition == 'O' else 'j_'
                    row = {'scene_id': scene, 'description_model': provider, 'provider': builder, 'condition': condition,
                        'global_schedule_index': len(slots), 'attempt_id': str(len(slots)), 'status': 'success',
                        'source_plan_identity': '5' * 64, 'source_gt_descriptor_hash': source['gt_descriptor_sha256'],
                        'gt_grid_sha256': source['gt_descriptor']['grid_sha256'],
                        'input_id': source[prefix + 'input_id'], 'input_sha256': source[prefix + 'input_sha256']}
                    slots.append(row)
                    if condition == 'O': source['builder_observations'][builder] = {k: row[k] for k in ('attempt_id', 'global_schedule_index', 'status', 'source_plan_identity')}
        for builder in ('openai', 'anthropic'):
            for condition in ('P', 'E'):
                slots.append({'scene_id': scene, 'description_model': None, 'provider': builder, 'condition': condition,
                              'global_schedule_index': len(slots)})
    for name in p.SOURCE_CODE:
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('synthetic source code\n')
    return {'descriptions': sources, 'cohort': cohort, 'slots': slots}


class TestProtocol(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='goal010-protocol-fixture-')
        self.root = Path(self.tmp.name)
        self.source = fixture_source(self.root)

    def tearDown(self): self.tmp.cleanup()

    def test_actual_six_vs_candidate_twelve_and_timestamp_semantics(self):
        row, images = p.audit_description(self.source, root=self.root, inventory={}, scene_cache={})
        self.assertEqual(row['actual_request_image_count'], 6)
        self.assertEqual(row['saved_candidate_view_count'], 12)
        self.assertEqual(row['actual_meta_view_indices_ordered'], [0, 2, 4, 6, 10, 11])
        self.assertEqual(len(images), 6)
        self.assertIsNone(row['api_request_start_utc']); self.assertIsNone(row['api_request_end_utc'])
        self.assertEqual(row['request_record_created_at_utc'], '2026-02-01T00:00:01+00:00')
        self.assertFalse(row['seed_recorded']); self.assertIsNone(images[0]['actual_yaw'])

    def test_only_allowlisted_metadata_is_exported(self):
        request = Path(self.source['source_path']).with_name('description.request.json')
        value = json.loads(request.read_text()); value.update(prompt='private fixture prompt', system_prompt='private fixture system',
                                                            unrelated_metadata='DO_NOT_EXPORT_THIS_FIXTURE_VALUE')
        write(request, value)
        row, images = p.audit_description(self.source, root=self.root, inventory={}, scene_cache={})
        exported = json.dumps([row, images])
        self.assertNotIn('private fixture', exported); self.assertNotIn('DO_NOT_EXPORT', exported)
        self.assertIn(p._text_hash('private fixture prompt'), exported)

    def test_request_image_traversal_is_rejected(self):
        path = Path(self.source['source_path']).with_name('description.request.json')
        value = json.loads(path.read_text()); value['images'][0] = '../../outside.png'; write(path, value)
        with self.assertRaisesRegex(ValueError, 'unsafe relative'): p.audit_description(self.source, root=self.root, inventory={}, scene_cache={})

    def test_duplicate_request_images_are_rejected(self):
        path = Path(self.source['source_path']).with_name('description.request.json')
        value = json.loads(path.read_text()); value['images'][1] = value['images'][0]; write(path, value)
        with self.assertRaisesRegex(ValueError, 'unique actual request'): p.audit_description(self.source, root=self.root, inventory={}, scene_cache={})

    def test_source_hash_change_is_rejected(self):
        Path(self.source['source_path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'description hash'): p.audit_description(self.source, root=self.root, inventory={}, scene_cache={})

    def test_model_identity_mismatch_is_rejected(self):
        path = Path(self.source['source_path']).with_name('description.request.json')
        value = json.loads(path.read_text()); value['model'] = 'different-fixture'; write(path, value)
        with self.assertRaisesRegex(ValueError, 'identity mismatch'): p.audit_description(self.source, root=self.root, inventory={}, scene_cache={})

    def test_image_resolution_mismatch_is_rejected(self):
        path = Path(self.source['gt_voxels_path']).parent.parent / 'meta.json'
        value = json.loads(path.read_text()); value['views'][0]['width'] = 13; write(path, value)
        with self.assertRaisesRegex(ValueError, 'dimensions/format'): p.audit_description(self.source, root=self.root, inventory={}, scene_cache={})

    def test_symlink_image_escape_is_rejected(self):
        path = Path(self.source['gt_voxels_path']).parent.parent / 'images/view00.png'
        other = self.root / 'outside.png'; path.rename(other); path.symlink_to(other)
        with self.assertRaisesRegex(ValueError, 'regular file'): p.audit_description(self.source, root=self.root, inventory={}, scene_cache={})

    def test_request_settings_are_not_default_imputed(self):
        path = Path(self.source['source_path']).with_name('description.request.json')
        value = json.loads(path.read_text()); del value['temperature']; write(path, value)
        row, _ = p.audit_description(self.source, root=self.root, inventory={}, scene_cache={})
        self.assertIsNone(row['temperature']); self.assertFalse(row['temperature_recorded'])

    def test_camera_nan_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'nonfinite'): p._camera({'yaw': float('nan')})


class TestCompleteProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='goal010-complete-protocol-fixture-'); cls.root = Path(cls.tmp.name)
        cls.package = fixture_package(cls.root)
        with patch('socket.create_connection', side_effect=AssertionError('network forbidden')):
            cls.result = p.audit_protocol(cls.package, cls.root / 'audit', root=cls.root)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_complete180_request_counts_and_budget_confound(self):
        self.assertEqual(self.result['request_records'], 180); self.assertEqual(self.result['independent_buildings'], 60)
        self.assertEqual(self.result['request_image_references'], 1080); self.assertEqual(self.result['unique_saved_images'], 720)
        self.assertTrue(self.result['all_three_providers_share_ordered_image_bytes_per_scene'])
        self.assertEqual(self.result['provider_settings']['gemini']['max_tokens'], [4096])
        self.assertEqual(self.result['provider_settings']['openai']['max_tokens'], [1800])
        self.assertFalse(self.result['causal_model_ability_comparison_supported'])

    def test_601030_selection_replay_and_independent_unit(self):
        result = p.audit_selection(self.package['cohort'])
        self.assertEqual(result['split_counts'], {'evaluation': 60, 'smoke': 10, 'rest': 30})
        self.assertTrue(result['replay_exact'])
        broken = copy.deepcopy(self.package['cohort']); broken['scenes'][0]['stratum_draw_order'] = 999
        with self.assertRaisesRegex(ValueError, 'selection replay mismatch'): p.audit_selection(broken)

    def test_selected_j_input_hash_mismatch_fails_closed(self):
        slots = copy.deepcopy(self.package['slots']); slots[1]['input_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'slot input/GT'): p.bind_selected_slots(self.package['descriptions'], slots)

    def test_all_metadata_is_not_projection_validation(self):
        result = p.visibility_feasibility(self.root / 'audit/PROTOCOL_MANIFEST.json', self.root / 'visibility')
        self.assertEqual(result['status'], 'NOT_VALIDATED_NO_VISIBILITY_CONCLUSION')
        self.assertEqual(result['source_binding'], 'PASS'); self.assertEqual(result['recorded_camera_field_counts']['fov'], 720)
        self.assertIsNone(result['visible_voxel_fraction']); self.assertIsNone(result['six_vs_twelve_visibility_difference'])
        self.assertFalse(result['independent_pixel_projection_validation_performed'])

    def test_visibility_gate_stops_on_modified_protocol_product(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'manifest.json'; value = copy.deepcopy(self.result)
            value['products'] = {'changed.json': {'bytes': 2, 'sha256': '0' * 64}}
            write(path, value); (path.parent / 'changed.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'frozen protocol output changed'):
                p.visibility_feasibility(path, path.parent / 'new')

    def test_existing_destination_is_not_overwritten(self):
        with self.assertRaises(FileExistsError): p.audit_protocol(self.package, self.root / 'audit', root=self.root)


if __name__ == '__main__': unittest.main()
