"""Synthetic-only fixed selection, exact surface mesh, and publication checks."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from offline_evaluation.analysis.goal008 import examples as ex

DEFINITION = Path(__file__).resolve().parents[4] / 'reports/server_only/goal008/example_definition.md'
GATE = {'status': 'PASS', 'input_kind': 'fixture'}


def fixture():
    slots, gt = [], {}
    scores = {'scene_a': (.125, 3., .2, 1.), 'scene_b': (.25, 4., .2, 1.),
              'scene_c': (.5, 5., .2, 1.), 'scene_d': (.75, 10., .2, 1.)}
    for scene, (oi, om, pi, pm) in scores.items():
        gt[scene] = {'verified': True, 'path': '/fixture/gt', 'sha256': 'a' * 64,
                     'bbox_path': '/fixture/bbox', 'bbox_sha256': 'b' * 64}
        for c in ('O', 'J', 'P', 'E'):
            failed = scene == 'scene_d' and c == 'E'
            slots.append({'scene_id': scene, 'provider': 'openai', 'description_model': 'openai' if c in 'OJ' else None,
                          'condition': c, 'input_id': scene + c, 'input_sha256': 'c' * 64, 'attempt_id': scene + c + 'attempt',
                          'global_schedule_index': len(slots), 'status': 'failed' if failed else 'success',
                          'failure_kind': 'truncation' if failed else None,
                          'metrics': None if failed else {'metric_version': ex.METRIC_VERSION, 'aligned': {'occupancy_iou': pi if c == 'P' else oi, 'dimension_mae': pm if c == 'P' else om}}})
    return slots, gt


def selection():
    rows, gt = fixture()
    return ex.select_examples(rows, gt, definition_path=DEFINITION, audit_gate=GATE)


def packages(selected):
    result = []
    for row in selected['selected']:
        panels = {'GT': {'status': 'success', 'source': {'path': '/fixture/gt', 'sha256': 'a' * 64},
                         'world_voxels': [[0, 0, 0, 'stone'], [1, 0, 0, 'stone'], [0, 1, 0, 'glass']]}}
        for name in ('O', 'J', 'P', 'E'):
            slot = row['slots'][name]
            panels[name] = {k: copy.deepcopy(slot[k]) for k in ('status', 'failure_kind', 'attempt_id', 'input_id', 'input_sha256', 'metrics')}
            panels[name]['source'] = {'path': '/fixture/' + name, 'sha256': 'b' * 64}
            panels[name]['world_voxels'] = [[-2, 0, 0, 'oak_planks'], [1, 3, 2, 'stone']] if slot['status'] == 'success' else None
        result.append({'scene_id': row['scene_id'], 'panels': panels, 'description': {'path': '/fixture/description.json', 'sha256': 'c' * 64,
                       'field': 'summary', 'excerpt': 'Literal fixture description.', 'full_field_characters': 28, 'excerpt_characters': 28, 'truncated': False},
                       'gt_bbox_source': {'path': '/fixture/bbox', 'sha256': 'd' * 64, 'bbox': {'xmin': 0, 'xmax': 1}}})
    return result


class TestSelection(unittest.TestCase):
    def test_frozen_definition_digest(self):
        self.assertEqual(hashlib.sha256(DEFINITION.read_bytes()).hexdigest(), ex.DEFINITION_SHA256)

    def test_actual_gate_precedes_score_access(self):
        class Forbidden:
            def __iter__(self):raise AssertionError('actual scores accessed')
        with self.assertRaisesRegex(ValueError, 'Phase0 and Phase1'):
            ex.select_examples(Forbidden(), {}, definition_path=DEFINITION, audit_gate={'status': 'PASS', 'input_kind': 'verified_sources', 'phase0_status': 'PASS', 'phase1_status': 'PENDING'})

    def test_fixed_rank_and_distinct_no_replacement(self):
        s = selection()
        self.assertEqual(s['eligible_O_iou_median'], .375)
        self.assertEqual([x['scene_id'] for x in s['selected']], ['scene_b', 'scene_a', 'scene_d'])
        self.assertEqual(s['selected'][2]['slots']['E']['failure_kind'], 'truncation')

    def test_wrong_roles_never_change_rank(self):
        rows, gt = fixture()
        wrong = copy.deepcopy(rows[0]); wrong.update(provider='anthropic', metrics={'metric_version': ex.METRIC_VERSION, 'aligned': {'occupancy_iou': 0, 'dimension_mae': 100}})
        wrong2 = copy.deepcopy(rows[1]); wrong2['description_model'] = 'gemini'
        s = ex.select_examples([wrong, wrong2, *rows], gt, definition_path=DEFINITION, audit_gate=GATE)
        self.assertEqual([x['scene_id'] for x in s['selected']], ['scene_b', 'scene_a', 'scene_d'])

    def test_invalid_or_missing_metric_is_excluded_not_zero(self):
        rows, gt = fixture();rows[0].update(status='operational_unmeasurable', metrics=None, failure_kind='http_failure')
        s = ex.select_examples(rows, gt, definition_path=DEFINITION, audit_gate=GATE)
        self.assertEqual(s['eligible_count'], 3)
        self.assertEqual(s['excluded_candidates'][0]['scene_id'], 'scene_a')
        self.assertNotIn('scene_a', [x['scene_id'] for x in s['selected']])

    def test_no_threshold_relaxation(self):
        rows, gt = fixture()
        for row in rows:
            if row['condition'] == 'P':row['metrics']['aligned']['occupancy_iou'] = .20000001
        with self.assertRaises(ex.ExampleSelectionError) as caught:
            ex.select_examples(rows, gt, definition_path=DEFINITION, audit_gate=GATE)
        self.assertEqual(caught.exception.audit['eligible_count'], 4)
        self.assertEqual(caught.exception.audit['status'], 'BLOCKED')

    def test_improvement_must_be_positive(self):
        rows, gt = fixture(); original = {r['scene_id']: r for r in rows if r['condition'] == 'O'}
        for row in rows:
            if row['condition'] == 'P':row['metrics']['aligned']['dimension_mae'] = original[row['scene_id']]['metrics']['aligned']['dimension_mae']
        with self.assertRaises(ex.ExampleSelectionError):ex.select_examples(rows, gt, definition_path=DEFINITION, audit_gate=GATE)

    def test_duplicate_slot_rejected(self):
        rows, gt = fixture()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            ex.select_examples(rows + [copy.deepcopy(rows[0])], gt, definition_path=DEFINITION, audit_gate=GATE)

    def test_bad_gt_descriptor_excluded(self):
        rows, gt = fixture();gt['scene_a']['verified'] = False
        s = ex.select_examples(rows, gt, definition_path=DEFINITION, audit_gate=GATE)
        self.assertIn('GT_SOURCE_NOT_VERIFIED', s['excluded_candidates'][0]['reasons'])


class TestGeometryAndSources(unittest.TestCase):
    def test_rectangles_cover_exactly_exposed_faces(self):
        world = {(0,0,0): 'stone', (1,0,0): 'stone', (0,1,0): 'glass', (-2,1,2): 'oak_planks'}
        mesh = ex.exposed_rectangles(world)
        expected = set()
        for pos, material in world.items():
            for axis in range(3):
                uv = [i for i in range(3) if i != axis]
                for sign in (-1,1):
                    q=list(pos);q[axis]+=sign
                    if tuple(q) not in world:expected.add((axis,sign,pos[axis]+(sign==1),material,pos[uv[0]],pos[uv[1]]))
        actual = set()
        for row in mesh['rectangles']:
            for u in range(row['u'],row['u']+row['width']):
                for v in range(row['v'],row['v']+row['height']):
                    key=(row['axis_xyz'],row['normal_sign'],row['plane'],row['material'],u,v)
                    self.assertNotIn(key,actual);actual.add(key)
        self.assertEqual(expected,actual)
        self.assertEqual(mesh['exposed_unit_faces'],len(expected))

    def test_solid_cuboid_merges_to_six_faces(self):
        world={(x,y,z):'stone'for x in range(10)for y in range(8)for z in range(6)}
        mesh=ex.exposed_rectangles(world)
        self.assertEqual(mesh['rectangle_count'],6)
        self.assertEqual(mesh['exposed_unit_faces'],2*(10*8+8*6+10*6))

    def test_material_boundaries_and_sparse_far_coordinates(self):
        world={(0,0,0):'stone',(1,0,0):'glass',(1000000,-500000,10):'stone'}
        mesh=ex.exposed_rectangles(world)
        self.assertEqual(mesh['exposed_unit_faces'],16)
        self.assertEqual(ex.grid_description(world)['dimensions_xyz'],[1000001,500001,11])
        self.assertEqual({r['material']for r in mesh['rectangles']},{'stone','glass'})

    def test_resource_limit_never_downsamples(self):
        with self.assertRaisesRegex(ValueError, 'RESOURCE_LIMIT'):
            ex.exposed_rectangles({(0,0,0):'stone'},max_exposed_faces=5)

    def test_description_excerpt_literal_unicode_and_hash(self):
        with tempfile.TemporaryDirectory()as t:
            p=Path(t)/'description.json';p.write_text(json.dumps({'summary':'石造の家。'*60},ensure_ascii=False));sha=hashlib.sha256(p.read_bytes()).hexdigest()
            d=ex.description_excerpt(p,sha,audit_gate=GATE)
            self.assertEqual(d['excerpt'],('石造の家。'*60)[:240]);self.assertTrue(d['truncated'])
            with self.assertRaisesRegex(ValueError,'hash mismatch'):ex.description_excerpt(p,'0'*64,audit_gate=GATE)

    def test_canonical_local_source_to_world_and_hash(self):
        from offline_evaluation.server_v2.core import Grid
        with tempfile.TemporaryDirectory()as t:
            p=Path(t)/'voxels.json';p.write_text(json.dumps(Grid({(0,1,2):'stone'},'local',(7,-2,1)).to_dict()))
            src={'kind':'server_v2_grid_json','path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            self.assertEqual(ex.load_source_world(src,audit_gate=GATE),{(7,-1,3):'stone'})
            p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'hash changed'):ex.load_source_world(src,audit_gate=GATE)


class TestRender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='goal008-examples-fixture-');cls.path=Path(cls.tmp.name)/'render'
        cls.selection=selection();cls.examples=packages(cls.selection);before=copy.deepcopy(cls.examples)
        with patch('socket.create_connection',side_effect=AssertionError('network forbidden')):
            cls.manifest=ex.render_examples(cls.examples,cls.selection,cls.path,definition_path=DEFINITION,audit_gate=GATE)
        assert cls.examples==before

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_six_real_plot_exports_and_hashes(self):
        from PIL import Image
        self.assertEqual(len(self.manifest['files']),11)
        for row in self.manifest['files']:
            p=self.path/row['path'];self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),row['sha256']);self.assertLessEqual(p.stat().st_size,ex.MAX_BYTES)
        for n in range(1,4):
            with Image.open(self.path/f'example_{n:02}.png')as im:self.assertGreater(im.width,5000);self.assertAlmostEqual(im.info['dpi'][0],300,places=1)
            path=self.path/f'example_{n:02}.svg';self.assertTrue(ET.parse(path).getroot().tag.endswith('svg'));self.assertIn('SYNTHETIC FIXTURE',path.read_text())

    def test_common_world_bounds_and_failure_not_fabricated(self):
        details=json.loads((self.path/'example_sources.json').read_text())
        for d in details:
            self.assertEqual(d['bounds_world_xyz'],[[-3,3],[-1,5],[-1,4]])
            self.assertFalse(d['camera']['alignment_shift_applied'])
        failed=details[2]['panels']['E'];self.assertEqual(failed['failure_kind'],'truncation');self.assertIsNone(failed['geometry']);self.assertIsNone(failed['metrics'])
        with (self.path/'example_companion.csv').open()as f:self.assertEqual(len(list(csv.DictReader(f))),15)

    def test_selected_slot_mismatch_fails_closed(self):
        data=copy.deepcopy(self.examples);data[0]['panels']['O']['attempt_id']='other'
        with self.assertRaisesRegex(ValueError,'selected final slot'):
            ex.render_examples(data,self.selection,self.path/'new',definition_path=DEFINITION,audit_gate=GATE)

    def test_wrong_attempt_artifact_rejected_before_voxel_loading(self):
        # Synthetic descriptors exercise the production source-ownership guard;
        # no actual source path is opened and no production figure is rendered.
        chosen=copy.deepcopy(self.selection);chosen['input_kind']='verified_sources'
        data=copy.deepcopy(self.examples)
        selected=chosen['selected'][0]['slots']['O']
        selected['source_run']='/synthetic/enabled'
        selected['artifacts']={'voxels.json':{'path':'attempts/expected/voxels.json','sha256':'b'*64}}
        data[0]['panels']={'O':data[0]['panels']['O'],**{k:v for k,v in data[0]['panels'].items()if k!='O'}}
        data[0]['panels']['O']['source']={'kind':'server_v2_grid_json','path':'/synthetic/enabled/attempts/wrong/voxels.json','sha256':'b'*64}
        gate={'status':'PASS','input_kind':'verified_sources','phase0_status':'PASS','phase1_status':'PASS'}
        with patch.object(ex,'load_source_world',side_effect=AssertionError('unexpected source read')):
            with self.assertRaisesRegex(ValueError,'does not belong to selected'):
                ex.render_examples(data,chosen,self.path/'new',definition_path=DEFINITION,audit_gate=gate)

    def test_single_body_figure_groups_three_rows(self):
        tex=(self.path/'examples_figure.tex').read_text()
        self.assertEqual(tex.count(r'\begin{figure*}'),1)
        self.assertEqual(tex.count(r'\includegraphics'),3)
        self.assertIn('do not estimate population-average performance',tex)

    def test_no_failed_geometry_or_overwrite(self):
        data=copy.deepcopy(self.examples);data[2]['panels']['E']['world_voxels']=[[0,0,0,'stone']]
        with self.assertRaisesRegex(ValueError,'cannot fabricate'):
            ex.render_examples(data,self.selection,self.path/'new',definition_path=DEFINITION,audit_gate=GATE)
        with self.assertRaises(FileExistsError):ex.render_examples(self.examples,self.selection,self.path,definition_path=DEFINITION,audit_gate=GATE)


if __name__=='__main__':unittest.main()
