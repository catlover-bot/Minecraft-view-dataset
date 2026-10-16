"""Synthetic source, commit, and Phase1 guards; never select research examples."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

from offline_evaluation.analysis.goal008 import examples_cli as cli


def source_fixture():
    slots, descriptions = [], []
    for n in range(60):
        scene = f'fixture_{n:03d}'
        observation = {'attempt_id': scene + 'O', 'global_schedule_index': n * 4,
                       'status': 'success', 'source_plan_identity': 'a' * 64}
        common = {'scene_id': scene, 'gt_descriptor': {'voxels_sha256': 'b' * 64,
            'bbox_sha256': 'c' * 64, 'grid_sha256': 'd' * 64},
            'gt_descriptor_sha256': 'e' * 64, 'gt_bbox_path': '/fixture/bbox.json',
            'gt_voxels_path': '/fixture/voxels.npy', 'O_J_typed_tree_equal': True,
            'source_sha256': 'f' * 64, 'input_sha256': 'f' * 64, 'input_id': scene + 'Oinput',
            'j_input_sha256': '1' * 64, 'j_input_id': scene + 'Jinput',
            'builder_observations': {'openai': observation}}
        for provider in ('openai', 'anthropic', 'gemini'):
            descriptions.append(dict(copy.deepcopy(common), description_provider=provider))
        for c in 'OJPE':
            row = {'scene_id': scene, 'provider': 'openai', 'condition': c,
                   'description_model': 'openai' if c in 'OJ' else None,
                   'source_gt_descriptor_hash': 'e' * 64, 'gt_grid_sha256': 'd' * 64,
                   'input_id': scene + c + 'input', 'input_sha256': 'f' * 64 if c == 'O' else '1' * 64}
            if c == 'O': row.update(observation)
            slots.append(row)
    return slots, descriptions


class TestVerifiedExampleGuards(unittest.TestCase):
    def test_pending_phase1_stops_before_any_source_or_commit_access(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'PHASE1_MANIFEST.json'
            path.write_text(json.dumps({'status': 'PENDING', 'version': 'goal008_phase1_v1'}))
            with patch.object(cli.audit, 'verify_report', side_effect=AssertionError('source touched')), \
                 patch.object(cli, 'verify_commit_files', side_effect=AssertionError('commit touched')):
                with self.assertRaisesRegex(ValueError, 'Phase0 and Phase1 PASS'):
                    cli.verify_phase1('/nonexistent/source_gate.json', path)

    def test_phase1_gate_mismatch_stops_before_source_access(self):
        with tempfile.TemporaryDirectory() as temp:
            gate = Path(temp) / 'gate.json'; gate.write_text('{}')
            path = Path(temp) / 'manifest.json'
            path.write_text(json.dumps({'status': 'PHASE0_PHASE1_PASS', 'version': 'goal008_phase1_v1',
                'source_gate_path': str(gate), 'source_gate_sha256': '0' * 64}))
            with patch.object(cli.audit, 'verify_report', side_effect=AssertionError('source touched')):
                with self.assertRaisesRegex(ValueError, 'gate binding mismatch'):
                    cli.verify_phase1(gate, path)

    def test_phase1_tampered_product_stops_before_source_access(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); gate = directory / 'gate.json'; gate.write_text('{}')
            names = ['definitions_frozen.json', 'proxy_scores_180.json', 'proxy_scores_180.csv',
                'exploratory_associations.json', 'exploratory_associations_360.csv', 'proxy_descriptive.csv',
                'joined_descriptions_180.json', 'source_map_180.csv']
            for name in names: (directory / name).write_text('frozen')
            manifest = {'status': 'PHASE0_PHASE1_PASS', 'version': 'goal008_phase1_v1',
                'source_gate_path': str(gate), 'source_gate_sha256': hashlib.sha256(gate.read_bytes()).hexdigest(),
                'matched_descriptions': 180, 'source_bound_builder_observations': 360, 'independent_buildings': 60,
                'planned_associations': 360, 'source_inventories_before_after_unchanged': True,
                'historical_RQ2_status': 'NOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES',
                **{field: 0 for field in ['research_api_requests_sent', 'api_key_queries', 'model_queries', 'human_experiments', 'checkpoint_mutations']},
                'products': {name: {'bytes': 6, 'sha256': hashlib.sha256(b'frozen').hexdigest()} for name in names}}
            (directory / names[3]).write_text('edited')
            path = directory / 'manifest.json'; path.write_text(json.dumps(manifest))
            with patch.object(cli.audit, 'verify_report', side_effect=AssertionError('source touched')), \
                 patch.object(cli, 'verify_commit_files', side_effect=AssertionError('commit touched')):
                with self.assertRaisesRegex(ValueError, 'product hash mismatch'):
                    cli.verify_phase1(gate, path)

    def test_bad_commit_never_reads_source_or_selects(self):
        with patch.object(cli, 'verify_own_commit', side_effect=ValueError('frozen commit mismatch')), \
             patch.object(cli, 'verify_phase1', side_effect=AssertionError('source touched')), \
             patch.object(cli.examples, 'select_examples', side_effect=AssertionError('scores selected')), \
             patch.object(cli.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 20 * 1024**3})()):
            with self.assertRaisesRegex(ValueError, 'frozen commit mismatch'):
                cli.run_from_verified('/fixture/gate', '/fixture/phase1', '/root/minecraft-runs/fixture-never-created-008', 'a' * 40)

    def test_nonpassing_gate_never_selects(self):
        with patch.object(cli, 'verify_own_commit', return_value={}), \
             patch.object(cli, 'verify_phase1', side_effect=ValueError('Phase0 and Phase1 PASS required')), \
             patch.object(cli.examples, 'select_examples', side_effect=AssertionError('scores selected')), \
             patch.object(cli.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 20 * 1024**3})()):
            with self.assertRaisesRegex(ValueError, 'Phase0 and Phase1 PASS'):
                cli.run_from_verified('/fixture/gate', '/fixture/phase1', '/root/minecraft-runs/fixture-never-created-008', 'a' * 40)

    def test_commit_check_compares_bytes_not_worktree_head(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'file.py').write_text('frozen\n')
            digest = hashlib.sha256(b'frozen\n').hexdigest()
            with patch.object(cli, 'ROOT', root), patch.object(cli.subprocess, 'run') as run:
                run.return_value = type('Result', (), {'returncode': 0, 'stdout': b'changed\n'})()
                with self.assertRaisesRegex(ValueError, 'frozen commit mismatch'):
                    cli.verify_commit_files('a' * 40, {'file.py': digest})
                self.assertEqual(run.call_args.args[0], ['git', 'show', 'a' * 40 + ':file.py'])

    def test_source_binding_complete_and_input_unchanged(self):
        slots, sources = source_fixture(); before = copy.deepcopy((slots, sources))
        gt, descriptions = cli.bind_example_sources(slots, sources)
        self.assertEqual(len(gt), 60); self.assertEqual(len(descriptions), 60)
        self.assertEqual(gt['fixture_000']['kind'], 'gt_numpy')
        self.assertEqual((slots, sources), before)

    def test_duplicate_description_fails_closed(self):
        slots, sources = source_fixture(); sources[-1] = copy.deepcopy(sources[0])
        with self.assertRaisesRegex(ValueError, '180-description'):
            cli.bind_example_sources(slots, sources)

    def test_gt_disagreement_fails_closed(self):
        slots, sources = source_fixture(); sources[1]['gt_descriptor']['bbox_sha256'] = '2' * 64
        with self.assertRaisesRegex(ValueError, 'GT source disagreement'):
            cli.bind_example_sources(slots, sources)

    def test_wrong_selected_attempt_fails_closed(self):
        slots, sources = source_fixture(); slots[0]['attempt_id'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'selected-attempt binding mismatch'):
            cli.bind_example_sources(slots, sources)

    def test_lossless_j_input_mismatch_fails_closed(self):
        slots, sources = source_fixture(); slots[1]['input_sha256'] = '9' * 64
        with self.assertRaisesRegex(ValueError, 'O/J input binding mismatch'):
            cli.bind_example_sources(slots, sources)

    def test_build_packages_retains_failed_e_without_voxels(self):
        from .test_examples import selection
        from offline_evaluation.server_v2.core import Grid
        selected = selection()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            bbox = directory / 'bbox.json'; bbox.write_text('{}')
            desc = directory / 'description.json'; desc.write_text('{"summary":"Original fixture."}')
            voxel = directory / 'voxels.json'; voxel.write_text(json.dumps(Grid({(0, 0, 0): 'stone'}, 'world').to_dict()))
            sources = {}
            for chosen in selected['selected']:
                chosen['gt_source']['bbox_path'] = str(bbox)
                sources[chosen['scene_id']] = {'source_path': str(desc), 'source_sha256': hashlib.sha256(desc.read_bytes()).hexdigest()}
                for row in chosen['slots'].values():
                    row['source_run'] = str(directory)
                    row['artifacts'] = {'voxels.json': {'path': voxel.name, 'sha256': hashlib.sha256(voxel.read_bytes()).hexdigest(), 'bytes': voxel.stat().st_size}} if row['status'] == 'success' else {}
            with patch.object(cli.examples, 'load_source_world', return_value={(0, 0, 0): 'stone'}):
                packages = cli.build_packages(selected, sources, {'status': 'PASS', 'input_kind': 'fixture'})
            panel = packages[2]['panels']['E']
            self.assertEqual(panel['failure_kind'], 'truncation')
            self.assertIsNone(panel['world_voxels']); self.assertIsNone(panel['metrics'])
            self.assertEqual(panel['source']['attempt_id'], selected['selected'][2]['slots']['E']['attempt_id'])
            self.assertEqual(panel['source']['artifacts'], {})
            self.assertEqual(packages[0]['description']['excerpt'], 'Original fixture.')

    def test_cli_stop_is_structured_and_never_claims_success(self):
        stream = io.StringIO()
        with patch.object(cli, 'run_from_verified', side_effect=ValueError('Phase1 mismatch')), redirect_stdout(stream):
            status = cli.main(['--phase0-gate', '/fixture/gate', '--phase1-manifest', '/fixture/manifest',
                               '--destination', '/fixture/out', '--code-commit', 'a' * 40])
        self.assertEqual(status, 5)
        self.assertEqual(json.loads(stream.getvalue()), {'status': 'STOP', 'error_type': 'ValueError',
                         'safe_reason': 'Phase1 mismatch', 'research_api_requests_sent': 0})


if __name__ == '__main__':
    unittest.main()
