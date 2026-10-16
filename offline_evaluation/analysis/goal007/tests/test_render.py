"""Synthetic publication tests. No real ledgers, statistics, keys or APIs are read."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from offline_evaluation.analysis.goal007 import render
from offline_evaluation.server_v2.analysis_v4 import comparison_families


def fixture():
    groups, per_group = [], []
    for provider in render.BUILDERS:
        for condition in render.CONDITIONS:
            planned = 180 if condition in 'OJ' else 60
            for level, model in [('builder_condition', None)] + ([('description_builder_condition', d) for d in ('openai', 'anthropic', 'gemini')] if condition in 'OJ' else []):
                count = planned if level == 'builder_condition' else 60
                missing = int(provider == 'openai' and condition == 'J' and model in (None, 'openai'))
                failed = int(provider == 'anthropic' and condition == 'O' and model in (None, 'openai'))
                d = {'group_level': level, 'provider': provider, 'condition': condition, 'description_model': model,
                     'planned_slots': count, 'independent_buildings': 60, 'completed_request_outcome_slots': count - missing,
                     'valid_only_aligned_iou_slots': count - missing - failed, 'model_output_failure_slots': failed,
                     'operational_unmeasurable_slots': missing, 'unknown_usage_slots': missing,
                     'bounded_iou_eligible_slots': count - missing}
                groups.append(d)
                for metric in render.METRICS:
                    bounded = metric == render.METRICS[0]
                    per_group.append({**d, 'metric': metric, 'valid_rows': count - missing - failed,
                                      'valid_buildings': 60, 'equal_building_mean': .321234 if bounded else 4.56789,
                                      'failure0_eligible_rows': count - missing if bounded else None,
                                      'failure0_buildings': 60 if bounded else None,
                                      'failure0_equal_building_mean': .318123 if bounded else None})
    comparisons = []
    for family in comparison_families():
        for comparison in family['comparisons']:
            bounded = comparison['metric'] == render.METRICS[0]
            comparisons.append({**comparison, 'family_id': family['id'], 'family_size': family['size'],
                'buildings': 59, 'common_valid_pair_count': 177 if comparison['a_condition'] in 'OJ' and comparison['b_condition'] in 'OJ' else 59,
                'mean_delta': -.125, 'ci95': [-.2, -.05], 'p_two_sided': .0000999900009999,
                'p_holm': .0007999200079992, 'common_failure0_pair_count': 178 if bounded else None,
                'common_failure0_buildings': 60 if bounded else None,
                **({'all_attempted_failure0_companion': {'mean_delta': -.12, 'ci95': [-.21, -.03]}} if bounded else {})})
    result = {'analysis_label': render.ANALYSIS_LABEL, 'input_kind': 'fixture', 'research_api_requests_sent': 0,
              'inference': {'seed': 20261013, 'bootstrap_replicates': 2000, 'signflip_replicates': 10000},
              'per_group': per_group, 'comparisons': comparisons,
              'denominators': {'overall': {'planned_slots': 960, 'independent_buildings': 60,
                  'completed_request_outcome_slots': 959, 'model_output_failure_slots': 1,
                  'operational_unmeasurable_slots': 1, 'unknown_usage_slots': 1,
                  'valid_only_aligned_iou_slots': 958, 'bounded_iou_eligible_slots': 959},
                  'groups': groups, 'comparisons': [{'id': r['id'], 'buildings': r['buildings']} for r in comparisons]},
              'sensitivity_missing942': {'exploratory': True, 'missing_slot': {'global_schedule_index': 942, 'metrics': None},
                                         'affected_comparisons': [], 'inference': 'No p-values.'}}
    context = {'status': 'PASS', 'input_kind': 'fixture', 'fixed_scope_attempts': 961, 'historical_calls': 991,
               'operational_error_attempts': 2, 'unknown_usage_holds': [{'http': 401, 'usd': '.165536'}, {'http': 520, 'usd': '.165536'}],
               'cost_rows': [{'source': 'fixture', 'known_usd': '1.23', 'invoice': 'UNKNOWN'}],
               'failure_taxonomy': [{'layer': 'model_output', 'kind': 'truncation', 'count': 1}, {'layer': 'operational', 'kind': 'HTTP520', 'count': 1}],
               'error_rows': [{'layer': 'model_output', 'count': 1}, {'layer': 'operational_attempt', 'count': 2}],
               'reproduction_command': 'python fixture_only.py --output /tmp/fixture_new', 'source_proofs': [{'source': 'fixture'}]}
    rq2 = {'status': render.RQ2_UNAVAILABLE, 'source_evidence': [{'path': 'fixture_scores.json', 'sha256': 'a' * 64}]}
    return result, context, rq2


class TestRender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='goal007-render-fixture-')
        cls.root = Path(cls.tmp.name)
        cls.results, cls.context, cls.rq2 = fixture()
        before = deepcopy(cls.results)
        with patch('socket.create_connection', side_effect=AssertionError('network forbidden')), patch('offline_evaluation.server_v2.analysis_v4.paired_building_inference', side_effect=AssertionError('inference forbidden')):
            cls.manifest = render.render_results(cls.results, cls.root, audit_context=cls.context, rq2_audit=cls.rq2)
        assert before == cls.results

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_exact_stored_numbers_and_nulls(self):
        self.assertEqual(json.loads((self.root / 'results.json').read_text()), self.results)
        self.assertEqual(json.loads((self.root / 'sensitivity_missing942.json').read_text())['missing_slot']['metrics'], None)
        with (self.root / 'per_group.csv').open() as f:
            rows = list(csv.DictReader(f))
        mae = next(r for r in rows if r['metric'] == render.METRICS[1])
        self.assertEqual(mae['failure0_equal_building_mean'], '')
        self.assertEqual(mae['equal_building_mean'], '4.56789')
        self.assertEqual(len([r for r in rows if r['group_level'] == 'description_builder_condition']), 24)

    def test_all_contrasts_and_full_precision_p_are_preserved(self):
        with (self.root / 'comparisons_20.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 20)
        self.assertEqual(float(rows[0]['p_two_sided']), .0000999900009999)
        latex = (self.root / 'table_statistics.tex').read_text()
        self.assertEqual(latex.count('-0.1250'), 20)
        self.assertIn('9.999e-05', latex)
        self.assertIn('178/60', latex)

    def test_all_artifacts_hash_and_images(self):
        from PIL import Image
        self.assertEqual({f['path'] for f in self.manifest['files']}, {p.name for p in self.root.iterdir() if p.is_file()} - {'render_artifacts.json'})
        for item in self.manifest['files']:
            self.assertEqual(hashlib.sha256((self.root / item['path']).read_bytes()).hexdigest(), item['sha256'])
        for figure in render.FIGURES:
            with Image.open(self.root / (figure + '.png')) as im:
                self.assertGreater(im.width, 2000)
                self.assertGreater(im.height, 1000)
                self.assertAlmostEqual(im.info['dpi'][0], 300, places=1)
            svg = (self.root / (figure + '.svg')).read_text()
            self.assertIn('<svg', svg)
            self.assertIn('<text', svg)
        sources = json.loads((self.root / 'figure_sources.json').read_text())
        self.assertEqual(len(sources[2]['statistics']), 20)

    def test_prose_explicit_missingness_and_rq2(self):
        text = (self.root / 'ARR_RESULTS_EN.md').read_text()
        self.assertIn('SYNTHETIC FIXTURE', text)
        self.assertIn('961 fixed-scope attempts and 991 historical calls', text)
        self.assertIn('HTTP520', text)
        self.assertIn('NOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES', text)
        self.assertIn('lossless JSON reserialization', text)
        self.assertIn('positive MAE differences indicate greater error', text)
        self.assertIn('null', text)
        self.assertNotIn('COMPLETE_FIXED960', text)
        self.assertNotIn('FINAL_COMPLETE_960_MEASURED', text)
        self.assertIn('4.5679', (self.root / 'SUMMARY_JA.md').read_text())

    def test_output_collision_never_overwrites(self):
        digest = hashlib.sha256((self.root / 'results.json').read_bytes()).hexdigest()
        with self.assertRaises(FileExistsError):
            render.render_results(self.results, self.root, audit_context=self.context, rq2_audit=self.rq2)
        self.assertEqual(digest, hashlib.sha256((self.root / 'results.json').read_bytes()).hexdigest())

    def test_latex_escapes_dynamic_text(self):
        self.assertEqual(render._tex('a_b&c%#'), r'a\_b\&c\%\#')

    def test_fails_closed_without_gate(self):
        with self.assertRaisesRegex(ValueError, 'passing audit'):
            render._validate(self.results, {**self.context, 'status': 'PENDING'}, self.rq2)

    def test_family_and_independent_n_guards(self):
        bad = deepcopy(self.results); bad['comparisons'][0]['family_id'] = 'posthoc'
        with self.assertRaisesRegex(ValueError, '8/4/8'):
            render._validate(bad, self.context, self.rq2)
        bad = deepcopy(self.results); bad['comparisons'][0]['buildings'] = 180
        with self.assertRaisesRegex(ValueError, 'building count'):
            render._validate(bad, self.context, self.rq2)

    def test_no_mae_zero_or_nonfinite(self):
        bad = deepcopy(self.results); bad['comparisons'][1]['common_failure0_pair_count'] = 12
        with self.assertRaisesRegex(ValueError, 'MAE'):
            render._validate(bad, self.context, self.rq2)
        bad = deepcopy(self.results); bad['comparisons'][0]['mean_delta'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            render._validate(bad, self.context, self.rq2)

    def test_real_scope_requires_both_holds_and_history(self):
        real = {**self.results, 'input_kind': 'verified_sources'}
        context = {**self.context, 'input_kind': 'verified_sources', 'unknown_usage_holds': []}
        with self.assertRaisesRegex(ValueError, 'both unknown'):
            render._validate(real, context, self.rq2)
        context['unknown_usage_holds'] = self.context['unknown_usage_holds']; context['fixed_scope_attempts'] = 960
        with self.assertRaisesRegex(ValueError, 'denominators'):
            render._validate(real, context, self.rq2)

    def test_bundle_cli_contract(self):
        bundle = self.root / 'cli-input.json'
        bundle.write_text(json.dumps({'results': self.results, 'audit_context': self.context, 'rq2_audit': self.rq2}))
        try:
            with patch.object(render, 'render_results', return_value={'files': []}) as fn:
                render.main(['--input', str(bundle), '--destination', str(self.root / 'new')])
            self.assertEqual(fn.call_args.args, (self.results, self.root / 'new'))
            self.assertEqual(fn.call_args.kwargs['audit_context'], self.context)
        finally:
            bundle.unlink()


if __name__ == '__main__':
    unittest.main()
