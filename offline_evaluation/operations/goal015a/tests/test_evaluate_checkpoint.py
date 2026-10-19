"""Synthetic parent72/child8 reuse and save-only round trips; no research API."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from offline_evaluation.operations.goal015a import evaluate as e, checkpoint as c, runtime as r
from offline_evaluation.operations.goal015a.tests.test_runtime import fixture_plan as minimal_fixture, execute, MockAdapter
from offline_evaluation.experiments.goal015.tests.test_checkpoint import FakeGithub as OriginalFakeGithub


def fixture_plan(tmp_path, monkeypatch):
    """Upgrade the synthetic runtime fixture to authentic frozen CPU products."""
    child_path = minimal_fixture(tmp_path)
    child = r.read_json(child_path)
    parent_path = Path(child['parent']['plan']['path'])
    parent = r.read_json(parent_path)
    truth = e.frozen.core.Grid({(0, 0, 0): 'stone'})
    parent['gt_sources'] = {}
    for scene in sorted({row['scene_id'] for row in parent['schedule']}):
        building = parent_path.parent / 'fake-gt' / scene
        (building / 'gt').mkdir(parents=True)
        voxels, bbox = building / 'gt/voxels.npy', building / 'gt/bbox.json'
        voxels.write_bytes(b'synthetic fixture only'); bbox.write_text('{}')
        parent['gt_sources'][scene] = {'building_directory': str(building),
            'voxels_sha256': r.descriptor(voxels)['sha256'], 'bbox_sha256': r.descriptor(bbox)['sha256'],
            'grid_sha256': e.frozen.frozen_grid_sha256(truth)}
    parent['plan_id'] = r.identity(parent)
    r._save(parent_path, parent)
    approval_path, authority_path, ledger_path = [Path(child['parent'][key]['path']) for key in ('approval', 'authority', 'ledger')]
    approval = r.read_json(approval_path); approval['plan_id'] = parent['plan_id']; r._save(approval_path, approval)
    authority = r.read_json(authority_path)
    authority.update(plan_id=parent['plan_id'], plan=r.descriptor(parent_path), approval=r.descriptor(approval_path))
    r._save(authority_path, authority)
    ledger = r.read_json(ledger_path); ledger['plan_id'] = parent['plan_id']
    ledger['calls'][0].update(completed_response=False, failure_kind='truncation', model_output_failure=True)
    r._save(ledger_path, ledger)
    checkpoint_path = Path(child['parent']['checkpoint']['path'])
    checkpoint = r.read_json(checkpoint_path); checkpoint['plan_id'] = parent['plan_id']; r._save(checkpoint_path, checkpoint)
    binding = {'plan_id': parent['plan_id'], 'plan': r.descriptor(parent_path), 'ledger': r.descriptor(ledger_path)}
    summary = e.frozen.evaluate_saved(parent['schedule'], ledger['calls'], parent_path.parent / 'evaluations/parent72',
        lambda _: truth, fixture=True, plan_binding=binding, gt_sources=parent['gt_sources'])
    child['parent_plan_id'] = parent['plan_id']
    for key in ('plan', 'approval', 'authority', 'ledger', 'checkpoint'):
        child['parent'][key] = r.descriptor(child['parent'][key]['path'])
    child['parent']['evaluation'] = summary['evaluation_artifact']
    child['plan_id'] = r.identity(child); r._save(child_path, child)
    monkeypatch.setattr(e, 'load_scene', lambda path: SimpleNamespace(scene_id=Path(path).name, grid=truth))
    r.verify_plan(child_path)
    return child_path


def _parent_bytes(plan):
    paths = {ref['path'] for ref in plan['parent'].values()}
    ledger = r.bound(plan['parent']['ledger'])
    paths.update(row[key]['path'] for row in ledger['calls'] for key in ('raw', 'response_text_path') if row.get(key))
    return {path: Path(path).read_bytes() for path in paths}


def test_merge_reuses72_exactly_and_only_evaluates8(tmp_path, monkeypatch):
    path = fixture_plan(tmp_path, monkeypatch); plan = r.verify_plan(path)
    original = _parent_bytes(plan); adapter = MockAdapter(); assert execute(path, adapter)['calls_settled'] == 8
    observed = []; evaluate_one = e.frozen.evaluate_one
    def tracked(record, *args, **kwargs):
        observed.append(record['call_id']); return evaluate_one(record, *args, **kwargs)
    monkeypatch.setattr(e.frozen, 'evaluate_one', tracked)
    summary = e.evaluate_plan(path)
    assert observed == r.CALL_IDS and adapter.calls == 8
    assert summary['parent_rows_reused'] == 72 and summary['actual_metric_rows'] == 79
    merged = r.bound(summary['evaluation_artifact']); parent = r.bound(plan['parent']['evaluation'])
    assert len(merged['rows']) == 80
    for new, old in zip(merged['rows'][:72], parent['rows'][:72]):
        assert {key: value for key, value in new.items() if key != 'selected_source'} == old
        assert new['selected_source']['selected_source'] == 'PARENT'
    assert all(row['selected_source']['selected_source'] == 'CHILD' for row in merged['rows'][72:])
    assert summary['plan_binding']['plan'] == plan['parent']['plan']
    assert summary['plan_binding']['plan_id'] == plan['parent_plan_id']
    assert e.evaluate_plan(path) == summary and observed == r.CALL_IDS
    assert all(Path(name).read_bytes() == data for name, data in original.items())


def test_unknown_and_model_failure_preserve_separate_utility_semantics(tmp_path, monkeypatch):
    path = fixture_plan(tmp_path, monkeypatch)
    class Adapter(MockAdapter):
        def parse_response(self, *args):
            response = super().parse_response(*args)
            if self.calls == 1: response.text = 'not JSON'
            return response
    adapter = Adapter(unknown_at=2)
    assert execute(path, adapter)['calls_uncertain'] == 1
    assert execute(path, adapter, resume=True)['calls_settled'] == 7 and adapter.calls == 8
    summary = e.evaluate_plan(path); rows = r.bound(summary['evaluation_artifact'])['rows']
    assert rows[72]['status'] == 'MODEL_OUTPUT_FAILURE' and rows[72]['failure_kind'] == 'parse_failure'
    assert rows[73]['status'] == 'OPERATIONAL_UNKNOWN' and rows[73]['metrics'] is None
    assert e.frozen.failure0_eligible(rows[72]) and not e.frozen.failure0_eligible(rows[73])
    assert summary['actual_metric_rows'] == 77 and summary['planned_independent_buildings'] == 10
    for result in summary['primary_paired_estimates']:
        if result['builder'] == 'anthropic':
            assert result['eligible_paired_blocks'] == 19
            assert result['scenes_with_one_repeat'] == 1


def test_disabled_child_evaluation_never_creates_authority_or_zeroes_missing(tmp_path, monkeypatch):
    path = fixture_plan(tmp_path, monkeypatch)
    summary = e.evaluate_plan(path); rows = r.bound(summary['evaluation_artifact'])['rows']
    assert len(rows) == 80 and all(row['state'] == 'NOT_SENT' and row['metrics'] is None for row in rows[72:])
    assert not (path.parent / 'enabled').exists() and not (path.parent / 'execution.lock').exists()
    assert summary['actual_metric_rows'] == 71


def test_changed_reused_metric_artifact_fails_before_reuse(tmp_path, monkeypatch):
    path = fixture_plan(tmp_path, monkeypatch); execute(path)
    summary = e.evaluate_plan(path); data = r.bound(summary['evaluation_artifact'])
    metric = Path(data['rows'][1]['artifacts']['metrics.json']['path']); metric.write_text('{}')
    with pytest.raises((ValueError, r.RuntimeErrorSafe), match='hash|artifact'):
        e.find_existing_evaluation(path)


def test_cached_parent_status_cannot_change_with_rehashed_evaluation(tmp_path, monkeypatch):
    path = fixture_plan(tmp_path, monkeypatch); execute(path)
    summary = e.evaluate_plan(path)
    evaluation_path = Path(summary['evaluation_artifact']['path'])
    document = r.read_json(evaluation_path)
    document['rows'][0]['status'] = 'PROTOCOL_VALID'
    document['rows'][0]['protocol_valid'] = True
    r._save(evaluation_path, document)
    summary['evaluation_artifact'] = r.descriptor(evaluation_path)
    r._save(evaluation_path.parent / 'summary.json', summary)
    with pytest.raises(r.RuntimeErrorSafe, match='reused parent evaluation row changed'):
        e.find_existing_evaluation(path)


class FakeGithub(OriginalFakeGithub):
    def __init__(self, commit):
        super().__init__(); self.commit = commit
    def call(self, stage, args, output=None):
        if stage == 'commit_ancestry':
            self.calls.append((stage, args, output))
            return {'status': 'identical', 'behind_by': 0, 'base_commit': {'sha': self.commit}}
        result = super().call(stage, args, output)
        if stage == 'asset_upload':
            self.assets[123][0]['name'] = 'goal015a-state.tar.gz'
            result['name'] = 'goal015a-state.tar.gz'
        return result


def test_child_merged_archive_and_fake_upload_retry_never_send(tmp_path, monkeypatch):
    path = fixture_plan(tmp_path, monkeypatch); plan = r.verify_plan(path)
    before = _parent_bytes(plan); adapter = MockAdapter()
    # Exercise the real default checkpoint callback under the runtime's locks.
    completed = r.execute(path, pricing_evidence=plan['pricing_evidence'], adapter=adapter,
                          confirmation=lambda _: True)
    assert completed['status'] == 'COMPLETE' and completed['calls_settled'] == 8
    assert completed['checkpoint']['status'] == 'LOCAL_VERIFIED'
    assert not r.bound(completed['checkpoint']['checkpoint'])['restoration']['merged_evaluation_included']
    summary = e.evaluate_plan(path)
    merged = r.bound(summary['evaluation_artifact'])
    assert len(merged['rows']) == summary['planned_calls'] == 80
    assert summary['planned_paired_blocks'] == 40 and summary['planned_independent_buildings'] == 10
    assert sum(row['empty_output'] is True and row['protocol_valid'] for row in merged['rows']) == 79
    assert sum(e.frozen.failure0_eligible(row) for row in merged['rows']) == 1
    assert all(row['metrics']['values']['aligned_occupancy_precision'] is None
               for row in merged['rows'] if row['metrics'] is not None)
    assert all(result['eligible_paired_blocks'] == 20 for result in summary['primary_paired_estimates']
               if result['estimand'] == 'completed_attempt_utility')
    saved = c.create_save_only(path); package = r.bound(saved['checkpoint'])
    assert package['schema'] == 'goal015a_checkpoint_v1' and package['restoration']['merged_evaluation_included']
    restored = Path(saved['path']) / 'verified-restoration'
    parent_root = Path(r.bound(plan['parent']['plan'])['canonical_root'])
    evaluation_path = Path(summary['evaluation_artifact']['path'])
    assert (restored / 'parent' / evaluation_path.relative_to(parent_root)).read_bytes() == evaluation_path.read_bytes()
    assert (restored / 'child/enabled/responses/goal015_call_073.raw.json').is_file()
    for row in merged['rows']:
        for key, artifact in row['artifacts'].items():
            if key == 'request_payload':
                continue
            source_path = Path(artifact['path'])
            child_source = source_path.is_relative_to(path.parent)
            archived = restored / ('child' if child_source else 'parent') / source_path.relative_to(
                path.parent if child_source else parent_root)
            assert archived.read_bytes() == source_path.read_bytes()
    assert not (restored / 'parent/payload.json').exists()
    github = FakeGithub(package['target_commitish']); github.fail_download = True
    with pytest.raises(RuntimeError, match='fixture interruption'):
        c.upload_checkpoint(saved['path'], tmp_path / 'failed-upload', allow_create=True, github=github)
    github.fail_download = False
    result = c.upload_checkpoint(saved['path'], tmp_path / 'saved-retry', github=github)
    assert result['status'] == 'GITHUB_VERIFIED' and result['draft'] and not result['uploaded']
    assert result['restoration']['merged_evaluation_included'] and adapter.calls == 8
    assert sum(stage == 'release_create' for stage, _, _ in github.calls) == 1
    assert sum(stage == 'asset_upload' for stage, _, _ in github.calls) == 1
    assert all(Path(name).read_bytes() == data for name, data in before.items())
    assert len(github.releases) == 1 and github.releases[0]['body'].startswith('Goal015A immutable')
    r._save(tmp_path / 'MOCK_END_TO_END_RECEIPT.json', {
        'status': 'PASS', 'fixture_only': True, 'research_api_requests_sent': 0,
        'fake_research_transport_calls': adapter.calls, 'planned_original_calls': 80,
        'parent_rows_reused': 72, 'child_slots_evaluated': 8,
        'planned_paired_blocks': 40, 'planned_independent_buildings': 10,
        'normal_empty_protocol_valid_rows': 79, 'identified_model_output_failures': 1,
        'original_parent_files_unchanged': True, 'runtime_default_checkpoint_passed': True,
        'all_selected_raw_parsed_voxel_metric_artifacts_restored_exactly': True,
        'fake_upload_interrupted_then_retried': True, 'fake_drafts_created': 1,
        'fake_assets_uploaded': 1, 'generation_retried_for_save': False,
        'evaluation': summary['evaluation_artifact'], 'checkpoint': saved['checkpoint'],
        'verification_status': result['status'], 'archive_sha256': result['archive_sha256'],
    }, exclusive=True)


def test_safe_stop_child_checkpoint_without_merged_evaluation(tmp_path, monkeypatch):
    path = fixture_plan(tmp_path, monkeypatch); adapter = MockAdapter(fail_at=1)
    execute(path, adapter)
    saved = c.create_save_only(path)
    assert not r.bound(saved['checkpoint'])['restoration']['merged_evaluation_included']
    assert adapter.calls == 1
