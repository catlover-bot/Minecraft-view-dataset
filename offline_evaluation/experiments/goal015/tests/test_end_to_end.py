"""Full fixture-only runtime, local checkpoint, and CPU evaluation integration.

Every transport is MockAdapter; every GT is a one-voxel synthetic Grid. The
receipt distinguishes 80 mock attempts per scenario from zero research calls.
"""
from collections import Counter
import hashlib
from pathlib import Path
import socket
import time

from offline_evaluation.experiments.goal015 import checkpoint as checkpoint_module
from offline_evaluation.experiments.goal015 import evaluate as evaluation
from offline_evaluation.experiments.goal015 import runtime
from offline_evaluation.experiments.goal015.tests.test_runtime import fixture_plan, MockAdapter
from offline_evaluation.server_v2 import core


class IdentifiedMockAdapter(MockAdapter):
    """Observe the actual durable selected-call identity before each mock send."""
    def __init__(self, plan_path, **kwargs):
        super().__init__(**kwargs)
        self.plan_path = plan_path
        self.call_ids = []

    def send_request(self, builder, payload, **kwargs):
        ledger = runtime.read_json(self.plan_path.parent / 'enabled/ledger.json')
        pending = [row for row in ledger['calls'] if row['state'] == 'SENT_PENDING']
        assert len(pending) == 1 and pending[0]['builder'] == builder
        self.call_ids.append(pending[0]['call_id'])
        return super().send_request(builder, payload, **kwargs)


def _check_bindings(plan, ledger, summary, expected_valid):
    document = runtime.bound(summary['evaluation_artifact'])
    assert len(document['rows']) == 80
    assert document['schedule_sha256'] == summary['schedule_sha256']
    assert document['plan_binding']['plan_id'] == plan['plan_id']
    original = {row['call_id']: row for row in ledger['calls']}
    counts = Counter()
    bindings = []
    for row in document['rows']:
        saved = original[row['call_id']]
        assert all(row[field] == saved[field] for field in evaluation.IDENTITY)
        assert row['artifacts']['request_payload'] == saved['payload']
        assert row['selected_call_id'] == row['call_id']
        for name, descriptor in row['artifacts'].items():
            assert runtime.descriptor(descriptor['path']) == descriptor
            counts[name] += 1
        if row['protocol_valid']:
            assert row['artifacts']['raw'] == saved['raw']
            assert row['artifacts']['response_text_path'] == saved['response_text_path']
            assert runtime.bound(row['artifacts']['parsed.json']) == {'operations': []}
            assert runtime.bound(row['artifacts']['voxels.json']) == core.grid_to_payload(core.Grid({}))
            assert runtime.bound(row['artifacts']['metrics.json']) == row['metrics']
            assert row['metrics']['values']['aligned_occupancy_iou'] == 0.
            assert row['metrics']['values']['aligned_surface_f1_at_1'] == 0.
            assert row['metrics']['values']['aligned_occupancy_precision'] is None
            assert row['empty_output'] is True and row['nonempty'] is False
        else:
            assert row['status'] == 'OPERATIONAL_UNKNOWN'
            assert row['metrics'] is None and not evaluation.failure0_eligible(row)
            assert set(row['artifacts']) == {'request_payload'}
        bindings.append({key: row[key] for key in (*evaluation.IDENTITY, 'artifacts')})
    assert counts['request_payload'] == 80
    assert all(counts[name] == expected_valid for name in
               ('raw', 'response_text_path', 'parsed.json', 'voxels.json', 'metrics.json'))
    return {'verified_artifact_counts': dict(counts),
            'identity_and_artifact_bindings_sha256': hashlib.sha256(runtime.canonical(bindings)).hexdigest()}


def _run_scenario(parent, *, missing_first):
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    parent.mkdir()
    plan_path = fixture_plan(parent)
    plan = runtime.verify_plan(plan_path)
    plan_descriptor = runtime.descriptor(plan_path)
    adapter = IdentifiedMockAdapter(plan_path, fail_at=1 if missing_first else None)
    execution = runtime.execute(plan_path, adapter=adapter, confirmation=lambda _: True)
    initial_state = execution
    if missing_first:
        assert execution['status'] == 'SAFE_STOP'
        assert adapter.calls == execution['calls_uncertain'] == 1
        assert execution['calls_not_sent'] == 79 and execution['calls_settled'] == 0
        first = runtime.read_json(plan_path.parent / 'enabled/ledger.json')['calls'][0]
        assert first['state'] == 'UNKNOWN_USAGE' and first['failure_kind'] == 'transport_failure'
        assert first['send_status'] == 'POSSIBLY_SENT' and first['raw'] is None
        execution = runtime.execute(plan_path, adapter=adapter, resume=True)
    expected_valid = 79 if missing_first else 80
    assert execution['status'] == ('PARTIAL_WITH_UNCERTAIN_CALLS' if missing_first else 'COMPLETE')
    assert execution['calls_settled'] == expected_valid and execution['calls_not_sent'] == 0
    assert execution['calls_uncertain'] == int(missing_first)
    assert adapter.calls == 80 and len(set(adapter.call_ids)) == 80
    assert adapter.call_ids == [row['call_id'] for row in plan['schedule']]
    ledger_path = plan_path.parent / 'enabled/ledger.json'
    ledger = runtime.read_json(ledger_path)
    runtime._validate_ledger(plan, ledger, runtime.read_json(plan_path.parent / 'enabled/approval.json'))
    final_checkpoint = execution['checkpoint']
    package = runtime.bound(final_checkpoint['checkpoint'])
    assert package['status'] == 'LOCAL_VERIFIED' and package['github_status'] == 'NOT_UPLOADED'
    archive = Path(final_checkpoint['path']) / 'goal015-state.tar.gz'
    verified = checkpoint_module.verify_archive(archive, package['archive_sha256'])
    assert verified['status'] == 'VERIFIED' and verified['geometry_or_generation_reexecuted'] is False
    restored = Path(final_checkpoint['path']) / 'verified-restoration'
    restored_ledger = runtime.read_json(restored / 'enabled/ledger.json')
    assert restored_ledger['calls'] == ledger['calls']
    for row in ledger['calls']:
        for field in ('raw', 'response_text_path'):
            if row[field] is not None:
                relative = Path(row[field]['path']).relative_to(plan_path.parent)
                restored_descriptor = runtime.descriptor(restored / relative)
                assert all(restored_descriptor[key] == row[field][key] for key in ('sha256', 'bytes'))
    checkpoints = [runtime.read_json(path) for path in (plan_path.parent / 'checkpoints').glob('*/checkpoint.json')]
    reasons = Counter(item['reason'] for item in checkpoints)
    assert reasons == {'BUILDING_COMPLETE': 10, 'FINAL': 1, **({'SAFE_STOP': 1} if missing_first else {})}
    before_ledger = runtime.descriptor(ledger_path)
    gt_reads = []
    def synthetic_gt(scene):
        gt_reads.append(scene)
        return core.Grid({(0, 0, 0): 'minecraft:stone'})
    output_dir = plan_path.parent / 'evaluations/end-to-end'
    binding = {'plan_id': plan['plan_id'], 'plan': plan_descriptor, 'ledger': before_ledger}
    summary = evaluation.evaluate_saved(plan['schedule'], ledger['calls'], output_dir,
                                        synthetic_gt, fixture=True, plan_binding=binding)
    assert len(gt_reads) == expected_valid
    assert summary['planned_calls'] == 80 and summary['planned_paired_blocks'] == 40
    assert summary['planned_independent_buildings'] == 10
    assert summary['actual_metric_rows'] == summary['evaluated_protocol_valid_rows'] == expected_valid
    assert summary['research_api_requests_sent_by_evaluation'] == 0
    assert len(summary['primary_paired_estimates']) == 8
    for estimate in summary['primary_paired_estimates']:
        missing_pair = missing_first and estimate['builder'] == 'openai'
        assert estimate['estimate'] == 0. and estimate['ci95'] == [0., 0.]
        assert estimate['bootstrap_finite_draws'] == 2000 and estimate['bootstrap_undefined_draws'] == 0
        assert estimate['planned_paired_blocks'] == 20 and estimate['contributing_scenes'] == 10
        assert estimate['eligible_paired_blocks'] == (19 if missing_pair else 20)
        assert estimate['scenes_with_both_repeats'] == (9 if missing_pair else 10)
        assert estimate['scenes_with_one_repeat'] == int(missing_pair)
        assert estimate['scenes_with_no_repeats'] == estimate['failure0_condition_values'] == 0
        individual = estimate['individual_paired_differences']
        assert sum(item['difference_full_minus_summary'] is None for item in individual) == int(missing_pair)
        if missing_pair:
            assert individual[0]['summary_value'] is None and individual[0]['full_value'] == 0.
            assert individual[0]['summary_null_reason'] == 'transport_failure'
    assert sum(row['normal_empty_protocol_valid_count'] for row in summary['reliability']) == expected_valid
    assert sum(row['possibly_sent_unconfirmed_count'] for row in summary['reliability']) == int(missing_first)
    assert runtime.descriptor(ledger_path) == before_ledger
    assert runtime.descriptor(plan_path) == plan_descriptor
    assert adapter.calls == 80
    artifacts = _check_bindings(plan, ledger, summary, expected_valid)
    saved_evaluation = checkpoint_module.create_save_only(plan_path, reason='FIXTURE_EVALUATION_COMPLETE')
    assert saved_evaluation['status'] == 'LOCAL_VERIFIED'
    saved_root = Path(saved_evaluation['path']) / 'verified-restoration'
    saved_count = 0
    for path in output_dir.rglob('*'):
        if path.is_file():
            restored_path = saved_root / path.relative_to(plan_path.parent)
            assert restored_path.read_bytes() == path.read_bytes()
            saved_count += 1
    assert saved_count == expected_valid * 3 + 2
    assert runtime.descriptor(ledger_path) == before_ledger and adapter.calls == 80
    return {'case': 'first_transport_unknown_then_resume' if missing_first else 'all_normal_empty',
            'status': 'PASS', 'fixture_only': True, 'mock_transport_calls': adapter.calls,
            'unique_selected_call_ids': len(set(adapter.call_ids)), 'generation_retries': 0,
            'real_research_api_requests': 0, 'initial_execution_status': initial_state['status'],
            'final_execution_status': execution['status'], 'settled_calls': expected_valid,
            'unknown_usage_calls': int(missing_first), 'planned_calls': 80, 'planned_paired_blocks': 40,
            'planned_independent_buildings': 10, 'normal_empty_protocol_valid_rows': expected_valid,
            'primary_paired_estimates': 8, 'all_contributing_scenes': 10,
            'openai_eligible_paired_blocks': 19 if missing_first else 20,
            'anthropic_eligible_paired_blocks': 20, 'all_failure0_condition_values': 0,
            'bootstrap_finite_draws_per_estimand': 2000, 'all_estimates': 0., 'all_ci95': [0., 0.],
            'checkpoint_counts': dict(reasons), 'plan': plan_descriptor,
            'ledger': before_ledger, 'checkpoint': final_checkpoint['checkpoint'],
            'archive': runtime.descriptor(archive), 'restoration': verified,
            'post_evaluation_checkpoint': saved_evaluation['checkpoint'],
            'restored_evaluation_files': saved_count,
            'evaluation': summary['evaluation_artifact'], 'summary': runtime.descriptor(output_dir / 'summary.json'),
            'gt_source': 'SYNTHETIC_ONE_VOXEL_GRID_ONLY',
            'cpu_seconds': time.process_time() - start_cpu, 'elapsed_seconds': time.perf_counter() - start_wall,
            **artifacts}


def test_full80_runtime_checkpoint_evaluation_and_missing_resume(tmp_path, tmp_path_factory, monkeypatch):
    """Actual local lifecycle in two synthetic scenarios; no transport escapes."""
    network_attempts = []
    def network_forbidden(*args, **kwargs):
        network_attempts.append(True)
        raise AssertionError('research/network transport forbidden in integration fixture')
    monkeypatch.setattr(socket, 'create_connection', network_forbidden)
    monkeypatch.setattr(socket.socket, 'connect', network_forbidden)
    cases = [_run_scenario(tmp_path / 'normal', missing_first=False),
             _run_scenario(tmp_path / 'missing', missing_first=True)]
    assert network_attempts == []
    receipt = {'schema': 'goal015_mock_end_to_end_receipt_v1', 'status': 'PASS',
               'fixture_only': True, 'scenarios': cases, 'mock_transport_calls_total': 160,
               'real_research_api_requests': 0, 'network_attempts': 0,
               'real_gt_reads': 0, 'model_inference_performed': False,
               'implementation': {name: runtime.descriptor(module.__file__) for name, module in
                                  (('runtime', runtime), ('checkpoint', checkpoint_module), ('evaluation', evaluation))},
               'test': runtime.descriptor(__file__)}
    runtime._save(tmp_path_factory.getbasetemp() / 'MOCK_END_TO_END_RECEIPT.json', receipt, exclusive=True)
