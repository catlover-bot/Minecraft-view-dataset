"""Reuse authenticated parent results and evaluate only the eight residual slots."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from ...experiments.goal015 import evaluate as frozen
from ...experiments.goal015 import runtime as original
from ...server_v2.data import load_scene

require = original.require


def select_calls(parent_plan, parent_ledger, child_plan, child_ledger=None):
    """Pure one-to-one selection; no parent row or schedule is rewritten."""
    schedule = parent_plan['schedule']
    frozen._schedule(schedule, parent_plan.get('fixture_only') is True)
    parents = parent_ledger['calls']
    require(len(schedule) == len(parents) == 80, 'original eighty slots required')
    require(all(all(row.get(k) == v for k, v in source.items())
                for source, row in zip(schedule, parents)), 'parent schedule identity differs')
    require(sum(r['state'] == 'SETTLED' for r in parents) == 72
            and sum(r['state'] == 'NOT_SENT' for r in parents) == 8, 'fixed parent72/8 required')
    residual = [source for source, row in zip(schedule, parents) if row['state'] == 'NOT_SENT']
    require(child_plan['schedule'] == residual, 'child must preserve exact ordered residual eight')
    require(child_plan['parent_plan_id'] == parent_plan['plan_id'], 'child parent identity differs')
    children = child_ledger['calls'] if child_ledger is not None else []
    if child_ledger is not None:
        require(child_ledger['plan_id'] == child_plan['plan_id'] and len(children) == 8,
                'eight child ledger slots required')
        require(all(all(row.get(k) == v for k, v in source.items())
                    for source, row in zip(residual, children)), 'child ledger identity/order differs')
    indexed = {row['call_id']: row for row in children}
    require(len(indexed) == len(children), 'duplicate child selected call')
    selected, provenance = [], []
    for source, parent in zip(schedule, parents):
        use_child = parent['state'] == 'NOT_SENT' and source['call_id'] in indexed
        row = indexed[source['call_id']] if use_child else parent
        selected.append(deepcopy(row))
        provenance.append({**{key: source[key] for key in frozen.IDENTITY},
            'selected_source': 'CHILD' if use_child else 'PARENT',
            'source_plan_id': child_plan['plan_id'] if use_child else parent_plan['plan_id'],
            'source_state': row['state'], 'parent_original_state': parent['state']})
    return selected, provenance


def _verify_artifacts(row, saved):
    require(all(row[key] == saved[key] for key in frozen.IDENTITY), 'evaluation row identity differs')
    require(row['state'] == saved['state'], 'evaluation row state differs')
    require(row['artifacts']['request_payload'] == saved['payload'], 'evaluation request binding differs')
    for name in ('raw', 'response_text_path'):
        require(row['artifacts'].get(name) == saved.get(name), 'evaluation response binding differs')
    for artifact in row['artifacts'].values():
        frozen._read_descriptor(artifact)
    if row['metrics'] is not None:
        require(original.bound(row['artifacts']['metrics.json']) == row['metrics'], 'metric artifact differs')


def parent_evaluation(plan, parent):
    document = original.bound(plan['parent']['evaluation'])
    binding = document.get('plan_binding', {})
    require(binding.get('plan_id') == parent['plan']['plan_id']
            and binding.get('plan') == plan['parent']['plan']
            and binding.get('ledger') == plan['parent']['ledger'], 'parent evaluation binding differs')
    require(len(document['rows']) == 80, 'parent evaluation must retain eighty rows')
    require(document['schedule_sha256'] == frozen._sha(frozen._json_bytes(parent['plan']['schedule'])),
            'parent evaluated schedule differs')
    for row, saved in zip(document['rows'], parent['ledger']['calls']):
        _verify_artifacts(row, saved)
    return document


def _child_state(plan):
    from . import runtime
    path = Path(plan['canonical_root']) / 'enabled/ledger.json'
    if not path.exists():
        return None, None
    ledger = runtime.verify_enabled(plan)['ledger']
    return ledger, original.descriptor(path)


def _binding(plan, plan_path, child_descriptor):
    return {'plan_id': plan['parent_plan_id'], 'plan': plan['parent']['plan'],
            'ledger': plan['parent']['ledger'], 'parent_evaluation': plan['parent']['evaluation'],
            'child_plan_id': plan['plan_id'], 'child_plan': original.descriptor(plan_path),
            'child_ledger': child_descriptor}


def find_existing_evaluation(plan_path):
    """Find authenticated merged output for exactly the current child ledger."""
    from . import runtime
    plan = runtime.verify_plan(plan_path)
    parent = runtime.verify_parent(plan)
    authenticated_parent = parent_evaluation(plan, parent)
    ledger, desc = _child_state(plan)
    calls, provenance = select_calls(parent['plan'], parent['ledger'], plan, ledger)
    expected = _binding(plan, plan_path, desc)
    directory = Path(parent['plan']['canonical_root']) / 'evaluations'
    for path in sorted(directory.glob('goal015a-*/summary.json'), reverse=True):
        summary = original.read_json(path)
        binding = summary.get('plan_binding', {})
        if not all(binding.get(key) == value for key, value in expected.items()):
            continue
        sources = original.bound(binding['selected_sources'])
        require(sources['rows'] == provenance, 'selected sources differ')
        data = original.bound(summary['evaluation_artifact'])
        require(data['plan_binding'] == binding and len(data['rows']) == 80, 'merged evaluation binding differs')
        for row, call, source, original_row in zip(data['rows'], calls, provenance,
                                                  authenticated_parent['rows']):
            _verify_artifacts(row, call)
            require(row['selected_source'] == source, 'row selected source differs')
            if source['parent_original_state'] == 'SETTLED':
                require({key: value for key, value in row.items() if key != 'selected_source'} == original_row,
                        'reused parent evaluation row changed')
        return summary
    return None


def evaluate_plan(plan_path, output_dir=None):
    """CPU only; preserve all parent72 results, artifacts and failures exactly."""
    from . import runtime
    plan = runtime.verify_plan(plan_path)
    with runtime.paired_locks(plan, create_child=False):
        require(runtime.verify_plan(plan_path) == plan, 'child plan changed while locking')
        if output_dir is None:
            existing = find_existing_evaluation(plan_path)
            if existing is not None:
                return existing
        parent = runtime.verify_parent(plan)
        authenticated = parent_evaluation(plan, parent)
        child, child_descriptor = _child_state(plan)
        selected, sources = select_calls(parent['plan'], parent['ledger'], plan, child)
        root = Path(parent['plan']['canonical_root']) / 'evaluations'
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        target = original.path_safe(output_dir or root / ('goal015a-' + plan['plan_id'][:12] + '-' + stamp))
        require(target.parent == root and target.name.startswith('goal015a-'), 'merged output must use new parent evaluations/goal015a-* directory')
        target.mkdir(parents=True, exist_ok=False)
        grids = {}
        def gt(scene):
            if scene not in grids:
                source = parent['plan']['gt_sources'][scene]
                building = original.path_safe(source['building_directory'])
                for relative, field in (('gt/voxels.npy', 'voxels_sha256'), ('gt/bbox.json', 'bbox_sha256')):
                    require(original.descriptor(building / relative)['sha256'] == source[field], 'original GT bytes differ')
                loaded = load_scene(building)
                require(loaded.scene_id == scene and frozen.frozen_grid_sha256(loaded.grid) == source['grid_sha256'],
                        'original GT identity differs')
                grids[scene] = loaded.grid
            return grids[scene]
        rows = []
        for index, (record, saved, source, old) in enumerate(zip(parent['plan']['schedule'], selected,
                sources, authenticated['rows'])):
            if source['parent_original_state'] == 'SETTLED':
                row = deepcopy(old)
            else:
                observation = deepcopy(saved)
                frozen._read_descriptor(record['payload'])
                if observation.get('response_received'):
                    require(observation.get('raw') is not None, 'received child response requires raw')
                if observation.get('raw'):
                    frozen._read_descriptor(observation['raw'])
                if observation.get('response_text_path'):
                    observation['response_text'] = frozen._read_descriptor(observation['response_text_path']).decode('utf-8')
                elif observation['state'] == 'SETTLED' and not observation.get('failure_kind'):
                    require(False, 'settled child response text missing')
                row = frozen.evaluate_one(record, observation, lambda: gt(record['scene_id']))
                row['gt_source'] = parent['plan']['gt_sources'][record['scene_id']]
                product = target / f'{index:03d}'
                product.mkdir()
                for field, name in (('parsed_document', 'parsed.json'), ('voxel_payload', 'voxels.json')):
                    value = row.pop(field)
                    if value is not None:
                        row['artifacts'][name] = frozen._write_new(product / name, value)
                if row['metrics'] is not None:
                    row['artifacts']['metrics.json'] = frozen._write_new(product / 'metrics.json', row['metrics'])
            row['selected_source'] = source
            rows.append(row)
        binding = _binding(plan, plan_path, child_descriptor)
        binding['selected_sources'] = frozen._write_new(target / 'SELECTED_SOURCES.json',
            {'parent_plan_id': plan['parent_plan_id'], 'child_plan_id': plan['plan_id'], 'rows': sources})
        summary = frozen.aggregate_rows(parent['plan']['schedule'], rows, fixture=plan.get('fixture_only') is True)
        summary.update(plan_binding=binding, parent_rows_reused=72, child_slots_evaluated=8,
            schedule_sha256=frozen._sha(frozen._json_bytes(parent['plan']['schedule'])),
            research_api_requests_sent=0)
        summary['evaluation_artifact'] = frozen._write_new(target / 'evaluation_80.json', {
            'rows': rows, 'planned_calls': 80, 'plan_binding': binding,
            'schedule_sha256': summary['schedule_sha256'], 'research_api_requests_sent': 0})
        frozen._write_new(target / 'summary.json', summary)
        return summary
