"""Goal008 source and representation audit; no scoring, statistics, or API calls.

Reuse Goal007's completed restore receipts and rehash existing evidence. O is the
saved JSON record, including metadata. J preserves its exact typed JSON tree.
Only safe, explicitly selected response metadata is retained in sampling reports.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import inspect
import json
from pathlib import Path

from ..goal007 import audit as old_audit
from ..goal007 import cli as old_cli
from ...server_v2.cohort_v4 import canonicalize_saved_text, strict_saved_json, typed_tree
from ...server_v2.core import grid_to_payload
from ...server_v2.data import load_scene, sha256_file
from ...server_v2.generation import digest, object_hash, utc_now
from ...server_v2.oracle import partial_oracle
from ...resmoke_v2 import adapter, cuboids

ROOT = Path(__file__).resolve().parents[3]
GOAL007 = ROOT / 'reports/server_only/goal007'
CANONICAL = Path('/root/minecraft-runs/goal007-analysis-20261016T130420Z-a8a7a399/analysis-final')
MANIFEST_SHA256 = 'ef05c2ed93247802e9a68a00d700d39a8934d6231c7787315ab37b01d2467714'
VERSION = 'goal008_phase0_source_audit_v1'
DESCRIPTION_PROVIDERS = ('openai', 'anthropic', 'gemini')
BUILDERS = ('openai', 'anthropic')
SAMPLING_FIELDS = ('temperature', 'top_p', 'top_k', 'seed', 'frequency_penalty', 'presence_penalty')
HUMAN_FIELDS = ('summary', 'building_type', 'shape', 'dimensions_estimate', 'materials', 'elements', 'rebuild_hints', 'uncertainties')
read_json = old_audit.read_json
regular = old_audit.regular
require = old_audit.require
write_new = old_audit.write_new
AuditError = old_audit.AuditError


def verify_goal007():
    """No download and no original validator or quality metric replay."""
    mirror, canonical = regular(GOAL007 / 'MANIFEST.json'), regular(CANONICAL / 'MANIFEST.json')
    require(sha256_file(mirror) == sha256_file(canonical) == MANIFEST_SHA256, 'goal007_manifest_mismatch')
    manifest = read_json(canonical)
    require(old_cli.implementation_hashes() == manifest['code_hashes'], 'frozen_goal007_code_changed')
    for name, descriptor in manifest['products'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe_goal007_product_path')
        a, b = regular(CANONICAL / relative), regular(GOAL007 / relative)
        require(sha256_file(a) == sha256_file(b) == descriptor['sha256']
                and a.stat().st_size == b.stat().st_size == descriptor['bytes'], 'goal007_product_mirror_changed:' + name)
    require(len(manifest['products']) == 56, 'goal007_product_count_changed')
    gate = old_audit.verify_gate(manifest['audit_gate_path'])
    require(manifest['sources'] == gate['sources'] and manifest['github_source_checkpoints'] == gate['github_checkpoints'], 'goal007_source_chain_changed')
    require(manifest['source_files'] == read_json(gate['inventory_path'])['files'], 'goal007_inventory_mismatch')
    require(manifest['fixed_slots'] == 960 and manifest['fixed_scope_attempts'] == 961 and manifest['historical_calls'] == 991
            and manifest['missing_global_indices'] == [942] and manifest['all_slots_measured'] is False, 'goal007_analysis_scope_changed')
    require(manifest['cost_validation']['combined_budget_usd'] == '21.56083625'
            and manifest['cost_validation']['unknown_usage_holds_released'] is False, 'goal007_accounting_changed')
    source_manifest = read_json(regular(manifest['source_manifest']['path']))
    require(sha256_file(manifest['source_manifest']['path']) == manifest['source_manifest']['sha256']
            and source_manifest['identity_hash'] == object_hash(source_manifest['identity']), 'fixed_manifest_changed')
    cohort_path = regular(source_manifest['identity']['cohort_path'])
    require(sha256_file(cohort_path) == source_manifest['identity']['cohort_file_sha256'], 'cohort_source_changed')
    slots = read_json(GOAL007 / 'data/final_slots_960.json')
    require(len(slots) == 960 and [r['global_schedule_index'] for r in slots] == list(range(960)), 'final_slot_order_changed')
    require(Counter(r['status'] for r in slots) == {'success': 934, 'failed': 25, 'operational_unmeasurable': 1}, 'final_slot_status_changed')
    require(all(slots[942].get(k) is None for k in ('metrics', 'usage', 'response')), 'missing942_imputed')
    return manifest, gate, source_manifest, read_json(cohort_path), slots


def bind_description(scene, provider, source, gt, original_rows, j_rows, slots, *, root=ROOT):
    """Bind one saved JSON description to both Builders and lossless J inputs."""
    root = Path(root).absolute()
    relative = Path(source['path'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'description_path_escape')
    path = regular(root / relative)
    raw = path.read_bytes()
    require(digest(raw) == source['sha256'], 'description_source_hash_changed')
    text = raw.decode('utf-8')
    canonical, proof = canonicalize_saved_text(text)
    parsed = strict_saved_json(text)
    require(isinstance(parsed, dict), 'description_must_be_json_object')
    require(source.get('typed_tree_sha256') == proof['typed_tree_sha256'] and source.get('typed_tree_equal') is True, 'frozen_description_typed_tree_mismatch')
    require(set(r['provider'] for r in original_rows) == set(BUILDERS) and len(original_rows) == 2
            and set(r['provider'] for r in j_rows) == set(BUILDERS) and len(j_rows) == 2, 'both_builder_inputs_required')
    for condition, records, expected in [('O', original_rows, text), ('J', j_rows, canonical)]:
        for record in records:
            require(record['scene_id'] == scene and record['description_model'] == provider and record['condition'] == condition
                    and record['source_sha256'] == source['sha256'] and record['input_text'] == expected
                    and record['input_sha256'] == digest(expected.encode('utf-8')), 'description_input_binding_changed')
        require(len({r['input_id'] for r in records}) == 1, 'builders_must_share_semantic_input')
    observations = {}
    for record in original_rows:
        row = slots[record['schedule_index']]
        require(row['global_schedule_index'] == record['schedule_index'] and row['condition'] == 'O'
                and row['provider'] == record['provider'] and row['description_model'] == provider
                and row['scene_id'] == scene and row['input_id'] == record['input_id']
                and row['input_sha256'] == record['input_sha256'] and row['gt_grid_sha256'] == gt['grid_sha256'], 'O_selected_slot_binding_changed')
        observations[record['provider']] = {k: row[k] for k in ('global_schedule_index', 'attempt_id', 'status', 'source_label', 'source_plan_identity', 'completed_request_outcome', 'usage_known')}
        observations[record['provider']]['builder_model'] = row['builder_model']
    gt_directory = Path(gt['building_directory'])
    for name, expected in [('bbox.json', gt['bbox_sha256']), ('voxels.npy', gt['voxels_sha256'])]:
        require(sha256_file(regular(gt_directory / 'gt' / name)) == expected, 'description_GT_source_changed')
    require(isinstance(parsed.get('provider'), str) and isinstance(parsed.get('model'), str), 'description_model_metadata_missing')
    return {'scene_id': scene, 'description_provider': provider, 'description_model_id': parsed['model'],
            'description_source_provider_metadata': parsed['provider'], 'source_path': str(path), 'source_sha256': source['sha256'],
            'source_bytes': len(raw), 'source_repository': source.get('repo'), 'source_git_ref': source.get('ref'), 'source_blob_hash': source.get('blob_hash'),
            'input_id': original_rows[0]['input_id'], 'input_sha256': original_rows[0]['input_sha256'],
            'j_input_id': j_rows[0]['input_id'], 'j_input_sha256': j_rows[0]['input_sha256'],
            'typed_tree_sha256': proof['typed_tree_sha256'], 'O_J_typed_tree_equal': True,
            'gt_descriptor': deepcopy(gt), 'gt_descriptor_sha256': object_hash(gt),
            'gt_bbox_path': str(gt_directory / 'gt/bbox.json'), 'gt_voxels_path': str(gt_directory / 'gt/voxels.npy'),
            'builder_observations': observations, 'roles': {'description_provider': 'image-to-text saved description generator',
                'builder_provider': 'independent text-to-actions reconstruction model; repeated observations of the same description'},
            'score_status': 'NOT_COMPUTED_BY_SOURCE_AUDIT'}


def description_source_map(manifest, cohort, slots):
    identity = manifest['identity']; full = identity['full_schedule']
    scenes = {s['scene_id']: s for s in cohort['scenes']}
    indexed = defaultdict(list)
    for record in full:
        indexed[(record['scene_id'], record['description_model'], record['condition'])].append(record)
    mapped = []
    for scene in identity['evaluation_scene_ids']:
        for provider in DESCRIPTION_PROVIDERS:
            mapped.append(bind_description(scene, provider, scenes[scene]['description_sources'][provider], identity['gt_by_scene'][scene],
                indexed[(scene, provider, 'O')], indexed[(scene, provider, 'J')], slots))
    require(len(mapped) == 180 and len({(r['scene_id'], r['description_provider']) for r in mapped}) == 180, 'exact180_description_sources_required')
    require(Counter(r['description_provider'] for r in mapped) == {p: 60 for p in DESCRIPTION_PROVIDERS}, 'description_provider_scope_changed')
    return mapped


def _safe_scalar(value):
    return value is None or type(value) in (str, int, float, bool)


def safe_response_sampling(raw):
    """Extract only sampling metadata; content, URLs and response IDs excluded."""
    result = {key: raw[key] for key in SAMPLING_FIELDS if key in raw and _safe_scalar(raw[key])}
    for key in ('model', 'service_tier', 'max_output_tokens', 'max_tokens'):
        if key in raw and _safe_scalar(raw[key]): result[key] = raw[key]
    reasoning = raw.get('reasoning')
    if isinstance(reasoning, dict):
        result['reasoning'] = {k: reasoning[k] for k in ('effort', 'mode', 'context') if k in reasoning and _safe_scalar(reasoning[k])}
    return result


def sampling_audit(plans):
    attempts, payload_fields = [], defaultdict(Counter)
    response_counts, missing_raw = defaultdict(Counter), Counter()
    for label, plan in plans.items():
        identity = plan['identity']; directory = Path(identity['enabled_directory']); ledger = read_json(directory / 'ledger.json')
        for record, entry in zip(identity['schedule'], ledger['entries']):
            payload = adapter.build_payload(record['provider'], record['model'], identity['prompt'], record['input_text'], identity['budgets']['max_output_tokens'])
            require(object_hash(payload) == entry['request_payload_hash'], 'attempt_payload_hash_mismatch')
            if record['condition'] not in ('O', 'J'): continue
            provider = record['provider']
            for key in SAMPLING_FIELDS:
                payload_fields[provider][key + (':explicit' if key in payload else ':omitted')] += 1
            metadata = None
            if 'raw.json' in entry['artifacts']:
                descriptor = entry['artifacts']['raw.json']; path = regular(directory / descriptor['path'])
                require(path.is_relative_to(directory) and sha256_file(path) == descriptor['sha256'] and path.stat().st_size == descriptor['bytes'], 'response_metadata_source_changed')
                metadata = safe_response_sampling(read_json(path))
                response_counts[provider][json.dumps(metadata, sort_keys=True, separators=(',', ':'))] += 1
            else: missing_raw[provider] += 1
            attempts.append({'source_label': label, 'plan_identity_hash': plan['identity_hash'], 'attempt_id': entry['attempt_id'],
                'global_schedule_index': record['schedule_index'], 'scene_id': record['scene_id'], 'description_provider': record['description_model'],
                'builder_provider': provider, 'builder_model': record['model'], 'condition': record['condition'],
                'request_payload_sha256': entry['request_payload_hash'], 'retry': entry['retry'], 'source_attempt_id': entry.get('source_attempt_id'),
                'status': entry['status'], 'http_status_code': entry.get('http_status_code'), 'usage_known': entry.get('usage') is not None,
                'explicit_sampling_parameters': {k: payload[k] for k in SAMPLING_FIELDS if k in payload},
                'raw_response_sha256': entry['artifacts'].get('raw.json', {}).get('sha256'), 'reported_sampling_metadata': metadata})
    require(len(attempts) == 721 and Counter(r['condition'] for r in attempts) == {'O': 361, 'J': 360}, 'OJ_attempt_frequency_changed')
    repetitions = Counter((r['scene_id'], r['description_provider'], r['builder_provider'], r['condition']) for r in attempts)
    require(Counter(repetitions.values()) == {1: 719, 2: 1}, 'unexpected_scientific_replication')
    duplicate = [r for r in attempts if repetitions[(r['scene_id'], r['description_provider'], r['builder_provider'], r['condition'])] == 2]
    require([r['global_schedule_index'] for r in duplicate] == [940, 940] and duplicate[1]['source_attempt_id'] == duplicate[0]['attempt_id']
            and duplicate[0]['condition'] == duplicate[1]['condition'] == 'O'
            and duplicate[0]['http_status_code'] == 401 and duplicate[1]['status'] == 'success'
            and duplicate[0]['request_payload_sha256'] == duplicate[1]['request_payload_sha256'], 'approved940_retry_changed')
    return {'status': 'PASS', 'attempts': attempts, 'OJ_attempts': len(attempts), 'O_attempts': 361, 'J_attempts': 360,
        'distinct_builder_description_condition_slots': len(repetitions), 'per_slot_attempt_frequencies': {str(k): v for k, v in Counter(repetitions.values()).items()},
        'request_parameter_presence': {p: dict(counts) for p, counts in payload_fields.items()},
        'response_metadata_distribution': {p: [{'metadata': json.loads(metadata), 'responses': count} for metadata, count in counts.items()] for p, counts in response_counts.items()},
        'attempts_without_raw_response': dict(missing_raw), 'approved940_retry': duplicate,
        'controlled_sampling_replicates': False, 'explicit_decoding_seed': None,
        'interpretation': 'O and J are separate single-output generation calls for the same typed JSON information; one additional O attempt is an approved authentication retry. No payload seed, temperature, top-p or penalties are explicitly set. Provider-reported values describe returned metadata, not an experimental guarantee of deterministic decoding. Anthropic unreported defaults remain unknown.',
        'cohort_and_statistics_seed_is_not_decoding_seed': True, 'research_api_requests_sent': 0, 'api_key_queries': 0}


def representation_audit(manifest, mapped):
    identity = manifest['identity']; schedule = identity['full_schedule']; oracle_rows = []
    for scene in identity['evaluation_scene_ids']:
        gt = identity['gt_by_scene'][scene]; grid = load_scene(gt['building_directory']).grid
        require(object_hash(grid_to_payload(grid)) == gt['grid_sha256'], 'GT_grid_binding_changed')
        p_records = [r for r in schedule if r['scene_id'] == scene and r['condition'] == 'P']
        e_records = [r for r in schedule if r['scene_id'] == scene and r['condition'] == 'E']
        p = json.loads(p_records[0]['input_text']); spec = json.loads(e_records[0]['input_text'].split('\n', 1)[1])
        require(len(p_records) == len(e_records) == 2 and p_records[0]['input_id'] == p_records[1]['input_id']
                and e_records[0]['input_id'] == e_records[1]['input_id'], 'oracle_builder_inputs_mismatch')
        require(p == partial_oracle(grid), 'partial_oracle_GT_attributes_changed')
        require(e_records[0]['input_text'] == e_records[1]['input_text'] == cuboids.exact_builder_text(spec), 'exact_oracle_serialization_changed')
        require(cuboids.decode_exact(spec).as_world().to_dict() == grid.as_world().to_dict(), 'exact_oracle_GT_reconstruction_mismatch')
        oracle_rows.append({'scene_id': scene, 'gt_grid_sha256': gt['grid_sha256'], 'P_input_id': p_records[0]['input_id'],
            'P_input_sha256': p_records[0]['input_sha256'], 'P_provided_attributes': p['provided_attributes'], 'P_not_provided': p['not_provided'],
            'E_input_id': e_records[0]['input_id'], 'E_input_sha256': e_records[0]['input_sha256'], 'E_coordinate_frame': spec['coordinate_frame'],
            'E_palette_entries': len(spec['palette']), 'E_cuboids': len(spec['cuboids']), 'E_decoded_GT_exact': True})
    fields, schema_types, models, samples = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter), []
    first_scene = identity['evaluation_scene_ids'][0]
    for row in mapped:
        value = read_json(row['source_path']); provider = row['description_provider']; models[provider][row['description_model_id']] += 1
        for key in HUMAN_FIELDS:
            fields[provider][key] += int(key in value)
            schema_types[provider][key + ':' + type(value.get(key)).__name__] += 1
        if row['scene_id'] == first_scene:
            samples.append({'scene_id': row['scene_id'], 'description_provider': provider, 'source_path': row['source_path'],
                'source_sha256': row['source_sha256'], 'input_sha256': row['input_sha256'], 'typed_tree_sha256': row['typed_tree_sha256'],
                'top_level_fields': list(value), 'excerpt': {'summary': value.get('summary', '')[:300],
                    'dimensions_estimate': value.get('dimensions_estimate'), 'materials_first_two': value.get('materials', [])[:2],
                    'elements_first_two': value.get('elements', [])[:2]}, 'excerpt_is_partial': True})
    evidence = []
    for fn in (canonicalize_saved_text, typed_tree, partial_oracle, cuboids.exact_builder_text, cuboids.decode_exact, adapter.build_payload):
        path = Path(inspect.getsourcefile(fn)); start = inspect.getsourcelines(fn)[1]
        evidence.append({'function': fn.__module__ + '.' + fn.__name__, 'path': str(path), 'line': start, 'sha256': sha256_file(path)})
    return {'status': 'PASS', 'O': 'The complete saved v4 JSON visual-description record itself, including human-readable fields and original metadata; not free-form prose.',
        'J': 'Exact typed-JSON-tree-preserving reserialization of O, with sorted keys and compact formatting; not historical Structured IR.',
        'P': 'Partial ground-truth oracle: occupied XYZ dimensions, world bbox, exact material set and voxel count; per-voxel geometry/material assignment omitted.',
        'E': 'Complete discrete ground-truth cuboid specification with palette and coordinate semantics; exact decode verified for60 scenes. Valid-only outcomes condition on successful model output and are not a whole-population upper bound.',
        'oracle_effect_limit': 'P/E also change available GT information and Builder decoding/execution demands. Their differences do not identify pure language-information loss.',
        'description_provider_models': {p: dict(v) for p,v in models.items()},
        'builder_models': dict(adapter.MODELS), 'human_readable_field_presence': {p: dict(v) for p,v in fields.items()},
        'human_readable_field_types': {p: dict(v) for p,v in schema_types.items()}, 'samples': samples,
        'oracle_GT_verification': oracle_rows, 'code_evidence': evidence, 'description_sources_verified': len(mapped),
        'O_J_typed_tree_equal_count': len(mapped), 'scores_computed': 0}


def phase0_markdown(semantics, sampling):
    lines = ['# Goal008 Phase 0: representations and generation provenance', '',
        'No proxy scores, correlations, provider requests or key lookups were performed by this audit.', '']
    for name in ('O','J','P','E'): lines += [f"**{name}.** {semantics[name]}", '']
    lines += [semantics['oracle_effect_limit'], '',
        'The three description providers produced saved image-to-text records. The two Builders independently generated action programs from each record. These are separate model roles.', '',
        '## Human-readable content inside JSON', '',
        'The record contains prose in summary, shape descriptions, material roles, reconstruction hints and uncertainties, alongside typed dimensions and lists of elements/materials. Original provider/model/building/time metadata is also retained in O and J. Full records remain at their canonical source paths.', '',
        '```json', json.dumps(semantics['human_readable_field_presence'], indent=2), '```', '',
        '## Source-bound examples', '']
    for sample in semantics['samples']:
        lines += [f"{sample['description_provider']} / {sample['scene_id']}: `{sample['source_path']}`; SHA256 `{sample['source_sha256']}`. Partial excerpt only:", '',
                  '```json', json.dumps(sample['excerpt'], ensure_ascii=False, indent=2), '```', '']
    lines += ['## Actual O/J call and sampling audit', '', sampling['interpretation'], '',
        f"O has {sampling['O_attempts']} attempts across360 slots, including the original401 and approved successful retry at global940; J has {sampling['J_attempts']} attempts across360 slots. Global942 remains the separate unknown-usage HTTP520 observation. There is no controlled repeated-sampling design.", '',
        'The seed20261013 used for cohort selection and statistical resampling is not a decoding seed. O/J contrasts cannot by themselves isolate a formatting-only causal effect from provider decoding non-determinism and single-output variability.', '',
        'Request fields and safe response-metadata distributions are recorded in `sampling_audit.json`. Unreported provider parameters remain unknown; omission never means temperature0 or a fixed seed.', '',
        '## Frozen code evidence', '']
    for evidence in semantics['code_evidence']:
        lines.append(f"- `{evidence['function']}`: [{Path(evidence['path']).name}]({evidence['path']}:{evidence['line']}), SHA256 `{evidence['sha256']}`.")
    lines += ['', 'The source gate preserves Goal007’s20 confirmatory comparisons and its RQ2 status NOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES. A later CPU attribute proxy is new exploratory evidence, not a recovered historical quality score.', '']
    return '\n'.join(lines)


def run_audit(output_directory):
    output = Path(output_directory).absolute()
    require(not output.exists(), 'new_goal008_audit_directory_required')
    before_goal007_hashes = old_cli.implementation_hashes()
    manifest, gate, fixed, cohort, slots = verify_goal007()
    plans = {label: read_json(ref['plan_path']) for label,ref in manifest['sources'].items()}
    # Restore original acquisition ordering for the approved940 duplicate check.
    plans = {label: plans[label] for label in ('parent160','original781','supplement3','supplement17')}
    with old_audit.source_locks(plans), old_audit.no_python_network():
        old_audit.process_audit()
        before = old_audit.inventory([], [GOAL007, CANONICAL])
        output.mkdir(parents=True, exist_ok=False)
        write_new(output / 'inventory_before.json', before)
        mapped = description_source_map(fixed, cohort, slots)
        semantics = representation_audit(fixed, mapped)
        sampling = sampling_audit(plans)
        old_audit.verify_gate(gate)
        old_audit.verify_inventory(before)
        require(old_cli.implementation_hashes() == before_goal007_hashes == manifest['code_hashes'], 'goal008_addition_changed_frozen_implementation_hashes')
        old_audit.process_audit()
        for name,value in [('description_sources.json',mapped),('representation_semantics.json',semantics),('sampling_audit.json',sampling),('inventory_after.json',before)]:
            write_new(output/name,value)
        with (output/'phase0_semantics.md').open('x',encoding='utf-8') as stream: stream.write(phase0_markdown(semantics,sampling))
        files={str(p):{'sha256':sha256_file(p),'bytes':p.stat().st_size}for p in sorted(output.iterdir())if p.is_file()}
        report={'version':VERSION,'status':'PASS','audited_at_utc':utc_now(),'goal007_manifest_path':str(GOAL007/'MANIFEST.json'),
            'goal007_manifest_sha256':MANIFEST_SHA256,'goal007_canonical_output':str(CANONICAL),'goal007_mirrored_products_verified':56,
            'goal007_audit_gate_path':manifest['audit_gate_path'],'goal007_audit_gate_sha256':manifest['audit_gate_sha256'],
            'original_immutable_file_count':len(manifest['source_files']),'goal007_inventory_path':str(output/'inventory_before.json'),
            'goal007_file_count':len(before['files']),'goal007_inventories_unchanged':True,'frozen_implementation_hashes_unchanged':True,
            'frozen_implementation_hashes':manifest['code_hashes'],'description_sources_path':str(output/'description_sources.json'),
            'description_sources_count':180,'description_provider_counts':{p:60 for p in DESCRIPTION_PROVIDERS},
            'source_match':'180/180 exact original bytes, typed-tree, input and GT binding',
            'fixed_slots':960,'valid':934,'model_output_failures':25,'operational_missing':1,'historical_calls':991,'fixed_scope_attempts':961,
            'predeclared_comparisons_unchanged':20,'unknown_usage_holds':manifest['cost_validation']['unknown_usage_holds'],
            'budget_accounted_usd':'21.56083625','historical_RQ2_status':'NOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES',
            'existing_restore_release_ids':[414157245,414396750,414429861],'existing_restore_proofs_reused':True,
            'new_downloads':0,'scores_computed':0,'correlations_computed':0,'research_api_requests_sent':0,'api_key_queries':0,'github_mutations':0,
            'files':files,'audit_code_sha256':sha256_file(__file__)}
        write_new(output/'source_gate.json',report)
        return report


def verify_report(report_or_path):
    report=read_json(report_or_path) if isinstance(report_or_path,(str,Path)) else report_or_path
    require(report.get('version')==VERSION and report.get('status')=='PASS','goal008_source_gate_not_passed')
    for path,descriptor in report['files'].items():
        require(sha256_file(regular(path))==descriptor['sha256'] and Path(path).stat().st_size==descriptor['bytes'],'goal008_audit_output_changed')
    old_audit.verify_inventory(read_json(report['goal007_inventory_path']))
    require(old_cli.implementation_hashes()==report['frozen_implementation_hashes'],'frozen_implementation_changed_after_goal008_audit')
    old_audit.verify_gate(report['goal007_audit_gate_path'])
    require(sha256_file(report['goal007_manifest_path'])==report['goal007_manifest_sha256']==MANIFEST_SHA256,'goal007_manifest_changed_after_goal008_audit')
    rows=read_json(report['description_sources_path'])
    require(len(rows)==180 and len({(r['scene_id'],r['description_provider'])for r in rows})==180,'description_source_map_changed')
    for row in rows:
        require(sha256_file(regular(row['source_path']))==row['source_sha256'],'description_source_changed_after_audit')
        require(sha256_file(regular(row['gt_bbox_path']))==row['gt_descriptor']['bbox_sha256'] and
                sha256_file(regular(row['gt_voxels_path']))==row['gt_descriptor']['voxels_sha256'],'GT_source_changed_after_audit')
    return report


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-directory',required=True)
    args=parser.parse_args(argv)
    report=run_audit(args.output_directory)
    print(json.dumps({'status':report['status'],'descriptions':report['description_sources_count'],'source_gate':str(Path(args.output_directory)/'source_gate.json')}))


if __name__=='__main__': main()
