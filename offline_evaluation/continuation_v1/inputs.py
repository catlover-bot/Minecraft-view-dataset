"""Take the already committed schedule suffix, without selecting or rewriting it."""
from collections import Counter

from ..server_v2.generation import GenerationError, object_hash


def suffix(manifest, parent_plan):
    if manifest.get('identity_hash') != object_hash(manifest['identity']):
        raise GenerationError('full_manifest_identity_mismatch')
    identity = manifest['identity']
    full = identity['full_schedule']
    if (len(full) != 960 or full[:160] != parent_plan['identity']['schedule']
            or identity['batch_schedule'] != full[:160]
            or [r['schedule_index'] for r in full] != list(range(960))):
        raise GenerationError('fixed960_prefix_or_global_index_mismatch')
    keys = [(r['provider'], r['input_id']) for r in full]
    if len(set(keys)) != 960:
        raise GenerationError('duplicate_provider_input_in_fixed_schedule')
    remaining = full[160:]
    scenes = identity['evaluation_scene_ids'][10:]
    if (len(scenes) != 50 or len(set(scenes)) != 50
            or list(dict.fromkeys(r['scene_id'] for r in remaining)) != scenes
            or set(keys[:160]) & set(keys[160:])
            or Counter((r['provider'], r['condition']) for r in remaining)
               != {(p, c): n for p in ('openai', 'anthropic')
                   for c, n in [('O', 150), ('J', 150), ('P', 50), ('E', 50)]}):
        raise GenerationError('fixed800_scope_mismatch')
    for r in remaining:
        if r['protocol_id'] != identity['protocol_id']:
            raise GenerationError('semantic_protocol_mismatch')
    segments = []
    for i in range(5):
        rows = remaining[i * 160:(i + 1) * 160]
        group = scenes[i * 10:(i + 1) * 10]
        if list(dict.fromkeys(r['scene_id'] for r in rows)) != group:
            raise GenerationError('fixed_segment_building_boundary_mismatch')
        segments.append({'segment': i + 1, 'local_start': i * 160,
                         'local_end_exclusive': (i + 1) * 160,
                         'global_start': 160 + i * 160,
                         'global_end_exclusive': 160 + (i + 1) * 160,
                         'scene_ids': group})
    return remaining, scenes, segments
