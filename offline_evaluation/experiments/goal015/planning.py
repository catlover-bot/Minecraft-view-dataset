"""Outcome-independent Goal015 selection and exact saved-field intervention.

No GT, outcome, credentials or network is read here. Inputs are private; public
manifests contain only identifiers, byte counts and binding hashes.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

from ...server_v2.cohort_v4 import strict_saved_json, typed_tree, _serialize_exact

FIELDS = ('summary', 'building_type', 'shape', 'dimensions_estimate', 'materials',
          'elements', 'rebuild_hints', 'uncertainties')
METADATA = frozenset(('provider', 'model', 'building', 'created_at'))
BUILDERS = ('openai', 'anthropic')
CONDITIONS = ('SUMMARY_ONLY', 'FULL_RECORD')
SELECTION_COMMIT = '9c32add47cb724b1afc01924aa047cbd5619258a'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def domain_hash(domain, *values):
    return sha(('\0'.join((domain, *map(str, values)))).encode('utf-8'))


def select_scenes(frozen_split):
    """Accept only saved selection fields, never outcome-bearing scene records."""
    frame = [{k: r[k] for k in ('scene_id', 'split', 'size_rank_stratum')}
             for r in frozen_split if r['split'] == 'evaluation']
    if len(frame) != 60 or len({r['scene_id'] for r in frame}) != 60:
        raise ValueError('exact fixed evaluation60 required')
    strata = sorted({r['size_rank_stratum'] for r in frame})
    if len(strata) != 5 or any(sum(r['size_rank_stratum'] == s for r in frame) != 12 for s in strata):
        raise ValueError('five saved strata of twelve required')
    selected = {}
    for s in strata:
        ordered = sorted((r for r in frame if r['size_rank_stratum'] == s),
                         key=lambda r: (domain_hash('goal015-scenes-v1', r['scene_id']), r['scene_id']))[:2]
        selected[s] = sorted(ordered, key=lambda r: (domain_hash('goal015-presentation-v1', r['scene_id']), r['scene_id']))
    result = []
    for round_index in (1, 2):
        for s in sorted(strata, key=lambda s: (domain_hash('goal015-strata-order-v1', round_index, s), s)):
            row = selected[s][round_index-1]
            result.append(dict(row, presentation_round=round_index,
                               selection_sha256=domain_hash('goal015-scenes-v1', row['scene_id'])))
    return result


def schedule(selected):
    rows = []
    for scene in selected:
        sid = scene['scene_id']
        first_builders = BUILDERS[::(-1 if int(domain_hash('goal015-builder-order-v1', sid), 16) & 1 else 1)]
        for repeat in (1, 2):
            for builder in first_builders[::(1 if repeat == 1 else -1)]:
                first = CONDITIONS[::(-1 if int(domain_hash('goal015-condition-order-v1', sid, builder), 16) & 1 else 1)]
                for condition in first[::(1 if repeat == 1 else -1)]:
                    rows.append({'call_id': f'goal015_call_{len(rows)+1:03d}', 'scene_id': sid,
                                 'builder': builder, 'condition': condition, 'repeat_index': repeat,
                                 'paired_block_id': f'{sid}_{builder}_r{repeat}',
                                 'size_rank_stratum': scene['size_rank_stratum'],
                                 'presentation_round': scene['presentation_round']})
    validate_schedule(rows)
    return rows


def validate_schedule(rows):
    if len(rows) != 80 or len({r['call_id'] for r in rows}) != 80:
        raise ValueError('80 unique calls required')
    cells = Counter((r['scene_id'], r['builder'], r['condition'], r['repeat_index']) for r in rows)
    scenes = {r['scene_id'] for r in rows}
    if len(scenes) != 10 or set(cells.values()) != {1} or any(
            (s,b,c,r) not in cells for s in scenes for b in BUILDERS for c in CONDITIONS for r in (1,2)):
        raise ValueError('balanced exact factorial cells required')
    if Counter(r['size_rank_stratum'] for r in rows) != {s:16 for s in range(5)}:
        raise ValueError('five strata with two buildings required')
    for start in range(0,80,2):
        a,b=rows[start:start+2]
        if a['paired_block_id']!=b['paired_block_id'] or {a['condition'],b['condition']}!=set(CONDITIONS):
            raise ValueError('adjacent paired block required')
    for s in scenes:
        for b in BUILDERS:
            first=[r['condition'] for r in rows if r['scene_id']==s and r['builder']==b and r['repeat_index']==1]
            second=[r['condition'] for r in rows if r['scene_id']==s and r['builder']==b and r['repeat_index']==2]
            if first!=second[::-1]:raise ValueError('AB/BA reversal required')
    return True


def extract_conditions(raw):
    original = strict_saved_json(raw.decode('utf-8'))
    if not isinstance(original, dict) or set(FIELDS)-set(original):
        raise ValueError('original human-readable fields missing; no imputation')
    unknown=set(original)-set(FIELDS)-METADATA
    if unknown:raise ValueError('unclassified top-level fields: '+','.join(sorted(unknown)))
    if not isinstance(original['summary'], str) or not original['summary'].strip():
        raise ValueError('nonempty original summary required')
    full={k:original[k] for k in FIELDS}
    summary={'summary':original['summary']}
    outputs={name:_serialize_exact(value).encode('utf-8') for name,value in zip(CONDITIONS,(summary,full))}
    decoded={name:strict_saved_json(raw.decode('utf-8')) for name,raw in outputs.items()}
    if decoded['SUMMARY_ONLY']['summary']!=decoded['FULL_RECORD']['summary']:
        raise ValueError('summary changed')
    if typed_tree(decoded['FULL_RECORD'])!=typed_tree(full):
        raise ValueError('typed values changed')
    evidence={'description_sha256':sha(raw), 'summary_sha256':sha(original['summary'].encode('utf-8')),
              'eight_field_typed_tree_sha256':sha(canonical(typed_tree(full))),
              'summary_utf8_bytes':len(original['summary'].encode('utf-8')),
              'top_level_fields':sorted(original), 'excluded_management_fields':sorted(set(original)&METADATA),
              'unknown_top_level_fields':[], 'summary_equal':True, 'typed_tree_equal':True,
              'serialization':'cohort_v4._serialize_exact: sorted compact UTF8, Decimal types preserved, no final newline'}
    return outputs,evidence


def descriptor(path):
    path=Path(path).absolute();raw=path.read_bytes()
    return {'path':str(path),'bytes':len(raw),'sha256':sha(raw)}


def write_json_new(path, value):
    path=Path(path)
    with path.open('xb') as stream:stream.write(canonical(value)+b'\n')
    return descriptor(path)


def prepare_sources(selected, source_rows, private_root):
    """Validate source bytes and metadata classification BEFORE writing inputs."""
    private_root=Path(private_root)
    sources={r['scene_id']:r for r in source_rows if r['description_provider']=='openai'}
    validated=[]
    for scene in selected:
        sid=scene['scene_id'];source=sources[sid];desc=descriptor(source['description_source_path'])
        if (desc['sha256'],desc['bytes'])!=(source['description_source_sha256'],source['description_source_bytes']):
            raise ValueError('bound original description changed')
        outputs,evidence=extract_conditions(Path(desc['path']).read_bytes())
        if evidence['summary_sha256']!=source['summary_sha256'] or desc['sha256']!=source['O_input_sha256']:
            raise ValueError('original O/summary binding changed')
        validated.append((scene,source,desc,outputs,evidence))
    bindings=[];conditions=[]
    for scene,source,desc,outputs,evidence in validated:
        sid=scene['scene_id']
        bindings.append(dict(scene,**evidence,description_source=desc,O_input_id=source['O_input_id'],
                             O_input_sha256=source['O_input_sha256'],description_provider='openai',
                             description_model_id=source['description_model_id']))
        for condition,raw in outputs.items():
            path=private_root/(sid+'_'+condition+'.json')
            with path.open('xb') as stream:stream.write(raw)
            conditions.append({'scene_id':sid,'condition':condition,'input':descriptor(path),
                               'summary_sha256':evidence['summary_sha256'],
                               'eight_field_typed_tree_sha256':evidence['eight_field_typed_tree_sha256'],
                               'source_description_sha256':desc['sha256'],'O_input_id':source['O_input_id']})
    return bindings,conditions
