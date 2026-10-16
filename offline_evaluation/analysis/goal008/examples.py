"""Fixed-rule Goal008 examples, with exact exposed-face meshes and no generation.

Actual inputs require a completed Phase0/Phase1 gate. Selection is independent of
proxy scores. Rendering uses absolute world coordinates and never shifts a grid.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics

RULE_ID = 'goal008_examples_openai_roles_v1'
DEFINITION_SHA256 = 'c898f6a0adbee018bbf8561bf3195aecbf1410968c619b65f6657d3bb91a4e8a'
METRIC_VERSION = 'server_v2.2_strict_translation'
PANELS = ('GT', 'O', 'J', 'P', 'E')
MAX_BYTES = 10 * 1024 * 1024
COLORS = ('#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#F0E442')
OTHER = '#757575'


class ExampleSelectionError(ValueError):
    def __init__(self, reason, audit):
        super().__init__(reason)
        self.audit = dict(audit, status='BLOCKED', reason=reason)


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'


def _gate(gate):
    if (not isinstance(gate, dict) or gate.get('status') != 'PASS'
            or gate.get('input_kind') not in {'fixture', 'verified_sources'}):
        raise ValueError('passing fixture or source gate required')
    if gate['input_kind'] == 'verified_sources' and (gate.get('phase0_status') != 'PASS' or gate.get('phase1_status') != 'PASS'):
        raise ValueError('actual examples prohibited until Phase0 and Phase1 pass')


def _definition(path):
    digest = _hash(path)
    if digest != DEFINITION_SHA256:
        raise ValueError('pre-selection example definition hash mismatch')
    return digest


def _primary(row):
    if not row or row.get('status') != 'success':
        raise ValueError('not a valid reconstruction')
    metric = row.get('metrics')
    if not isinstance(metric, dict) or metric.get('metric_version') != METRIC_VERSION:
        raise ValueError('strict metric version required')
    block = metric.get('aligned')
    if not isinstance(block, dict):
        raise ValueError('aligned metrics unavailable')
    iou, mae = block.get('occupancy_iou'), block.get('dimension_mae')
    if (type(iou) not in (int, float) or not math.isfinite(iou) or not 0 <= iou <= 1
            or type(mae) not in (int, float) or not math.isfinite(mae) or mae < 0):
        raise ValueError('finite valid aligned IoU and dimension MAE required')
    return iou, mae


def _slot_ref(row):
    if row is None:
        return None
    return {k: deepcopy(row.get(k)) for k in ('scene_id', 'provider', 'description_model', 'condition',
             'global_schedule_index', 'input_id', 'input_sha256', 'attempt_id', 'status', 'failure_kind',
             'selection_reason', 'source_plan_identity', 'source_attempt_id', 'source_entry_hash',
             'source_gt_descriptor_hash', 'source_plan_path', 'source_run', 'artifacts', 'metrics')}


def select_examples(slots, gt_sources, *, definition_path, audit_gate):
    """Return a frozen-rule candidate audit and three selections, without I/O writes.

    slots: audited Goal007 canonical slot dictionaries. gt_sources: scene-ID map
    to verified GT descriptors {path, sha256, bbox_path, bbox_sha256, verified}.
    Actual source construction/validation remains the caller's responsibility.
    """
    _gate(audit_gate)
    digest = _definition(definition_path)
    rows = list(slots)
    if audit_gate['input_kind'] == 'verified_sources':
        if len(rows) != 960 or {r['global_schedule_index'] for r in rows} != set(range(960)) or len({r['scene_id'] for r in rows}) != 60:
            raise ValueError('actual selection requires canonical960 fixed60 slots')
    by_scene = defaultdict(dict)
    for row in rows:
        c = row.get('condition')
        if row.get('provider') != 'openai' or c not in PANELS[1:]:
            continue
        if (c in ('O', 'J') and row.get('description_model') != 'openai') or (c in ('P', 'E') and row.get('description_model') is not None):
            continue
        scene = row['scene_id']
        if c in by_scene[scene]:
            raise ValueError('duplicate source slot for fixed example roles')
        by_scene[scene][c] = row
    candidates, excluded = [], []
    for scene in sorted(by_scene):
        source = gt_sources.get(scene)
        reasons = []
        if not isinstance(source, dict) or source.get('verified') is not True or not isinstance(source.get('sha256'), str) or len(source['sha256']) != 64:
            reasons.append('GT_SOURCE_NOT_VERIFIED')
        values = {}
        for c in ('O', 'J', 'P'):
            try:
                values[c] = _primary(by_scene[scene].get(c))
            except ValueError as exc:
                reasons.append(c + ': ' + str(exc))
        if reasons:
            excluded.append({'scene_id': scene, 'reasons': reasons,
                             'slots': {c: _slot_ref(row) for c, row in by_scene[scene].items()}})
            continue
        iou, mae = values['O']; p_iou, p_mae = values['P']
        candidates.append({'scene_id': scene, 'O_aligned_iou': iou, 'P_aligned_iou': p_iou,
                           'O_dimension_mae': mae, 'P_dimension_mae': p_mae,
                           'P_dimension_improvement': mae - p_mae,
                           'third_category_eligible': p_iou <= .20 and mae - p_mae > 0,
                           'gt_source': deepcopy(source),
                           'slots': {c: _slot_ref(by_scene[scene].get(c)) for c in PANELS[1:]}})
    audit = {'rule_id': RULE_ID, 'definition_sha256': digest, 'input_kind': audit_gate['input_kind'],
             'builder_provider': 'openai', 'O_J_description_provider': 'openai',
             'eligible_candidates': candidates, 'excluded_candidates': excluded,
             'eligible_count': len(candidates), 'research_api_requests_sent': 0}
    if len(candidates) < 3:
        raise ExampleSelectionError('FEWER_THAN_THREE_ELIGIBLE_BUILDINGS', audit)
    median = statistics.median(c['O_aligned_iou'] for c in candidates)
    first = min(candidates, key=lambda c: (abs(c['O_aligned_iou'] - median), c['scene_id']))
    remaining = [c for c in candidates if c['scene_id'] != first['scene_id']]
    second = min(remaining, key=lambda c: (c['O_aligned_iou'], c['scene_id']))
    remaining = [c for c in remaining if c['scene_id'] != second['scene_id'] and c['third_category_eligible']]
    audit.update(eligible_O_iou_median=median)
    if not remaining:
        raise ExampleSelectionError('NO_DISTINCT_LOW_IOU_POSITIVE_P_DIMENSION_GAIN_EXAMPLE', audit)
    third = min(remaining, key=lambda c: (-c['P_dimension_improvement'], c['scene_id']))
    audit.update(status='SELECTED', selected=[dict(deepcopy(c), selection_category=category) for c, category in zip(
        (first, second, third), ('near_median_O', 'low_O_overlap', 'P_dimension_gain_low_overlap'))])
    return audit


def description_excerpt(path, expected_sha256, *, audit_gate):
    _gate(audit_gate)
    if _hash(path) != expected_sha256:
        raise ValueError('original description source hash mismatch')
    record = json.loads(Path(path).read_text(encoding='utf-8'))
    summary = record.get('summary')
    if not isinstance(summary, str):
        raise ValueError('literal summary field unavailable; no generated replacement')
    return {'path': str(Path(path).resolve()), 'sha256': expected_sha256, 'field': 'summary',
            'excerpt': summary[:240], 'full_field_characters': len(summary), 'excerpt_characters': min(len(summary), 240),
            'truncated': len(summary) > 240, 'interpretation': 'literal prefix of original O JSON summary; no rewriting'}


def load_source_world(source, *, audit_gate):
    """Read a hash-bound canonical voxel source, never a generated replacement."""
    _gate(audit_gate)
    path = Path(source['path'])
    if _hash(path) != source['sha256']:
        raise ValueError('example voxel source hash changed')
    if source.get('kind') == 'server_v2_grid_json':
        from offline_evaluation.server_v2.core import payload_to_grid
        grid = payload_to_grid(json.loads(path.read_text(encoding='utf-8')))
    elif source.get('kind') == 'gt_numpy':
        from offline_evaluation.server_v2.data import load_grid
        bbox = Path(source['bbox_path'])
        if _hash(bbox) != source['bbox_sha256']:
            raise ValueError('GT bbox source hash changed')
        grid = load_grid(path, bbox)
    else:
        raise ValueError('explicit canonical voxel source kind required')
    return grid.as_world().voxels


def _cells(values):
    result = {}
    items = values.items() if isinstance(values, dict) else ((tuple(v[:3]), v[3]) for v in values)
    for pos, material in items:
        if len(pos) != 3 or any(type(v) is not int for v in pos):
            raise ValueError('world XYZ voxel coordinates must be integers')
        if not isinstance(material, str) or not material or material in {'air', 'minecraft:air', 'cave_air', 'void_air'}:
            raise ValueError('occupied exact material identifiers required')
        if tuple(pos) in result:
            raise ValueError('duplicate voxel coordinate')
        result[tuple(pos)] = material
    return result


def grid_description(cells):
    """Geometric metadata only; no reconstruction-quality metric computation."""
    cells = _cells(cells)
    if not cells:
        return {'occupied_voxels': 0, 'bbox_xyz': None, 'dimensions_xyz': [0, 0, 0], 'materials': {}}
    low = [min(c[i] for c in cells) for i in range(3)]
    high = [max(c[i] for c in cells) for i in range(3)]
    return {'occupied_voxels': len(cells), 'bbox_xyz': dict(zip(('xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax'), low + high)),
            'dimensions_xyz': [high[i] - low[i] + 1 for i in range(3)], 'materials': dict(sorted(Counter(cells.values()).items()))}


def exposed_rectangles(values, *, max_exposed_faces=1_500_000):
    """Greedy same-material coplanar rectangles preserve every exposed unit face.

    World XYZ is converted to display X/Z/Y only when constructing polygons.
    The sparse algorithm never creates the full enclosing grid volume.
    """
    cells = _cells(values)
    planes = defaultdict(set)
    count = 0
    for pos, material in sorted(cells.items()):
        for axis in range(3):
            uv = [i for i in range(3) if i != axis]
            for sign in (-1, 1):
                neighbor = list(pos); neighbor[axis] += sign
                if tuple(neighbor) in cells:
                    continue
                plane = pos[axis] + int(sign == 1)
                planes[(axis, sign, plane, material)].add((pos[uv[0]], pos[uv[1]]))
                count += 1
                if count > max_exposed_faces:
                    raise ValueError('EXAMPLE_SURFACE_RESOURCE_LIMIT: no cropping or downsampling applied')
    rectangles = []
    for (axis, sign, plane, material), faces in sorted(planes.items()):
        uv = [i for i in range(3) if i != axis]
        while faces:
            u, v = min(faces)
            width = 1
            while (u + width, v) in faces:
                width += 1
            height = 1
            while all((x, v + height) in faces for x in range(u, u + width)):
                height += 1
            for x in range(u, u + width):
                for y in range(v, v + height):
                    faces.remove((x, y))
            verts = []
            for a, b in ((u, v), (u + width, v), (u + width, v + height), (u, v + height)):
                point = [0, 0, 0]; point[axis] = plane; point[uv[0]] = a; point[uv[1]] = b
                verts.append([point[0], point[2], point[1]])
            rectangles.append({'axis_xyz': axis, 'normal_sign': sign, 'plane': plane, 'material': material,
                               'u': u, 'v': v, 'width': width, 'height': height, 'vertices_display_XZY': verts})
    assert sum(r['width'] * r['height'] for r in rectangles) == count
    return {'exposed_unit_faces': count, 'rectangle_count': len(rectangles), 'rectangles': rectangles}


def _palette(examples):
    counts = Counter()
    all_materials = set()
    for example in examples:
        counts.update(_cells(example['panels']['GT']['world_voxels']).values())
        for panel in example['panels'].values():
            if panel.get('world_voxels') is not None:
                all_materials.update(_cells(panel['world_voxels']).values())
    names = sorted(counts, key=lambda m: (-counts[m], m))[:7]
    colors = {name: color for name, color in zip(names, COLORS)}
    return {'exact_material_colors': {m: colors.get(m, OTHER) for m in sorted(all_materials)},
            'legend': [{'label': m, 'color': colors[m]} for m in names] +
                      ([{'label': 'Other exact materials', 'color': OTHER}] if all_materials - set(names) else []),
            'GT_exact_material_counts': dict(sorted(counts.items())), 'display_grouping_only': True}


def render_examples(examples, selection, destination, *, definition_path, audit_gate):
    """Render three source-bound example packages without reading raw responses.

    Each package: {scene_id, description, panels:{GT/O/J/P/E:{status,
    failure_kind, world_voxels, source:{path,sha256,...}, metrics, attempt_id,
    input_id,input_sha256}}, gt_bbox_source:{path,sha256,bbox}}. world_voxels is
    a sparse mapping of world XYZ tuple to exact material or [[x,y,z,material]].
    Source paths/hashes are caller-audited immutable artifact references.
    """
    _gate(audit_gate)
    digest = _definition(definition_path)
    if selection.get('status') != 'SELECTED' or selection.get('definition_sha256') != digest:
        raise ValueError('verified frozen selection required')
    if selection.get('input_kind') != audit_gate['input_kind']:
        raise ValueError('selection and rendering audit kinds differ')
    examples = list(examples)
    selected = selection['selected']
    if len(examples) != 3 or [e['scene_id'] for e in examples] != [s['scene_id'] for s in selected] or len({e['scene_id'] for e in examples}) != 3:
        raise ValueError('exactly the three selected distinct buildings required in selection order')
    for example, chosen in zip(examples, selected):
        if set(example['panels']) != set(PANELS):
            raise ValueError('GT/O/J/P/E panels required')
        for name, panel in example['panels'].items():
            valid = panel.get('status') == 'success'
            if name != 'E' and not valid:
                raise ValueError('required GT/O/J/P valid panel unavailable')
            if valid and panel.get('world_voxels') is None:
                raise ValueError('valid panel requires its observed voxel artifact')
            if not valid and (panel.get('world_voxels') is not None or panel.get('metrics') is not None):
                raise ValueError('failed/unmeasured panel cannot fabricate voxels or metrics')
            if not isinstance(panel.get('source'), dict):
                raise ValueError('every panel requires source provenance, including failed attempt')
            if name != 'GT':
                selected_slot = chosen['slots'].get(name)
                if selected_slot is not None:
                    for field in ('status', 'failure_kind', 'attempt_id', 'input_id', 'input_sha256', 'metrics'):
                        if panel.get(field) != selected_slot.get(field):
                            raise ValueError('panel differs from selected final slot: ' + field)
            if valid and audit_gate['input_kind'] == 'verified_sources':
                if name == 'GT':
                    expected_source = chosen['gt_source']
                    for key in ('path', 'sha256', 'bbox_path', 'bbox_sha256'):
                        if panel['source'].get(key) != expected_source.get(key):
                            raise ValueError('GT panel differs from selected source: ' + key)
                    bbox_source = example.get('gt_bbox_source', {})
                    if (bbox_source.get('path') != panel['source']['bbox_path']
                            or bbox_source.get('sha256') != panel['source']['bbox_sha256']
                            or bbox_source.get('bbox') != json.loads(Path(bbox_source['path']).read_text())):
                        raise ValueError('GT bbox companion provenance mismatch')
                else:
                    artifact = selected_slot.get('artifacts', {}).get('voxels.json')
                    if not isinstance(artifact, dict):
                        raise ValueError('selected successful attempt has no voxel artifact')
                    expected_path = (Path(selected_slot['source_run']) / artifact['path']).resolve()
                    if (Path(panel['source']['path']).resolve() != expected_path
                            or panel['source']['sha256'] != artifact['sha256']):
                        raise ValueError('voxel artifact does not belong to selected final attempt')
                source_world = load_source_world(panel['source'], audit_gate=audit_gate)
                if _cells(panel['world_voxels']) != source_world:
                    raise ValueError('displayed voxels differ from the hash-bound observed source')
        description = example.get('description', {})
        if not isinstance(description.get('excerpt'), str) or len(description['excerpt']) > 240 or description.get('field') != 'summary':
            raise ValueError('literal source-bound summary excerpt required')
        if audit_gate['input_kind'] == 'verified_sources':
            checked = description_excerpt(description['path'], description['sha256'], audit_gate=audit_gate)
            if checked != description:
                raise ValueError('literal original summary excerpt mismatch')
    target = Path(destination)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError('fresh empty example output directory required')
    target.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str((target / '.matplotlib-cache').resolve())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.ticker import MaxNLocator
    # The 22-inch canvas is included at approximately 7 inches in the paper:
    # 21-point canvas type therefore remains approximately 7-point body type.
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 23, 'svg.fonttype': 'none', 'svg.hashsalt': RULE_ID})
    palette = _palette(examples)
    details, artifacts = [], []
    for index, (example, chosen) in enumerate(zip(examples, selected), 1):
        worlds = {name: _cells(panel['world_voxels']) if panel.get('world_voxels') is not None else None for name, panel in example['panels'].items()}
        positions = [pos for world in worlds.values() if world is not None for pos in world]
        if not positions:
            raise ValueError('GT occupied geometry required; no invented empty example')
        bounds = [[min(p[i] for p in positions) - 1, max(p[i] for p in positions) + 2] for i in range(3)]
        fig = plt.figure(figsize=(22, 6.2))
        fig.subplots_adjust(left=.04, right=.95, bottom=.30, top=.76, wspace=.13)
        panel_records = {}
        for number, name in enumerate(PANELS, 1):
            panel, world = example['panels'][name], worlds[name]
            ax = fig.add_subplot(1, 5, number, projection='3d', proj_type='ortho')
            ax.view_init(elev=25, azim=-60)
            ax.set(xlim=bounds[0], ylim=bounds[2], zlim=bounds[1], xlabel='X', ylabel='Z', zlabel='Y')
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.labelpad = 14
            ax.set_box_aspect([bounds[i][1] - bounds[i][0] for i in (0, 2, 1)])
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_major_locator(MaxNLocator(nbins=3, integer=True))
                axis.set_tick_params(labelsize=21, pad=0)
            source = {k: deepcopy(v) for k, v in panel.items() if k != 'world_voxels'}
            if world is None:
                status = panel.get('failure_kind') or panel['status']
                ax.text2D(.5, .5, status + '\nNo observed\nreconstruction', transform=ax.transAxes, ha='center', va='center', color='#9C2F21', fontsize=22)
                source.update(geometry=None, exposed_unit_faces=None, mesh_rectangles=None)
                title = name + ' (' + status + ')'
            else:
                mesh = exposed_rectangles(world)
                polygons = [r['vertices_display_XZY'] for r in mesh['rectangles']]
                colors = [palette['exact_material_colors'][r['material']] for r in mesh['rectangles']]
                ax.add_collection3d(Poly3DCollection(polygons, facecolors=colors, edgecolors=(0, 0, 0, .12), linewidths=.15, zsort='average'))
                source.update(geometry=grid_description(world), exposed_unit_faces=mesh['exposed_unit_faces'], mesh_rectangles=mesh['rectangle_count'])
                title = 'GT (ground truth)' if name == 'GT' else name
                if not world:
                    ax.text2D(.5, .5, 'Valid empty output', transform=ax.transAxes, ha='center')
            ax.set_title(title, fontsize=24, pad=6)
            panel_records[name] = source
        heading = (('SYNTHETIC FIXTURE — NOT RESEARCH RESULTS | ' + example['scene_id'])
                   if audit_gate['input_kind'] == 'fixture'
                   else example['scene_id'] + ' | ' + chosen['selection_category'].replace('_', ' '))
        fig.suptitle(heading + '\nBuilder: OpenAI; O/J description provider: OpenAI', fontsize=25, y=.98)
        fig.text(.5, .19, 'Absolute XYZ (voxels); fixed-rule illustrations, not population-average estimates.', ha='center', fontsize=22)
        handles = [Patch(facecolor=x['color'], edgecolor='black', linewidth=.3, label=x['label']) for x in palette['legend']]
        fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(.5, -.01), fontsize=21, frameon=False)
        base = f'example_{index:02d}'
        for ext in ('png', 'svg'):
            buf = io.BytesIO()
            # A fixed full canvas retains projected 3D axis labels that tight
            # bounding-box calculation can omit at the right edge.
            fig.savefig(buf, format=ext, dpi=300, metadata={'Software': RULE_ID} if ext == 'png' else {'Date': None, 'Creator': RULE_ID})
            data = buf.getvalue()
            if len(data) > MAX_BYTES:
                plt.close(fig)
                raise ValueError('EXAMPLE_EXPORT_EXCEEDS_10MIB: no geometry removed')
            with (target / (base + '.' + ext)).open('xb') as f:
                f.write(data)
            artifacts.append({'path': base + '.' + ext, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
        plt.close(fig)
        details.append({'scene_id': example['scene_id'], 'selection_category': chosen['selection_category'],
            'selection_rank_values': {k: chosen[k] for k in ('O_aligned_iou', 'P_aligned_iou', 'O_dimension_mae', 'P_dimension_mae', 'P_dimension_improvement')},
            'description': example['description'], 'gt_bbox_source': example.get('gt_bbox_source'),
            'bounds_world_xyz': bounds, 'camera': {'projection': 'orthographic', 'elevation': 25, 'azimuth': -60, 'display_axis_order': 'XZY', 'alignment_shift_applied': False},
            'panels': panel_records})
    table = []
    for detail in details:
        for name, panel in detail['panels'].items():
            table.append({'scene_id': detail['scene_id'], 'selection_category': detail['selection_category'], 'condition': name,
                          'status': panel['status'], 'failure_kind': panel.get('failure_kind'), 'source': panel['source'],
                          'attempt_id': panel.get('attempt_id'), 'input_id': panel.get('input_id'), 'input_sha256': panel.get('input_sha256'),
                          'geometry': panel['geometry'], 'metrics': panel.get('metrics'),
                          'description_sha256': detail['description']['sha256'], 'literal_summary_excerpt': detail['description']['excerpt'],
                          'scope_note': 'Illustrative examples selected by a fixed rule; these examples do not estimate population-average performance.'})
    handle = io.StringIO(newline=''); writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator='\n'); writer.writeheader()
    for row in table:
        writer.writerow({k: json.dumps(v, sort_keys=True, ensure_ascii=False, allow_nan=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    tex = r'''\begin{figure*}[p]
\centering
\includegraphics[width=\textwidth]{example_01.png}\\[2pt]
\includegraphics[width=\textwidth]{example_02.png}\\[2pt]
\includegraphics[width=\textwidth]{example_03.png}
\caption{Three distinct buildings selected by the fixed near-median O, low-O-overlap,
and P-dimension-gain with P IoU at most 0.20 rules, respectively.
All panels use unmodified absolute world coordinates, a common orthographic camera,
and identical axes and voxel scale within each row. The stored strict-aligned metrics
were used for selection only. O is the saved JSON-form visual description record;
J preserves that JSON tree; P/E supply partial/full GT information.
Failed E outputs remain explicitly marked, with no invented geometry.
Colors group the seven most frequent exact GT materials and other materials for display
only; exact identities and metrics remain in the source table.
These illustrative examples do not estimate population-average performance.
Source hashes, literal description excerpts, metrics and per-panel provenance appear in
the accompanying example table and manifest.}
\label{fig:goal008-reconstruction-examples}
\end{figure*}
'''
    documents = {'example_selection.json': _json(selection), 'example_sources.json': _json(details),
                 'example_palette.json': _json(palette), 'example_companion.csv': handle.getvalue(),
                 'examples_figure.tex': tex}
    for name, content in documents.items():
        data = content.encode('utf-8')
        if len(data) > MAX_BYTES:
            raise ValueError('EXAMPLE_METADATA_EXCEEDS_10MIB')
        with (target / name).open('xb') as f:f.write(data)
        artifacts.append({'path': name, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
    manifest = {'status': 'RENDERED', 'rule_id': RULE_ID, 'definition_sha256': digest, 'input_kind': audit_gate['input_kind'],
        'coordinate_policy': 'unmodified absolute world XYZ; no alignment shift, crop or per-panel rescaling',
        'mesh_policy': 'exposed unit faces merged only across coplanar same-exact-material cells',
        'examples': [e['scene_id'] for e in examples], 'files': artifacts, 'research_api_requests_sent': 0,
        'model_generated_images': 0, 'reconstruction_metrics_recomputed': False, 'matplotlib_version': matplotlib.__version__}
    manifest['publication_layout'] = {'canvas_inches': [22, 6.2], 'intended_paper_width_inches': 7,
        'minimum_canvas_font_points': 21, 'typography_revision_only': True,
        'full_disclaimer': 'Illustrative examples selected by a fixed rule; these examples do not estimate population-average performance.',
        'selection_metric_note': 'Selection used stored strict-aligned metrics; display uses unmodified absolute world coordinates.'}
    with (target / 'example_manifest.json').open('x', encoding='utf-8') as f:f.write(_json(manifest))
    return manifest
