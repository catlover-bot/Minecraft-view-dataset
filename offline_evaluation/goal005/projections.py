"""CPU-only, deterministic orthographic views of saved measured voxel grids."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from offline_evaluation.server_v2.core import payload_to_grid
from offline_evaluation.server_v2.data import load_scene
from offline_evaluation.server_v2.generation import digest
from offline_evaluation.resmoke_v2.plan import load_plan
from offline_evaluation.resmoke_v2.runner import _artifact, _load_enabled


def render(plan_path, destination):
    plan = load_plan(plan_path)
    directory, ledger = _load_enabled(plan)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=13)
    rows = []
    views = [('top X/Z', 0, 2, 1), ('front X/Y', 0, 1, 2), ('side Z/Y', 2, 1, 0)]
    for scene, descriptor in plan['identity']['gt_by_scene'].items():
        gt = load_scene(Path(descriptor['building_directory'])).grid.as_world().voxels
        grids = [('GT', gt)]
        for record, entry in zip(plan['identity']['schedule'], ledger['entries']):
            if record['scene_id'] != scene:
                continue
            label = record['provider'] + ' ' + record['condition']
            if entry['status'] == 'success':
                path = _artifact(directory, entry['artifacts']['voxels.json'])
                voxels = payload_to_grid(json.loads(path.read_text())).as_world().voxels
                grids.append((label, voxels))
                rows.append({'scene_id': scene, 'provider': record['provider'], 'condition': record['condition'], 'status': 'rendered_actual_voxels', 'voxels_sha256': digest(path.read_bytes())})
            else:
                grids.append((label + ' MISSING', None))
                rows.append({'scene_id': scene, 'provider': record['provider'], 'condition': record['condition'], 'status': 'missing_' + entry['failure_kind'], 'voxels_sha256': None})
        coordinates = [point for _, grid in grids if grid is not None for point in grid]
        lo = [min(p[a] for p in coordinates) for a in range(3)]
        hi = [max(p[a] for p in coordinates) for a in range(3)]
        materials = sorted({m for _, grid in grids if grid is not None for m in grid.values()})
        colors = {m: tuple(45 + x * 180 // 255 for x in hashlib.sha256(m.encode()).digest()[:3]) for m in materials}
        row_h, cell_w, left, top = 190, 310, 155, 90
        legend_rows = (len(materials) + 3) // 4
        image = Image.new('RGB', (1100, top + row_h * len(grids) + 40 + 25 * legend_rows), 'white')
        draw = ImageDraw.Draw(image)
        draw.text((12, 10), scene + ': actual saved voxels, absolute world coordinates', fill='black', font=font)
        draw.text((12, 34), 'Nearest visible material (maximum depth). Shared extent across rows; no alignment or GT repair.', fill='black', font=small)
        for col, (label, _, _, _) in enumerate(views):
            draw.text((left + col * cell_w, 62), label, fill='black', font=font)
        for i, (label, grid) in enumerate(grids):
            y0 = top + i * row_h
            draw.text((8, y0 + 12), label, fill='black', font=small)
            if grid is None:
                draw.text((left, y0 + 60), 'MISSING: validation failure; no reconstruction', fill='darkred', font=font)
                continue
            for col, (_, a, b, depth) in enumerate(views):
                projected = Image.new('RGB', (hi[a] - lo[a] + 1, hi[b] - lo[b] + 1), '#f0f0f0')
                for point, material in sorted(grid.items(), key=lambda item: item[0][depth]):
                    projected.putpixel((point[a] - lo[a], hi[b] - point[b]), colors[material])
                projected.thumbnail((cell_w - 16, row_h - 20), Image.Resampling.NEAREST)
                factor = min((cell_w - 16) / projected.width, (row_h - 20) / projected.height)
                projected = projected.resize((max(1, int(projected.width * factor)), max(1, int(projected.height * factor))), Image.Resampling.NEAREST)
                image.paste(projected, (left + col * cell_w, y0))
        base = top + row_h * len(grids) + 8
        for index, material in enumerate(materials):
            x, y = (index % 4) * 270 + 12, base + (index // 4) * 25
            draw.rectangle((x, y, x + 14, y + 14), fill=colors[material])
            draw.text((x + 20, y), material, fill='black', font=small)
        image.save(destination / (scene + '.png'))
    report = {'research_api_additional_sends': 0, 'source': 'actual frozen GT and saved successful model voxel artifacts; missing outputs explicitly marked', 'projection': 'orthographic maximum-depth material, shared absolute extent per scene', 'rows': rows}
    (destination / 'projections.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    result = render(args.plan, args.destination)
    print(json.dumps({'rows': len(result['rows']), 'rendered': sum(r['status'] == 'rendered_actual_voxels' for r in result['rows']), 'missing': sum(r['status'].startswith('missing_') for r in result['rows']), 'research_api_additional_sends': 0}))
