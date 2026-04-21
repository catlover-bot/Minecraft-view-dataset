from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .generator import BuildingSpec

Block = Tuple[int, int, int, str]
Cell2D = Tuple[int, int]


DEFAULT_ROLE_BLOCKS: Dict[str, str] = {
    "foundation": "stonebrick",
    "wall": "planks",
    "roof": "nether_brick",
    "window": "glass",
    "accent": "quartz_block",
    "trim": "stone_slab",
    "floor": "planks",
    "light": "glowstone",
    "pillar": "stonebrick",
}


@dataclass(frozen=True)
class LLMAuthoredCase:
    case_id: str
    title: str
    difficulty: str
    author_provider: str
    author_model: str
    source_condition: str
    spec: Dict[str, Any]


def _safe_int(v: Any, default: int) -> int:
    try:
        return int(round(float(v)))
    except Exception:
        return default


def _bbox_from_blocks(blocks: List[Block]) -> Dict[str, int]:
    xs = [b[0] for b in blocks]
    ys = [b[1] for b in blocks]
    zs = [b[2] for b in blocks]
    return {
        "xmin": min(xs),
        "xmax": max(xs),
        "ymin": min(ys),
        "ymax": max(ys),
        "zmin": min(zs),
        "zmax": max(zs),
    }


def _neighbors4(cell: Cell2D) -> Tuple[Cell2D, Cell2D, Cell2D, Cell2D]:
    x, z = cell
    return ((x + 1, z), (x - 1, z), (x, z + 1), (x, z - 1))


def _rect_cells(x0: int, x1: int, z0: int, z1: int) -> Set[Cell2D]:
    return {(x, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1)}


def _normalize_cells(cells: Set[Cell2D]) -> Set[Cell2D]:
    if not cells:
        return set()
    min_x = min(x for x, _ in cells)
    min_z = min(z for _, z in cells)
    return {(x - min_x, z - min_z) for x, z in cells}


def _bbox_2d(cells: Set[Cell2D]) -> Tuple[int, int, int, int]:
    xs = [x for x, _ in cells]
    zs = [z for _, z in cells]
    return min(xs), max(xs), min(zs), max(zs)


def _footprint_cells(spec: Dict[str, Any]) -> Set[Cell2D]:
    f = spec.get("footprint", {}) if isinstance(spec.get("footprint"), dict) else {}
    kind = str(f.get("kind", "rectangle")).strip().lower()
    width = max(8, _safe_int(spec.get("width", 14), 14))
    depth = max(8, _safe_int(spec.get("depth", 12), 12))

    cells = _rect_cells(0, width - 1, 0, depth - 1)
    if kind == "rectangle":
        return cells

    if kind == "l_shape":
        notch_w = max(3, min(width - 3, _safe_int(f.get("notch_width", width // 2), width // 2)))
        notch_d = max(3, min(depth - 3, _safe_int(f.get("notch_depth", depth // 2), depth // 2)))
        corner = str(f.get("corner", "nw")).strip().lower()
        for x in range(notch_w):
            for z in range(notch_d):
                if corner == "nw":
                    cells.discard((x, z))
                elif corner == "ne":
                    cells.discard((width - 1 - x, z))
                elif corner == "sw":
                    cells.discard((x, depth - 1 - z))
                else:
                    cells.discard((width - 1 - x, depth - 1 - z))
        return _normalize_cells(cells)

    if kind == "u_shape":
        opening = str(f.get("opening", "south")).strip().lower()
        gap_w = max(3, min(width - 4, _safe_int(f.get("gap_width", width // 2), width // 2)))
        gap_start = max(1, min(width - gap_w - 1, (width - gap_w) // 2))
        thickness = max(2, min(6, _safe_int(f.get("thickness", 4), 4)))
        if opening == "south":
            cut = _rect_cells(gap_start, gap_start + gap_w - 1, 0, depth - thickness - 1)
        elif opening == "north":
            cut = _rect_cells(gap_start, gap_start + gap_w - 1, thickness, depth - 1)
        elif opening == "west":
            cut = _rect_cells(0, width - thickness - 1, gap_start, gap_start + gap_w - 1)
        else:
            cut = _rect_cells(thickness, width - 1, gap_start, gap_start + gap_w - 1)
        cells.difference_update(cut)
        return _normalize_cells(cells)

    if kind == "plus":
        arm_w = max(3, min(width, _safe_int(f.get("arm_width", width // 3), width // 3)))
        arm_d = max(3, min(depth, _safe_int(f.get("arm_depth", depth // 3), depth // 3)))
        cx = (width - 1) // 2
        cz = (depth - 1) // 2
        out: Set[Cell2D] = set()
        for x in range(width):
            for z in range(depth):
                if abs(x - cx) <= arm_w // 2 or abs(z - cz) <= arm_d // 2:
                    out.add((x, z))
        return _normalize_cells(out)

    if kind == "ring":
        t = max(2, min(5, _safe_int(f.get("thickness", 3), 3)))
        out: Set[Cell2D] = set()
        for x in range(width):
            for z in range(depth):
                if x < t or x >= width - t or z < t or z >= depth - t:
                    out.add((x, z))
        return _normalize_cells(out)

    return cells


def _role_blocks(spec: Dict[str, Any]) -> Dict[str, str]:
    mats = spec.get("materials", {}) if isinstance(spec.get("materials"), dict) else {}
    out = dict(DEFAULT_ROLE_BLOCKS)
    for k in out.keys():
        v = mats.get(k)
        if isinstance(v, str) and v.strip():
            out[k] = v.strip().lower().replace("minecraft:", "")
    return out


def _pick_door_cell(cells: Set[Cell2D], side: str) -> Tuple[Cell2D, Tuple[int, int]]:
    xmin, xmax, zmin, zmax = _bbox_2d(cells)
    boundary = {c for c in cells if any(n not in cells for n in _neighbors4(c))}
    if not boundary:
        c = ((xmin + xmax) // 2, (zmin + zmax) // 2)
        return c, (0, -1)

    side = side.lower()
    if side == "south":
        cand = sorted([c for c in boundary if c[1] == zmin], key=lambda p: abs(p[0] - (xmin + xmax) / 2.0))
        if cand:
            return cand[0], (0, -1)
    if side == "north":
        cand = sorted([c for c in boundary if c[1] == zmax], key=lambda p: abs(p[0] - (xmin + xmax) / 2.0))
        if cand:
            return cand[0], (0, 1)
    if side == "west":
        cand = sorted([c for c in boundary if c[0] == xmin], key=lambda p: abs(p[1] - (zmin + zmax) / 2.0))
        if cand:
            return cand[0], (-1, 0)
    if side == "east":
        cand = sorted([c for c in boundary if c[0] == xmax], key=lambda p: abs(p[1] - (zmin + zmax) / 2.0))
        if cand:
            return cand[0], (1, 0)

    c = sorted(boundary, key=lambda p: abs(p[0] - (xmin + xmax) / 2.0) + abs(p[1] - (zmin + zmax) / 2.0))[0]
    return c, (0, -1)


def build_spec_to_building(case: LLMAuthoredCase, origin: Tuple[int, int, int] = (0, 4, 0), style_id: int = 0) -> BuildingSpec:
    spec = case.spec
    cells = _footprint_cells(spec)
    cells = _normalize_cells(cells)
    if not cells:
        raise ValueError(f"Empty footprint for case={case.case_id}")

    role = _role_blocks(spec)
    x0, y0, z0 = origin

    floors = max(1, _safe_int(spec.get("floors", 1), 1))
    floor_h = max(3, _safe_int(spec.get("floor_height", 4), 4))
    roof = spec.get("roof", {}) if isinstance(spec.get("roof"), dict) else {}
    roof_type = str(roof.get("type", "flat")).strip().lower()
    roof_h = max(1, _safe_int(roof.get("height", 2), 2))

    door_obj = spec.get("entrance", {}) if isinstance(spec.get("entrance"), dict) else {}
    door_side = str(door_obj.get("side", "south")).strip().lower()
    door_w = max(1, min(3, _safe_int(door_obj.get("width", 1), 1)))
    door_h = max(2, min(4, _safe_int(door_obj.get("height", 2), 2)))

    win_obj = spec.get("windows", {}) if isinstance(spec.get("windows"), dict) else {}
    win_pattern = str(win_obj.get("pattern", "checker")).strip().lower()
    win_spacing = max(2, _safe_int(win_obj.get("spacing", 3), 3))
    win_h = max(1, min(3, _safe_int(win_obj.get("height", 2), 2)))

    features = spec.get("features", {}) if isinstance(spec.get("features"), dict) else {}

    block_map: Dict[Tuple[int, int, int], str] = {}

    def set_block(x: int, y: int, z: int, block: str) -> None:
        key = (x, y, z)
        if block == "air":
            block_map.pop(key, None)
            return
        block_map[key] = block

    boundary = {c for c in cells if any(n not in cells for n in _neighbors4(c))}
    corners = {c for c in boundary if sum(1 for n in _neighbors4(c) if n not in cells) >= 2}

    # Foundation and floors.
    for cx, cz in cells:
        wx, wz = x0 + cx, z0 + cz
        set_block(wx, y0 - 1, wz, role["foundation"])

    for f in range(floors):
        fy = y0 + f * floor_h
        for cx, cz in cells:
            wx, wz = x0 + cx, z0 + cz
            set_block(wx, fy, wz, role["floor"])
            for yy in range(fy + 1, fy + floor_h):
                if (cx, cz) not in boundary:
                    set_block(wx, yy, wz, "air")
                    continue
                if (cx, cz) in corners:
                    set_block(wx, yy, wz, role["pillar"])
                    continue

                on_window_band = fy + 2 <= yy <= fy + 2 + (win_h - 1)
                if on_window_band:
                    if win_pattern == "stripe_x":
                        is_window = (cx % win_spacing) != 0
                    elif win_pattern == "stripe_z":
                        is_window = (cz % win_spacing) != 0
                    else:
                        is_window = ((cx + cz + f) % win_spacing) != 0
                    set_block(wx, yy, wz, role["window"] if is_window else role["wall"])
                else:
                    set_block(wx, yy, wz, role["wall"])

        top_y = fy + floor_h - 1
        for cx, cz in boundary:
            set_block(x0 + cx, top_y, z0 + cz, role["trim"])

    # Entrance.
    door_cell, door_vec = _pick_door_cell(cells, door_side)
    dxc, dzc = door_cell
    side_cells = [(dxc, dzc)]
    if door_w == 2:
        if abs(door_vec[0]) == 1:
            side_cells.append((dxc, dzc + 1))
        else:
            side_cells.append((dxc + 1, dzc))
    if door_w >= 3:
        if abs(door_vec[0]) == 1:
            side_cells.extend([(dxc, dzc - 1), (dxc, dzc + 1)])
        else:
            side_cells.extend([(dxc - 1, dzc), (dxc + 1, dzc)])

    for ccx, ccz in side_cells:
        if (ccx, ccz) not in boundary:
            continue
        for yy in range(y0 + 1, y0 + 1 + door_h):
            set_block(x0 + ccx, yy, z0 + ccz, "air")
    for ccx, ccz in side_cells:
        ox, oz = ccx + door_vec[0], ccz + door_vec[1]
        set_block(x0 + ox, y0, z0 + oz, role["accent"])

    # Roof.
    roof_y = y0 + floors * floor_h
    xmin, xmax, zmin, zmax = _bbox_2d(cells)

    if roof_type == "flat":
        for cx, cz in cells:
            set_block(x0 + cx, roof_y, z0 + cz, role["roof"])
        for cx, cz in boundary:
            set_block(x0 + cx, roof_y + 1, z0 + cz, role["trim"])
    elif roof_type == "gable_x":
        mid = (xmin + xmax) / 2.0
        for cx, cz in cells:
            h = max(0, min(roof_h, int(round((roof_h + 0.2) * (1.0 - abs(cx - mid) / max(1.0, (xmax - xmin) / 2.0))))))
            for dy in range(h + 1):
                set_block(x0 + cx, roof_y + dy, z0 + cz, role["roof"])
    elif roof_type == "gable_z":
        mid = (zmin + zmax) / 2.0
        for cx, cz in cells:
            h = max(0, min(roof_h, int(round((roof_h + 0.2) * (1.0 - abs(cz - mid) / max(1.0, (zmax - zmin) / 2.0))))))
            for dy in range(h + 1):
                set_block(x0 + cx, roof_y + dy, z0 + cz, role["roof"])
    else:  # hip as fallback
        for cx, cz in cells:
            dist = min(cx - xmin, xmax - cx, cz - zmin, zmax - cz)
            h = max(0, min(roof_h, dist))
            for dy in range(h + 1):
                set_block(x0 + cx, roof_y + dy, z0 + cz, role["roof"])

    # Optional substructures.
    if bool(features.get("tower")):
        tx0, tz0 = xmin, zmax - 3
        for cx in range(tx0, tx0 + 4):
            for cz in range(tz0, tz0 + 4):
                if (cx, cz) not in cells:
                    continue
                for yy in range(roof_y, roof_y + floor_h + 2):
                    on_edge = cx in (tx0, tx0 + 3) or cz in (tz0, tz0 + 3)
                    set_block(x0 + cx, yy, z0 + cz, role["pillar"] if on_edge else "air")
                set_block(x0 + cx, roof_y + floor_h + 2, z0 + cz, role["roof"])

    if bool(features.get("porch")):
        for ccx, ccz in side_cells:
            px, pz = ccx + door_vec[0], ccz + door_vec[1]
            for sx in range(-1, 2):
                for sz in range(-1, 2):
                    if abs(door_vec[0]) == 1:
                        tx, tz = px + door_vec[0] * sx, pz + sz
                    else:
                        tx, tz = px + sx, pz + door_vec[1] * sz
                    set_block(x0 + tx, y0, z0 + tz, role["floor"])
                    if abs(sx) == 1 or abs(sz) == 1:
                        set_block(x0 + tx, y0 + 1, z0 + tz, role["trim"])

    if bool(features.get("balcony")) and floors >= 2:
        by = y0 + floor_h + 1
        for ccx, ccz in side_cells:
            bx, bz = ccx + door_vec[0], ccz + door_vec[1]
            for step in range(0, 3):
                tx, tz = bx + door_vec[0] * step, bz + door_vec[1] * step
                set_block(x0 + tx, by, z0 + tz, role["floor"])
                set_block(x0 + tx, by + 1, z0 + tz, role["trim"])

    for cx, cz in corners:
        set_block(x0 + cx, y0 + 2, z0 + cz, role["light"])

    blocks = [
        (x, y, z, block)
        for (x, y, z), block in sorted(block_map.items(), key=lambda item: (item[0][1], item[0][0], item[0][2]))
    ]
    if not blocks:
        raise ValueError(f"No blocks generated for case={case.case_id}")

    palette_name = role.get("wall", "custom") + "_" + role.get("roof", "custom")
    return BuildingSpec(
        style=case.title,
        palette_name=palette_name,
        origin=origin,
        bbox=_bbox_from_blocks(blocks),
        blocks=blocks,
        style_id=style_id,
    )
