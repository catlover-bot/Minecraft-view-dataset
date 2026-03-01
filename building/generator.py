from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .palette import choose_palette

Block = Tuple[int, int, int, str]
Cell2D = Tuple[int, int]


@dataclass(frozen=True)
class BuildingSpec:
    style: str
    palette_name: str
    origin: Tuple[int, int, int]
    bbox: Dict[str, int]
    blocks: List[Block]
    style_id: int


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


def _boundary_cells(cells: Set[Cell2D]) -> Set[Cell2D]:
    return {c for c in cells if any(n not in cells for n in _neighbors4(c))}


def _corner_cells(cells: Set[Cell2D]) -> Set[Cell2D]:
    corners: Set[Cell2D] = set()
    for cell in _boundary_cells(cells):
        missing = sum(1 for n in _neighbors4(cell) if n not in cells)
        if missing >= 2:
            corners.add(cell)
    return corners


def _shrink_cells(cells: Set[Cell2D]) -> Set[Cell2D]:
    return {c for c in cells if all(n in cells for n in _neighbors4(c))}


def _connected_components(cells: Set[Cell2D]) -> List[Set[Cell2D]]:
    remaining = set(cells)
    components: List[Set[Cell2D]] = []
    while remaining:
        start = next(iter(remaining))
        comp: Set[Cell2D] = set()
        queue: deque[Cell2D] = deque([start])
        remaining.remove(start)
        while queue:
            cur = queue.popleft()
            comp.add(cur)
            for nxt in _neighbors4(cur):
                if nxt in remaining:
                    remaining.remove(nxt)
                    queue.append(nxt)
        components.append(comp)
    return components


def _void_cells(cells: Set[Cell2D]) -> Set[Cell2D]:
    if not cells:
        return set()
    xmin, xmax, zmin, zmax = _bbox_2d(cells)
    sx0, sx1 = xmin - 1, xmax + 1
    sz0, sz1 = zmin - 1, zmax + 1

    outside: Set[Cell2D] = set()
    queue: deque[Cell2D] = deque([(sx0, sz0)])
    outside.add((sx0, sz0))

    while queue:
        x, z = queue.popleft()
        for nx, nz in _neighbors4((x, z)):
            if nx < sx0 or nx > sx1 or nz < sz0 or nz > sz1:
                continue
            if (nx, nz) in cells or (nx, nz) in outside:
                continue
            outside.add((nx, nz))
            queue.append((nx, nz))

    voids: Set[Cell2D] = set()
    for x in range(xmin, xmax + 1):
        for z in range(zmin, zmax + 1):
            if (x, z) in cells:
                continue
            if (x, z) not in outside:
                voids.add((x, z))
    return voids


def _footprint_rectangle(style_rng: random.Random) -> Tuple[Set[Cell2D], str]:
    width = style_rng.randint(18, 30)
    depth = style_rng.randint(16, 28)
    return _rect_cells(0, width - 1, 0, depth - 1), "rectangle"


def _footprint_l_shape(style_rng: random.Random) -> Tuple[Set[Cell2D], str]:
    width = style_rng.randint(24, 36)
    depth = style_rng.randint(22, 34)
    cells = _rect_cells(0, width - 1, 0, depth - 1)

    notch_w = min(width - 6, style_rng.randint(max(6, width // 3), max(7, width // 2)))
    notch_d = min(depth - 6, style_rng.randint(max(6, depth // 3), max(7, depth // 2)))
    quadrant = style_rng.randrange(4)

    for x in range(notch_w):
        for z in range(notch_d):
            if quadrant == 0:
                cells.discard((x, z))
            elif quadrant == 1:
                cells.discard((width - 1 - x, z))
            elif quadrant == 2:
                cells.discard((x, depth - 1 - z))
            else:
                cells.discard((width - 1 - x, depth - 1 - z))

    return _normalize_cells(cells), "l_shape"


def _footprint_u_shape(style_rng: random.Random) -> Tuple[Set[Cell2D], str]:
    width = style_rng.randint(24, 36)
    depth = style_rng.randint(20, 32)
    thickness = style_rng.randint(4, 7)
    cells = _rect_cells(0, width - 1, 0, depth - 1)

    gap_w = min(width - 2 * thickness - 2, style_rng.randint(max(6, width // 3), max(8, width // 2)))
    gap_w = max(4, gap_w)
    start_x = (width - gap_w) // 2
    open_side = style_rng.randrange(4)

    if open_side == 0:
        cut = _rect_cells(start_x, start_x + gap_w - 1, 0, depth - thickness - 1)
    elif open_side == 1:
        cut = _rect_cells(start_x, start_x + gap_w - 1, thickness, depth - 1)
    elif open_side == 2:
        cut = _rect_cells(0, width - thickness - 1, start_x, start_x + gap_w - 1)
    else:
        cut = _rect_cells(thickness, width - 1, start_x, start_x + gap_w - 1)

    cells.difference_update(cut)
    return _normalize_cells(cells), "u_shape"


def _footprint_ring(style_rng: random.Random) -> Tuple[Set[Cell2D], str]:
    width = style_rng.randint(22, 36)
    depth = style_rng.randint(22, 36)
    thick = style_rng.randint(3, 5)
    cells: Set[Cell2D] = set()
    for x in range(width):
        for z in range(depth):
            if x < thick or x >= width - thick or z < thick or z >= depth - thick:
                cells.add((x, z))
    return cells, "ring"


def _footprint_plus(style_rng: random.Random) -> Tuple[Set[Cell2D], str]:
    width = style_rng.randint(25, 37)
    depth = style_rng.randint(25, 37)
    cx = (width - 1) // 2
    cz = (depth - 1) // 2
    arm_x = style_rng.randint(6, 10)
    arm_z = style_rng.randint(6, 10)

    cells: Set[Cell2D] = set()
    for x in range(width):
        for z in range(depth):
            if abs(x - cx) <= arm_x // 2 or abs(z - cz) <= arm_z // 2:
                cells.add((x, z))
    return cells, "plus"


def _footprint_h(style_rng: random.Random) -> Tuple[Set[Cell2D], str]:
    width = style_rng.randint(26, 38)
    depth = style_rng.randint(18, 30)
    leg_w = style_rng.randint(5, 7)
    bridge_h = style_rng.randint(4, 8)

    cells: Set[Cell2D] = set()
    cells.update(_rect_cells(0, leg_w - 1, 0, depth - 1))
    cells.update(_rect_cells(width - leg_w, width - 1, 0, depth - 1))

    z_mid = (depth - 1) // 2
    z0 = max(0, z_mid - bridge_h // 2)
    z1 = min(depth - 1, z0 + bridge_h)
    cells.update(_rect_cells(leg_w - 1, width - leg_w, z0, z1))

    return _normalize_cells(cells), "h_shape"


def _footprint_compound(style_rng: random.Random) -> Tuple[Set[Cell2D], str]:
    main_w = style_rng.randint(18, 28)
    main_d = style_rng.randint(16, 26)
    wing_w = style_rng.randint(10, 16)
    wing_d = style_rng.randint(10, 16)
    side = style_rng.choice(("west", "east", "north", "south"))

    cells = _rect_cells(0, main_w - 1, 0, main_d - 1)
    if side == "west":
        z0 = style_rng.randint(1, max(1, main_d - wing_d - 1))
        wing = _rect_cells(-wing_w + 2, 1, z0, z0 + wing_d - 1)
        connector = _rect_cells(1, 3, z0 + wing_d // 2 - 1, z0 + wing_d // 2 + 1)
    elif side == "east":
        z0 = style_rng.randint(1, max(1, main_d - wing_d - 1))
        wing = _rect_cells(main_w - 2, main_w + wing_w - 3, z0, z0 + wing_d - 1)
        connector = _rect_cells(main_w - 4, main_w - 2, z0 + wing_d // 2 - 1, z0 + wing_d // 2 + 1)
    elif side == "north":
        x0 = style_rng.randint(1, max(1, main_w - wing_w - 1))
        wing = _rect_cells(x0, x0 + wing_w - 1, -wing_d + 2, 1)
        connector = _rect_cells(x0 + wing_w // 2 - 1, x0 + wing_w // 2 + 1, 1, 3)
    else:
        x0 = style_rng.randint(1, max(1, main_w - wing_w - 1))
        wing = _rect_cells(x0, x0 + wing_w - 1, main_d - 2, main_d + wing_d - 3)
        connector = _rect_cells(x0 + wing_w // 2 - 1, x0 + wing_w // 2 + 1, main_d - 4, main_d - 2)

    cells.update(wing)
    cells.update(connector)
    return _normalize_cells(cells), "compound"


def _footprint_octagon(style_rng: random.Random) -> Tuple[Set[Cell2D], str]:
    width = style_rng.randint(24, 36)
    depth = style_rng.randint(24, 36)
    cut = style_rng.randint(4, 7)
    cells: Set[Cell2D] = set()

    for x in range(width):
        for z in range(depth):
            if x + z < cut:
                continue
            if (width - 1 - x) + z < cut:
                continue
            if x + (depth - 1 - z) < cut:
                continue
            if (width - 1 - x) + (depth - 1 - z) < cut:
                continue
            cells.add((x, z))

    return cells, "octagon"


def _footprint_star(style_rng: random.Random) -> Tuple[Set[Cell2D], str]:
    width = style_rng.randint(28, 40)
    depth = style_rng.randint(28, 40)
    core_w = style_rng.randint(10, 16)
    core_d = style_rng.randint(10, 16)
    cx = (width - 1) // 2
    cz = (depth - 1) // 2

    cells: Set[Cell2D] = set()
    cells.update(_rect_cells(cx - core_w // 2, cx + core_w // 2, cz - core_d // 2, cz + core_d // 2))

    arm_w = style_rng.randint(6, 10)
    arm_d = style_rng.randint(6, 10)
    cells.update(_rect_cells(cx - arm_w // 2, cx + arm_w // 2, 0, cz + core_d // 2))
    cells.update(_rect_cells(cx - arm_w // 2, cx + arm_w // 2, cz - core_d // 2, depth - 1))
    cells.update(_rect_cells(0, cx + core_w // 2, cz - arm_d // 2, cz + arm_d // 2))
    cells.update(_rect_cells(cx - core_w // 2, width - 1, cz - arm_d // 2, cz + arm_d // 2))

    return _normalize_cells(cells), "star"


def _attach_wing(cells: Set[Cell2D], side: str, wing_w: int, wing_d: int, offset: int) -> Set[Cell2D]:
    xmin, xmax, zmin, zmax = _bbox_2d(cells)
    updated = set(cells)

    if side == "west":
        z0 = max(zmin, min(zmax - wing_d + 1, offset))
        wing = _rect_cells(xmin - wing_w + 2, xmin + 1, z0, z0 + wing_d - 1)
        connector = _rect_cells(xmin + 1, xmin + 3, z0 + wing_d // 2 - 1, z0 + wing_d // 2 + 1)
    elif side == "east":
        z0 = max(zmin, min(zmax - wing_d + 1, offset))
        wing = _rect_cells(xmax - 1, xmax + wing_w - 2, z0, z0 + wing_d - 1)
        connector = _rect_cells(xmax - 3, xmax - 1, z0 + wing_d // 2 - 1, z0 + wing_d // 2 + 1)
    elif side == "north":
        x0 = max(xmin, min(xmax - wing_w + 1, offset))
        wing = _rect_cells(x0, x0 + wing_w - 1, zmin - wing_d + 2, zmin + 1)
        connector = _rect_cells(x0 + wing_w // 2 - 1, x0 + wing_w // 2 + 1, zmin + 1, zmin + 3)
    else:
        x0 = max(xmin, min(xmax - wing_w + 1, offset))
        wing = _rect_cells(x0, x0 + wing_w - 1, zmax - 1, zmax + wing_d - 2)
        connector = _rect_cells(x0 + wing_w // 2 - 1, x0 + wing_w // 2 + 1, zmax - 3, zmax - 1)

    updated.update(wing)
    updated.update(connector)
    return updated


def _cap_footprint_span(cells: Set[Cell2D], max_span: int = 62) -> Set[Cell2D]:
    capped = set(cells)
    while capped:
        xmin, xmax, zmin, zmax = _bbox_2d(capped)
        changed = False
        span_x = xmax - xmin + 1
        span_z = zmax - zmin + 1
        if span_x > max_span:
            left = sum(1 for x, _z in capped if x == xmin)
            right = sum(1 for x, _z in capped if x == xmax)
            if left <= right:
                capped = {(x, z) for x, z in capped if x != xmin}
            else:
                capped = {(x, z) for x, z in capped if x != xmax}
            changed = True
        if span_z > max_span:
            top = sum(1 for _x, z in capped if z == zmin)
            bottom = sum(1 for _x, z in capped if z == zmax)
            if top <= bottom:
                capped = {(x, z) for x, z in capped if z != zmin}
            else:
                capped = {(x, z) for x, z in capped if z != zmax}
            changed = True
        if not changed:
            break
        comps = _connected_components(capped)
        if comps:
            capped = max(comps, key=len)
    return _normalize_cells(capped)


def _window_pattern(mode: int, x: int, z: int, floor_idx: int) -> bool:
    if mode == 0:
        return (x + z + floor_idx) % 2 == 0
    if mode == 1:
        return x % 3 == 0
    if mode == 2:
        return z % 3 == 0
    if mode == 3:
        return ((x // 2) + (z // 2) + floor_idx) % 2 == 0
    if mode == 4:
        return (x + floor_idx) % 4 in (1, 2)
    return (z + floor_idx) % 4 in (1, 2)


def _choose_footprint(style_rng: random.Random) -> Tuple[Set[Cell2D], str]:
    factories = (
        _footprint_rectangle,
        _footprint_l_shape,
        _footprint_u_shape,
        _footprint_ring,
        _footprint_plus,
        _footprint_h,
        _footprint_compound,
        _footprint_octagon,
        _footprint_star,
    )
    fn = factories[style_rng.randrange(len(factories))]
    return fn(style_rng)


def _select_door(cells: Set[Cell2D], style_rng: random.Random) -> Tuple[Cell2D, Tuple[int, int]]:
    xmin, xmax, zmin, zmax = _bbox_2d(cells)
    cx = (xmin + xmax) / 2.0
    cz = (zmin + zmax) / 2.0

    side_vectors = {
        "north": (0, -1),
        "south": (0, 1),
        "west": (-1, 0),
        "east": (1, 0),
    }
    side_order = ["north", "south", "west", "east"]
    style_rng.shuffle(side_order)

    boundary = _boundary_cells(cells)
    for side in side_order:
        dx, dz = side_vectors[side]
        candidates: List[Cell2D] = []
        for x, z in boundary:
            if (x + dx, z + dz) in cells:
                continue
            if side == "north" and z != zmin:
                continue
            if side == "south" and z != zmax:
                continue
            if side == "west" and x != xmin:
                continue
            if side == "east" and x != xmax:
                continue
            candidates.append((x, z))
        if not candidates:
            continue
        candidates.sort(key=lambda p: abs(p[0] - cx) + abs(p[1] - cz))
        return candidates[0], (dx, dz)

    fallback = min(boundary, key=lambda p: abs(p[0] - cx) + abs(p[1] - cz))
    for n in _neighbors4(fallback):
        if n not in cells:
            return fallback, (n[0] - fallback[0], n[1] - fallback[1])
    return fallback, (0, -1)


def _generate_simple_building(rng: random.Random, origin: Tuple[int, int, int], style_id: int) -> BuildingSpec:
    style_rng = random.Random(style_id * 1_000_003 + 17)
    palette = choose_palette(rng, style_id=style_id)

    x0, y0, z0 = origin
    block_map: Dict[Tuple[int, int, int], str] = {}

    def set_block(x: int, y: int, z: int, block: str) -> None:
        key = (x, y, z)
        if block == "air":
            block_map.pop(key, None)
            return
        block_map[key] = block

    width = style_rng.randint(11, 18)
    depth = style_rng.randint(9, 16)
    floors = 1 if style_rng.random() < 0.7 else 2
    floor_h = 4
    roof_mode = style_rng.randrange(3)  # 0=flat,1=gable_x,2=gable_z

    x1 = x0 + width - 1
    z1 = z0 + depth - 1

    # Simple rectangular base.
    for fx in range(x0, x1 + 1):
        for fz in range(z0, z1 + 1):
            set_block(fx, y0 - 1, fz, palette.pillar)

    for floor_idx in range(floors):
        fy = y0 + floor_idx * floor_h
        for x in range(x0, x1 + 1):
            for z in range(z0, z1 + 1):
                set_block(x, fy, z, palette.floor)
                for y in range(fy + 1, fy + floor_h):
                    on_wall = x in (x0, x1) or z in (z0, z1)
                    if not on_wall:
                        set_block(x, y, z, "air")
                        continue
                    if (x in (x0, x1) and z in (z0, z1)):
                        set_block(x, y, z, palette.pillar)
                        continue
                    is_window_band = y == fy + 2
                    if is_window_band and ((x + z + floor_idx) % 3 != 0):
                        set_block(x, y, z, palette.window)
                    else:
                        set_block(x, y, z, palette.wall)
        # Simple trim ring on each floor top.
        for x in range(x0, x1 + 1):
            set_block(x, fy + floor_h - 1, z0, palette.trim)
            set_block(x, fy + floor_h - 1, z1, palette.trim)
        for z in range(z0, z1 + 1):
            set_block(x0, fy + floor_h - 1, z, palette.trim)
            set_block(x1, fy + floor_h - 1, z, palette.trim)

    # Front door on south wall center.
    door_x = (x0 + x1) // 2
    door_z = z0
    set_block(door_x, y0 + 1, door_z, "air")
    set_block(door_x, y0 + 2, door_z, "air")
    set_block(door_x, y0 + 3, door_z, palette.trim)
    set_block(door_x - 1, y0 + 2, door_z, palette.light)
    set_block(door_x + 1, y0 + 2, door_z, palette.light)
    set_block(door_x, y0, door_z - 1, palette.trim)
    set_block(door_x, y0, door_z - 2, palette.trim)

    roof_y = y0 + floors * floor_h
    if roof_mode == 0:
        roof_name = "flat"
        for x in range(x0, x1 + 1):
            for z in range(z0, z1 + 1):
                set_block(x, roof_y, z, palette.roof)
        for x in range(x0, x1 + 1):
            set_block(x, roof_y + 1, z0, palette.roof_trim)
            set_block(x, roof_y + 1, z1, palette.roof_trim)
        for z in range(z0, z1 + 1):
            set_block(x0, roof_y + 1, z, palette.roof_trim)
            set_block(x1, roof_y + 1, z, palette.roof_trim)
    elif roof_mode == 1:
        roof_name = "gable_x"
        mid = (x0 + x1) / 2.0
        for x in range(x0, x1 + 1):
            h = max(0, int((width / 2.0 - abs(x - mid)) * 0.45))
            for z in range(z0, z1 + 1):
                for dy in range(h + 1):
                    set_block(x, roof_y + dy, z, palette.roof)
                if z in (z0, z1):
                    set_block(x, roof_y + h, z, palette.roof_trim)
    else:
        roof_name = "gable_z"
        mid = (z0 + z1) / 2.0
        for z in range(z0, z1 + 1):
            h = max(0, int((depth / 2.0 - abs(z - mid)) * 0.45))
            for x in range(x0, x1 + 1):
                for dy in range(h + 1):
                    set_block(x, roof_y + dy, z, palette.roof)
                if x in (x0, x1):
                    set_block(x, roof_y + h, z, palette.roof_trim)

    # Small yard border for context.
    margin = 2
    for x in range(x0 - margin, x1 + margin + 1):
        for z in range(z0 - margin, z1 + margin + 1):
            if x0 <= x <= x1 and z0 <= z <= z1:
                continue
            if x in (x0 - margin, x1 + margin) or z in (z0 - margin, z1 + margin):
                set_block(x, y0, z, palette.trim)

    blocks = [
        (x, y, z, block)
        for (x, y, z), block in sorted(block_map.items(), key=lambda item: (item[0][1], item[0][0], item[0][2]))
    ]
    if not blocks:
        raise RuntimeError("Simple building generation produced zero blocks.")

    return BuildingSpec(
        style=f"simple_house_{roof_name}_f{floors}_v{style_id:03d}",
        palette_name=palette.name,
        origin=origin,
        bbox=_bbox_from_blocks(blocks),
        blocks=blocks,
        style_id=style_id,
    )


def generate_building(
    rng: random.Random,
    origin: Tuple[int, int, int],
    style_id: Optional[int] = None,
    profile: str = "complex",
) -> BuildingSpec:
    if style_id is None:
        style_id = rng.randint(0, 999_999)
    style_id = int(style_id)
    if profile == "simple":
        return _generate_simple_building(rng, origin, style_id)
    if profile != "complex":
        raise ValueError(f"Unknown building profile: {profile}")

    style_rng = random.Random(style_id * 1_000_003 + 991)
    palette = choose_palette(rng, style_id=style_id)

    x0, y0, z0 = origin
    block_map: Dict[Tuple[int, int, int], str] = {}

    def set_block(x: int, y: int, z: int, block: str) -> None:
        key = (x, y, z)
        if block == "air":
            block_map.pop(key, None)
            return
        block_map[key] = block

    def fill_rect(xa: int, xb: int, y: int, za: int, zb: int, block: str) -> None:
        for xx in range(xa, xb + 1):
            for zz in range(za, zb + 1):
                set_block(xx, y, zz, block)

    base_cells_rel, footprint_name = _choose_footprint(style_rng)
    base_cells_rel = _normalize_cells(base_cells_rel)

    wing_mode = style_rng.randrange(6)
    xmin, xmax, zmin, zmax = _bbox_2d(base_cells_rel)
    if wing_mode in (1, 3):
        wing_w = style_rng.randint(6, 10)
        wing_d = style_rng.randint(6, 10)
        offset = style_rng.randint(zmin, max(zmin, zmax - wing_d + 1))
        base_cells_rel = _attach_wing(base_cells_rel, style_rng.choice(("west", "east")), wing_w, wing_d, offset)
    if wing_mode in (2, 4):
        wing_w = style_rng.randint(6, 10)
        wing_d = style_rng.randint(6, 10)
        offset = style_rng.randint(xmin, max(xmin, xmax - wing_w + 1))
        base_cells_rel = _attach_wing(base_cells_rel, style_rng.choice(("north", "south")), wing_w, wing_d, offset)

    if wing_mode == 5:
        # Detached outpost + connector for very distinctive silhouettes.
        bxmin, bxmax, bzmin, bzmax = _bbox_2d(base_cells_rel)
        outpost_w = style_rng.randint(6, 9)
        outpost_d = style_rng.randint(6, 9)
        ox = bxmax + style_rng.randint(7, 10)
        oz = (bzmin + bzmax) // 2 - outpost_d // 2
        outpost = _rect_cells(ox, ox + outpost_w - 1, oz, oz + outpost_d - 1)
        bridge_z = oz + outpost_d // 2
        connector = _rect_cells(bxmax - 1, ox, bridge_z - 1, bridge_z + 1)
        base_cells_rel.update(outpost)
        base_cells_rel.update(connector)

    base_cells_rel = _cap_footprint_span(base_cells_rel, max_span=62)

    # Keep only large connected components to avoid accidental tiny artifacts.
    components = sorted(_connected_components(base_cells_rel), key=len, reverse=True)
    if components:
        keep = set(components[0])
        for comp in components[1:]:
            if len(comp) >= 40:
                keep.update(comp)
        base_cells_rel = keep

    base_cells: Set[Cell2D] = {(x0 + x, z0 + z) for x, z in base_cells_rel}

    floors = style_rng.randint(2, 4)
    floor_h = 3 + style_rng.randrange(2)
    facade_mode = style_rng.randrange(6)

    floor_sets: List[Set[Cell2D]] = []
    cur = set(base_cells)
    floor_sets.append(set(cur))
    for _ in range(1, floors):
        if style_rng.random() < 0.55:
            shrink = _shrink_cells(cur)
            if len(shrink) >= max(60, int(len(cur) * 0.58)):
                cur = shrink
        floor_sets.append(set(cur))

    # Foundation and courtyard treatment.
    for x, z in _boundary_cells(base_cells):
        set_block(x, y0 - 1, z, palette.pillar)

    courtyard = _void_cells(base_cells)
    if courtyard:
        for x, z in courtyard:
            set_block(x, y0, z, "grass")
            if (x + z + style_id) % 7 == 0:
                set_block(x, y0 + 1, z, palette.light)

    for floor_idx, footprint in enumerate(floor_sets):
        floor_y = y0 + floor_idx * floor_h
        boundary = _boundary_cells(footprint)
        corners = _corner_cells(footprint)

        for x, z in footprint:
            if floor_idx == 0 or (x, z) not in courtyard:
                set_block(x, floor_y, z, palette.floor)

        window_levels = [floor_y + 2]
        if floor_h >= 4:
            window_levels.append(floor_y + 3)

        for x, z in footprint:
            if (x, z) in boundary:
                for y in range(floor_y + 1, floor_y + floor_h):
                    if y == floor_y + floor_h - 1:
                        set_block(x, y, z, palette.trim)
                    elif (x, z) not in corners and y in window_levels and _window_pattern(facade_mode, x, z, floor_idx):
                        set_block(x, y, z, palette.window)
                    else:
                        set_block(x, y, z, palette.pillar if (x, z) in corners else palette.wall)
            else:
                for y in range(floor_y + 1, floor_y + floor_h):
                    set_block(x, y, z, "air")

        # Floor cornice.
        for x, z in boundary:
            if (x + z + floor_idx) % 2 == 0:
                set_block(x, floor_y + floor_h - 1, z, palette.trim)

    door_cell, (ddx, ddz) = _select_door(floor_sets[0], style_rng)
    door_x, door_z = door_cell
    for y in range(y0 + 1, y0 + 3):
        set_block(door_x, y, door_z, "air")
    set_block(door_x, y0 + 3, door_z, palette.trim)

    left_dx, left_dz = -ddz, ddx
    right_dx, right_dz = ddz, -ddx
    set_block(door_x + left_dx, y0 + 2, door_z + left_dz, palette.light)
    set_block(door_x + right_dx, y0 + 2, door_z + right_dz, palette.light)

    for step in range(1, 5):
        sx = door_x + ddx * step
        sz = door_z + ddz * step
        set_block(sx, y0, sz, palette.trim)
        if step <= 2:
            set_block(sx, y0 + 1, sz, "air")

    # Buttresses for richer silhouettes.
    for x, z in sorted(_boundary_cells(base_cells)):
        if (x * 37 + z * 19 + style_id) % 13 != 0:
            continue
        outward = [n for n in _neighbors4((x, z)) if n not in base_cells]
        if not outward:
            continue
        ox, oz = outward[0]
        for by in range(y0 + 1, y0 + min(6, floor_h + 3)):
            set_block(ox, by, oz, palette.trim)

    top_cells = floor_sets[-1]
    roof_base_y = y0 + floors * floor_h
    roof_mode = style_rng.randrange(6)

    roof_name = "parapet"

    if roof_mode == 0:
        roof_name = "parapet"
        boundary = _boundary_cells(top_cells)
        for x, z in top_cells:
            set_block(x, roof_base_y, z, palette.roof)
        for x, z in boundary:
            set_block(x, roof_base_y, z, palette.roof_trim)
            set_block(x, roof_base_y + 1, z, palette.trim)
            if (x + z + style_id) % 11 == 0:
                set_block(x, roof_base_y + 2, z, palette.light)

    elif roof_mode == 1:
        roof_name = "stepped"
        layer_cells = set(top_cells)
        y = roof_base_y
        for _ in range(8):
            if not layer_cells:
                break
            boundary = _boundary_cells(layer_cells)
            interior = layer_cells - boundary
            for x, z in boundary:
                set_block(x, y, z, palette.roof_trim)
            for x, z in interior:
                set_block(x, y, z, palette.roof)
            if len(interior) < 15:
                break
            shr = _shrink_cells(layer_cells)
            if len(shr) < 12:
                break
            layer_cells = shr
            y += 1
        txmin, txmax, tzmin, tzmax = _bbox_2d(top_cells)
        set_block((txmin + txmax) // 2, y + 1, (tzmin + tzmax) // 2, palette.light)

    elif roof_mode == 2:
        roof_name = "ridge"
        boundary = _boundary_cells(top_cells)
        txmin, txmax, tzmin, tzmax = _bbox_2d(top_cells)
        axis = style_rng.choice(("x", "z"))
        if axis == "x":
            center = (txmin + txmax) / 2.0
            span = max(1.0, (txmax - txmin + 1) / 2.0)
        else:
            center = (tzmin + tzmax) / 2.0
            span = max(1.0, (tzmax - tzmin + 1) / 2.0)
        for x, z in top_cells:
            dist = abs(x - center) if axis == "x" else abs(z - center)
            h = max(0, int((span - dist) * 0.55))
            for dy in range(h + 1):
                set_block(x, roof_base_y + dy, z, palette.roof)
            if (x, z) in boundary:
                set_block(x, roof_base_y + h, z, palette.roof_trim)

    elif roof_mode == 3:
        roof_name = "dome"
        boundary = _boundary_cells(top_cells)
        txmin, txmax, tzmin, tzmax = _bbox_2d(top_cells)
        cx = (txmin + txmax) / 2.0
        cz = (tzmin + tzmax) / 2.0
        rx = max(1.0, (txmax - txmin + 1) / 2.0)
        rz = max(1.0, (tzmax - tzmin + 1) / 2.0)
        height = max(4, int(min(rx, rz) * 0.8))

        for x, z in top_cells:
            nx = (x - cx) / rx
            nz = (z - cz) / rz
            r2 = nx * nx + nz * nz
            if r2 > 1.0:
                continue
            h = int((1.0 - r2) * height)
            for dy in range(h + 1):
                block = palette.roof
                if dy < h and (x + z + dy + style_id) % 9 == 0:
                    block = palette.window
                set_block(x, roof_base_y + dy, z, block)
            if (x, z) in boundary:
                set_block(x, roof_base_y + h, z, palette.roof_trim)
        set_block(int(round(cx)), roof_base_y + height + 1, int(round(cz)), palette.light)

    elif roof_mode == 4:
        roof_name = "sawtooth"
        boundary = _boundary_cells(top_cells)
        axis = style_rng.choice(("x", "z"))
        for x, z in top_cells:
            stripe = (x if axis == "x" else z) % 4
            h = 1
            if stripe == 0:
                h = 3
            elif stripe == 1:
                h = 2
            for dy in range(h):
                set_block(x, roof_base_y + dy, z, palette.roof)
            if (x, z) in boundary:
                set_block(x, roof_base_y + h - 1, z, palette.roof_trim)

    else:
        roof_name = "lantern"
        boundary = _boundary_cells(top_cells)
        for x, z in top_cells:
            set_block(x, roof_base_y, z, palette.roof)
        for x, z in boundary:
            set_block(x, roof_base_y, z, palette.roof_trim)
            set_block(x, roof_base_y + 1, z, palette.trim)

        txmin, txmax, tzmin, tzmax = _bbox_2d(top_cells)
        cx = (txmin + txmax) // 2
        cz = (tzmin + tzmax) // 2
        s = 5 if (txmax - txmin) >= 10 and (tzmax - tzmin) >= 10 else 3
        hs = s // 2
        lx0, lx1 = cx - hs, cx + hs
        lz0, lz1 = cz - hs, cz + hs

        for y in range(roof_base_y + 1, roof_base_y + 5):
            for x in range(lx0, lx1 + 1):
                for z in range(lz0, lz1 + 1):
                    if x in (lx0, lx1) or z in (lz0, lz1):
                        block = palette.window
                        if (x in (lx0, lx1)) and (z in (lz0, lz1)):
                            block = palette.pillar
                        set_block(x, y, z, block)
                    else:
                        set_block(x, y, z, "air")
        fill_rect(lx0 - 1, lx1 + 1, roof_base_y + 5, lz0 - 1, lz1 + 1, palette.roof_trim)
        set_block(cx, roof_base_y + 6, cz, palette.light)

    # Optional towers to further diversify silhouettes.
    tower_mode = style_rng.randrange(7)
    txmin, txmax, tzmin, tzmax = _bbox_2d(base_cells)
    corners = [(txmin, tzmin), (txmin, tzmax), (txmax, tzmin), (txmax, tzmax)]
    mids = [((txmin + txmax) // 2, tzmin), ((txmin + txmax) // 2, tzmax), (txmin, (tzmin + tzmax) // 2), (txmax, (tzmin + tzmax) // 2)]
    tower_anchors: List[Tuple[int, int]] = []
    tower_name = "no_tower"

    if tower_mode == 1:
        tower_name = "pair_tower"
        start = style_id % 4
        tower_anchors = [corners[start], corners[(start + 2) % 4]]
    elif tower_mode == 2:
        tower_name = "corner_towers"
        tower_anchors = corners
    elif tower_mode == 3:
        tower_name = "corner_mid_mix"
        tower_anchors = corners + mids[:2]
    elif tower_mode == 4:
        tower_name = "gate_twin"
        tower_anchors = [mids[0], mids[1]]
    elif tower_mode == 5:
        tower_name = "triple_tower"
        picks = corners + mids
        style_rng.shuffle(picks)
        tower_anchors = picks[:3]
    elif tower_mode == 6:
        tower_name = "spire"
        tower_anchors = [((txmin + txmax) // 2, (tzmin + tzmax) // 2)]

    tower_base_height = floors * floor_h + style_rng.randint(3, 7)

    def build_tower(anchor_x: int, anchor_z: int, size: int, height: int) -> None:
        hs = size // 2
        xa, xb = anchor_x - hs, anchor_x + hs
        za, zb = anchor_z - hs, anchor_z + hs

        fill_rect(xa, xb, y0, za, zb, palette.floor)

        for y in range(y0 + 1, y0 + height + 1):
            for x in range(xa, xb + 1):
                for z in range(za, zb + 1):
                    on_edge = x in (xa, xb) or z in (za, zb)
                    if on_edge:
                        is_corner = x in (xa, xb) and z in (za, zb)
                        block = palette.pillar if is_corner else palette.wall
                        if not is_corner and y % 4 == 0 and (x + z + style_id) % 2 == 0:
                            block = palette.window
                        set_block(x, y, z, block)
                    else:
                        set_block(x, y, z, "air")

        top_y = y0 + height + 1
        for layer in range(3):
            lx0 = xa - layer
            lx1 = xb + layer
            lz0 = za - layer
            lz1 = zb + layer
            y = top_y + layer
            for x in range(lx0, lx1 + 1):
                for z in range(lz0, lz1 + 1):
                    if x in (lx0, lx1) or z in (lz0, lz1):
                        set_block(x, y, z, palette.roof_trim)
                    else:
                        set_block(x, y, z, palette.roof)
        set_block(anchor_x, top_y + 3, anchor_z, palette.light)

    if tower_anchors:
        for i, (ax, az) in enumerate(tower_anchors):
            size = 3 if (i + style_id) % 2 == 0 else 5
            height = tower_base_height + (i % 3) * 2
            build_tower(ax, az, size=size, height=height)

    # Decorative plaza ring to improve framing context.
    bxmin, bxmax, bzmin, bzmax = _bbox_2d(base_cells)
    path_margin = 3
    for x in range(bxmin - path_margin, bxmax + path_margin + 1):
        for z in range(bzmin - path_margin, bzmax + path_margin + 1):
            if (x, z) in base_cells:
                continue
            if x in (bxmin - path_margin, bxmax + path_margin) or z in (bzmin - path_margin, bzmax + path_margin):
                if (x + z + style_id) % 3 != 0:
                    set_block(x, y0, z, palette.trim)

    blocks = [
        (x, y, z, block)
        for (x, y, z), block in sorted(block_map.items(), key=lambda item: (item[0][1], item[0][0], item[0][2]))
    ]
    if not blocks:
        raise RuntimeError("Building generation produced zero blocks.")

    archetypes = (
        "civic",
        "fortress",
        "sanctum",
        "atelier",
        "library",
        "observatory",
        "palace",
        "market",
        "guildhall",
        "temple",
        "citadel",
        "garden_complex",
    )
    archetype = archetypes[style_id % len(archetypes)]

    bbox = _bbox_from_blocks(blocks)
    return BuildingSpec(
        style=f"{archetype}_{footprint_name}_{roof_name}_{tower_name}_v{style_id:03d}",
        palette_name=palette.name,
        origin=origin,
        bbox=bbox,
        blocks=blocks,
        style_id=style_id,
    )
