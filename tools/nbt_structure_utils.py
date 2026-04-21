#!/usr/bin/env python3
from __future__ import annotations

import gzip
import io
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from tools.evaluate_rebuild_metrics import normalize_block_type

# NBT tag ids
TAG_End = 0
TAG_Byte = 1
TAG_Short = 2
TAG_Int = 3
TAG_Long = 4
TAG_Float = 5
TAG_Double = 6
TAG_Byte_Array = 7
TAG_String = 8
TAG_List = 9
TAG_Compound = 10
TAG_Int_Array = 11
TAG_Long_Array = 12


@dataclass(frozen=True)
class StructureExtractResult:
    voxels: np.ndarray
    bbox: Dict[str, int]
    metadata: Dict[str, Any]


def _read_exact(buf: io.BufferedReader, n: int) -> bytes:
    b = buf.read(n)
    if b is None or len(b) != n:
        raise EOFError(f"Unexpected EOF: expected {n} bytes")
    return b


def _read_u16(buf: io.BufferedReader) -> int:
    return struct.unpack(">H", _read_exact(buf, 2))[0]


def _read_i8(buf: io.BufferedReader) -> int:
    return struct.unpack(">b", _read_exact(buf, 1))[0]


def _read_u8(buf: io.BufferedReader) -> int:
    return struct.unpack(">B", _read_exact(buf, 1))[0]


def _read_i16(buf: io.BufferedReader) -> int:
    return struct.unpack(">h", _read_exact(buf, 2))[0]


def _read_i32(buf: io.BufferedReader) -> int:
    return struct.unpack(">i", _read_exact(buf, 4))[0]


def _read_i64(buf: io.BufferedReader) -> int:
    return struct.unpack(">q", _read_exact(buf, 8))[0]


def _read_f32(buf: io.BufferedReader) -> float:
    return struct.unpack(">f", _read_exact(buf, 4))[0]


def _read_f64(buf: io.BufferedReader) -> float:
    return struct.unpack(">d", _read_exact(buf, 8))[0]


def _read_string_payload(buf: io.BufferedReader) -> str:
    ln = _read_u16(buf)
    return _read_exact(buf, ln).decode("utf-8")


def _parse_payload(buf: io.BufferedReader, tag_id: int) -> Any:
    if tag_id == TAG_Byte:
        return _read_i8(buf)
    if tag_id == TAG_Short:
        return _read_i16(buf)
    if tag_id == TAG_Int:
        return _read_i32(buf)
    if tag_id == TAG_Long:
        return _read_i64(buf)
    if tag_id == TAG_Float:
        return _read_f32(buf)
    if tag_id == TAG_Double:
        return _read_f64(buf)
    if tag_id == TAG_Byte_Array:
        ln = _read_i32(buf)
        return list(_read_exact(buf, ln))
    if tag_id == TAG_String:
        return _read_string_payload(buf)
    if tag_id == TAG_List:
        elem_id = _read_u8(buf)
        ln = _read_i32(buf)
        return [_parse_payload(buf, elem_id) for _ in range(ln)]
    if tag_id == TAG_Compound:
        out: Dict[str, Any] = {}
        while True:
            child_id = _read_u8(buf)
            if child_id == TAG_End:
                break
            name = _read_string_payload(buf)
            out[name] = _parse_payload(buf, child_id)
        return out
    if tag_id == TAG_Int_Array:
        ln = _read_i32(buf)
        return [_read_i32(buf) for _ in range(ln)]
    if tag_id == TAG_Long_Array:
        ln = _read_i32(buf)
        return [_read_i64(buf) for _ in range(ln)]
    raise ValueError(f"Unsupported NBT tag id: {tag_id}")


def read_nbt_root(path: Path) -> Tuple[str, Dict[str, Any]]:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)

    buf = io.BufferedReader(io.BytesIO(raw))
    root_tag_id = _read_u8(buf)
    if root_tag_id != TAG_Compound:
        raise ValueError(f"Root tag must be TAG_Compound(10), got {root_tag_id}: {path}")
    root_name = _read_string_payload(buf)
    root_payload = _parse_payload(buf, TAG_Compound)
    if not isinstance(root_payload, dict):
        raise ValueError(f"Root payload must be compound dict: {path}")
    return root_name, root_payload


def _palette_entry_to_block_name(entry: Dict[str, Any]) -> str:
    if not isinstance(entry, dict):
        return "air"
    name = entry.get("Name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    name = entry.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return "air"


def _canonical_to_minecraft(block: str) -> str:
    b = normalize_block_type(block)
    mapping = {
        "air": "minecraft:air",
        "stonebrick": "minecraft:stone_bricks",
        "brick": "minecraft:bricks",
        "wood": "minecraft:oak_planks",
        "glass": "minecraft:glass",
        "fence": "minecraft:oak_fence",
        "slab_stone": "minecraft:stone_slab",
        "slab_wood": "minecraft:oak_slab",
        "stairs_wood": "minecraft:oak_stairs",
        "glowstone": "minecraft:glowstone",
        "stone": "minecraft:stone",
        "cobblestone": "minecraft:cobblestone",
        "nether_brick": "minecraft:nether_bricks",
        "quartz_block": "minecraft:quartz_block",
        "log": "minecraft:oak_log",
    }
    return mapping.get(b, f"minecraft:{b}")


def extract_structure_to_voxels(path: Path) -> StructureExtractResult:
    _root_name, root = read_nbt_root(path)

    size = root.get("size")
    if not (isinstance(size, list) and len(size) == 3):
        raise ValueError(f"Missing/invalid `size` in structure nbt: {path}")
    sx, sy, sz = (int(size[0]), int(size[1]), int(size[2]))
    if sx <= 0 or sy <= 0 or sz <= 0:
        raise ValueError(f"Invalid structure size {size} in {path}")

    palette_obj = root.get("palette")
    palettes_obj = root.get("palettes")
    palette: List[Dict[str, Any]]
    palette_source = "palette"

    if isinstance(palette_obj, list) and palette_obj and isinstance(palette_obj[0], dict):
        palette = palette_obj
    elif isinstance(palettes_obj, list) and palettes_obj and isinstance(palettes_obj[0], list):
        p0 = palettes_obj[0]
        if not (p0 and isinstance(p0[0], dict)):
            raise ValueError(f"Invalid palettes[0] in {path}")
        palette = p0
        palette_source = "palettes[0]"
    else:
        raise ValueError(f"Missing palette information in {path}")

    blocks = root.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError(f"Missing/invalid `blocks` list in {path}")

    vox = np.full((sy, sx, sz), "air", dtype="<U64")
    out_of_bounds = 0

    for block in blocks:
        if not isinstance(block, dict):
            continue
        pos = block.get("pos")
        state = block.get("state")
        if not (isinstance(pos, list) and len(pos) == 3):
            continue
        if state is None:
            continue

        x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
        sidx = int(state)
        if sidx < 0 or sidx >= len(palette):
            continue
        if x < 0 or x >= sx or y < 0 or y >= sy or z < 0 or z >= sz:
            out_of_bounds += 1
            continue

        raw_name = _palette_entry_to_block_name(palette[sidx])
        norm = normalize_block_type(raw_name)
        vox[y, x, z] = norm

    bbox = {
        "xmin": 0,
        "xmax": sx - 1,
        "ymin": 0,
        "ymax": sy - 1,
        "zmin": 0,
        "zmax": sz - 1,
        "order": "xmin,xmax,ymin,ymax,zmin,zmax",
        "voxel_axis_order": "Y,X,Z",
    }

    metadata = {
        "source_path": str(path),
        "size_xyz": [sx, sy, sz],
        "palette_len": len(palette),
        "palette_source": palette_source,
        "blocks_entries": len(blocks),
        "non_air_voxels": int(np.sum(vox != "air")),
        "out_of_bounds_block_entries": out_of_bounds,
    }
    return StructureExtractResult(voxels=vox, bbox=bbox, metadata=metadata)


def _write_u8(buf: io.BufferedWriter, v: int) -> None:
    buf.write(struct.pack(">B", int(v) & 0xFF))


def _write_i8(buf: io.BufferedWriter, v: int) -> None:
    buf.write(struct.pack(">b", int(v)))


def _write_i16(buf: io.BufferedWriter, v: int) -> None:
    buf.write(struct.pack(">h", int(v)))


def _write_i32(buf: io.BufferedWriter, v: int) -> None:
    buf.write(struct.pack(">i", int(v)))


def _write_i64(buf: io.BufferedWriter, v: int) -> None:
    buf.write(struct.pack(">q", int(v)))


def _write_str_payload(buf: io.BufferedWriter, s: str) -> None:
    b = s.encode("utf-8")
    buf.write(struct.pack(">H", len(b)))
    buf.write(b)


def _write_named_tag(buf: io.BufferedWriter, tag_id: int, name: str, payload: Any) -> None:
    _write_u8(buf, tag_id)
    _write_str_payload(buf, name)
    _write_payload(buf, tag_id, payload)


def _write_payload(buf: io.BufferedWriter, tag_id: int, payload: Any) -> None:
    if tag_id == TAG_Byte:
        _write_i8(buf, int(payload))
        return
    if tag_id == TAG_Short:
        _write_i16(buf, int(payload))
        return
    if tag_id == TAG_Int:
        _write_i32(buf, int(payload))
        return
    if tag_id == TAG_Long:
        _write_i64(buf, int(payload))
        return
    if tag_id == TAG_String:
        _write_str_payload(buf, str(payload))
        return
    if tag_id == TAG_List:
        elem_id, elems = payload
        _write_u8(buf, int(elem_id))
        _write_i32(buf, len(elems))
        for e in elems:
            _write_payload(buf, int(elem_id), e)
        return
    if tag_id == TAG_Compound:
        if not isinstance(payload, dict):
            raise ValueError("TAG_Compound payload must be dict[name -> (tag_id, payload)]")
        for child_name, child in payload.items():
            if not (isinstance(child, tuple) and len(child) == 2):
                raise ValueError(f"Invalid compound child for {child_name}")
            c_tag, c_payload = child
            _write_named_tag(buf, int(c_tag), str(child_name), c_payload)
        _write_u8(buf, TAG_End)
        return
    raise ValueError(f"Writing for tag id {tag_id} not implemented")


def write_structure_nbt_from_voxels(path: Path, voxels: np.ndarray, compress_gzip: bool = True) -> None:
    if voxels.ndim != 3:
        raise ValueError("voxels must be 3D with axis order Y,X,Z")

    sy, sx, sz = int(voxels.shape[0]), int(voxels.shape[1]), int(voxels.shape[2])
    palette_names: List[str] = []
    palette_index: Dict[str, int] = {}

    blocks_entries: List[Dict[str, Any]] = []

    for y in range(sy):
        for x in range(sx):
            for z in range(sz):
                block = normalize_block_type(voxels[y, x, z])
                if block == "air":
                    continue
                mc_name = _canonical_to_minecraft(block)
                if mc_name not in palette_index:
                    palette_index[mc_name] = len(palette_names)
                    palette_names.append(mc_name)
                blocks_entries.append(
                    {
                        "pos": [x, y, z],
                        "state": palette_index[mc_name],
                    }
                )

    root_compound: Dict[str, Tuple[int, Any]] = {
        "size": (TAG_List, (TAG_Int, [sx, sy, sz])),
        "entities": (TAG_List, (TAG_Compound, [])),
        "palette": (
            TAG_List,
            (
                TAG_Compound,
                [
                    {
                        "Name": (TAG_String, name),
                    }
                    for name in palette_names
                ],
            ),
        ),
        "blocks": (
            TAG_List,
            (
                TAG_Compound,
                [
                    {
                        "pos": (TAG_List, (TAG_Int, b["pos"])),
                        "state": (TAG_Int, b["state"]),
                    }
                    for b in blocks_entries
                ],
            ),
        ),
    }

    out_buf = io.BytesIO()
    writer = io.BufferedWriter(out_buf)
    _write_u8(writer, TAG_Compound)
    _write_str_payload(writer, "")
    _write_payload(writer, TAG_Compound, root_compound)
    writer.flush()
    raw = out_buf.getvalue()

    path.parent.mkdir(parents=True, exist_ok=True)
    if compress_gzip:
        path.write_bytes(gzip.compress(raw))
    else:
        path.write_bytes(raw)
