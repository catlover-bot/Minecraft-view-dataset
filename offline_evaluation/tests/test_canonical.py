from pathlib import Path
from offline_evaluation.executor.canonical import Block, VoxelGrid, load_jsonl

def test_jsonl_roundtrip(tmp_path: Path):
    p=tmp_path/"v.jsonl"; g=VoxelGrid({(-1,2,3):Block.create("minecraft:glass_pane",{"color":"blue"})})
    g.write_jsonl(p,{"schema_version":"1.0"})
    h=load_jsonl(p)
    assert h.voxels == g.voxels and not h.errors

def test_invalid_record_is_explicit(tmp_path: Path):
    p=tmp_path/"bad.jsonl";p.write_text('{"x":"x","y":0,"z":0,"material":"stone"}\nnot-json\n')
    g=load_jsonl(p)
    assert not g.voxels and len(g.errors)==2
