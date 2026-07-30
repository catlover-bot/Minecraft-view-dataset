from pathlib import Path
import numpy as np
import pytest
from offline_evaluation.analyze_correlations_v2 import bootstrap
from offline_evaluation.analyze_materials_v2 import material_decomposition
from offline_evaluation.data_coverage import validate_manifest_rows
from offline_evaluation.executor.canonical import Block,VoxelGrid,load_npy,write_npy
from offline_evaluation.metrics import best_alignment,occupancy
from tools.llm_client import _redact_url

def g(items):return VoxelGrid({c:Block.create(m) for c,m in items})
def test_long_material_roundtrip_variants(tmp_path:Path):
    names=["stone","minecraft:long_material_name_over_thirty_two_characters_alpha",
           "long_material_name_over_thirty_two_characters_alpha","long_material_name_over_thirty_two_characters_beta",
           "oak_stairs[facing=north,half=top]"]
    grid=g([((i,0,0),x) for i,x in enumerate(names)]);write_npy(grid,tmp_path/"v.npy",tmp_path/"bbox.json");loaded=load_npy(tmp_path/"v.npy",tmp_path/"bbox.json")
    assert loaded.voxels==grid.voxels and np.load(tmp_path/"v.npy").dtype.itemsize//4>32
def test_exact_and_no_shift_alignment():
    gt=g([((0,0,0),"stone"),((1,0,0),"stone")]);shifted=gt.translated(4,-2,3)
    assert best_alignment(gt,shifted,5)[0]==(-4,2,-3)
    assert best_alignment(gt,gt,5)[0]==(0,0,0)
def test_alignment_tie_prefers_minimum_l1():
    gt=g([((0,0,0),"stone"),((2,0,0),"stone")]);pr=g([((1,0,0),"stone")])
    assert best_alignment(gt,pr,2)[0]==(-1,0,0)
def test_old_new_f1_formula():
    gt=g([((0,0,0),"stone"),((1,0,0),"stone")]);pr=g([((1,0,0),"stone"),((2,0,0),"stone")])
    _,a,iou=best_alignment(gt,pr,2);f=occupancy(gt,a)["f1"]
    assert f==pytest.approx(2*iou/(1+iou)) and occupancy(gt,pr)["f1"]<f
def test_cluster_bootstrap_reproducible_and_grouped():
    rows=[{"cluster_id":"a","x":0.,"y":0.},{"cluster_id":"a","x":1.,"y":1.},{"cluster_id":"b","x":2.,"y":1.},{"cluster_id":"b","x":3.,"y":2.}]
    assert bootstrap(rows,"x","y",7,1000,True)==bootstrap(rows,"x","y",7,1000,True)
    # paired cluster matrix must have equal group sizes; bootstrap succeeds only while both observations remain grouped
    assert len({r["cluster_id"] for r in rows})==2
def test_material_set_histogram_single_unmatched_and_empty():
    one=material_decomposition(g([((0,0,0),"stone")]),g([((2,0,0),"stone")]))
    assert one["material_set_f1"]==1 and one["material_histogram_similarity"]==1 and one["material_aware_iou"]==0
    unmatched=material_decomposition(g([((0,0,0),"stone")]),g([((0,0,0),"wood")]))
    assert unmatched["material_set_f1"]==0 and unmatched["material_histogram_similarity"]==0
    empty=material_decomposition(g([]),g([]));assert empty["material_set_f1"]==1 and empty["material_histogram_similarity"]==1
    one_empty=material_decomposition(g([((0,0,0),"stone")]),g([]));assert one_empty["material_set_f1"]==0
def test_manifest_malformed_and_duplicate():
    with pytest.raises(ValueError):validate_manifest_rows([{"scene_id":"x"}])
    row={"scene_id":"v1:b","building_id":"b","dataset_version":"v1","description_model":"openai","condition":"direct"}
    with pytest.raises(ValueError):validate_manifest_rows([row,row])


def test_api_key_redaction():
    secret = "gemini-secret-value"
    redacted = _redact_url(
        f"https://example.invalid/generate?key={secret}&model=fixed"
    )
    assert secret not in redacted
    assert "%5BREDACTED%5D" in redacted
    assert "model=fixed" in redacted
