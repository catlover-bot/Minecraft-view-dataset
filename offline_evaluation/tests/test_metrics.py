from offline_evaluation.executor import Block, VoxelGrid
from offline_evaluation.metrics import evaluate_pair, occupancy, repair

def grid(items):
    return VoxelGrid({c:Block.create(m) for c,m in items})

def test_identical_metrics():
    g=grid([((0,0,0),"stone"),((1,0,0),"brick")])
    m=evaluate_pair(g,g,1)
    assert m["occupancy_iou"] == 1 and m["material_aware_iou"] == 1
    assert m["total_normalized_repair_cost"] == 0

def test_translation_alignment_and_repair_breakdown():
    g=grid([((0,0,0),"stone"),((1,0,0),"brick")])
    p=grid([((1,0,0),"stone"),((2,0,0),"glass")])
    m=evaluate_pair(g,p,2)
    assert m["absolute_iou"] == 1/3 and m["translation_aligned_iou"] == 1
    assert (m["aligned_dx"],m["aligned_dy"],m["aligned_dz"]) == (-1,0,0)
    assert m["replace_count"] == 1

def test_empty_convention():
    e=VoxelGrid()
    assert occupancy(e,e)["occupancy_iou"] == 1
    assert repair(e,e)["total_normalized_repair_cost"] == 0
