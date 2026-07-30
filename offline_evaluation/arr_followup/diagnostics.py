from __future__ import annotations
import math,re
def position_grounding_loss(absolute_iou,aligned_iou):return aligned_iou-absolute_iou
def residual_intrinsic_error(aligned_iou):return 1-aligned_iou
def voxel_ratio(pred,gt):return 0 if gt==0 else pred/gt
def occupancy_density(voxels,volume):return 0 if volume==0 else voxels/volume
def bbox_min_error(pred,gt):return tuple(pred[i]-gt[i] for i in range(3))
def centroid_delta(a,b):return math.dist(a,b)
def alternative_repair(add,delete,replace,gt,pred,overlap):
 total=add+delete+replace
 safe=lambda n,d:0 if not d else n/d
 return {"gt_norm":safe(total,gt),"max_norm":safe(total,max(gt,pred)),"sum_norm":safe(total,gt+pred),"delete_pred_norm":safe(delete,pred),"add_gt_norm":safe(add,gt),"replace_overlap_norm":safe(replace,overlap)}
def numeric_changed(text,value):
 nums=[float(x) for x in re.findall(r"\d+(?:\.\d+)?",text)];return bool(nums) and float(value) not in nums
def material_diff(free,structured):return set(structured)-set(free),set(free)-set(structured)
def relation_inverted(free,structured):
 pairs=[("left","right"),("front","back"),("above","below"),("inside","outside")]
 return any(a in free.lower() and b in structured.lower() for a,b in pairs for a,b in ((a,b),(b,a)))
def uncertainty_changed(free,structured):
 marks=("approximately","about","roughly","around","uncertain","possibly")
 return any((m in free.lower())!=(m in structured.lower()) for m in marks)
def dropped_spans(spans,structured):return [x for x in spans if x not in structured]
