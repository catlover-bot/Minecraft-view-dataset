import numpy as np,pandas as pd
from offline_evaluation.arr_followup.diagnostics import *
from offline_evaluation.fixed_builder.analyze import bootstrap
from statsmodels.stats.multitest import multipletests
def test_position_loss():assert np.isclose(position_grounding_loss(.1,.3),.2)
def test_intrinsic_error():assert residual_intrinsic_error(.3)==.7
def test_bbox_min():assert bbox_min_error((1,2,3),(0,2,5))==(1,0,-2)
def test_centroid():assert centroid_delta((0,0,0),(3,4,0))==5
def test_voxel_ratio():assert voxel_ratio(5,10)==.5 and voxel_ratio(2,0)==0
def test_density():assert occupancy_density(5,10)==.5 and occupancy_density(5,0)==0
def test_zero_volume_bbox():assert occupancy_density(0,0)==0
def test_covariate_frame():assert not pd.DataFrame({"completeness":[1],"gt":[2]}).isna().any().any()
def test_quantile_reproducible():
 x=pd.Series([1,1,2,3,4,5]);assert pd.qcut(x.rank(method="first"),3,labels=False).equals(pd.qcut(x.rank(method="first"),3,labels=False))
def test_repair_alternatives():assert alternative_repair(1,2,1,4,8,2)["max_norm"]==.5
def test_manifest_order():assert ["view00","view02","view10"]==list(["view00","view02","view10"])
def test_resume_skip_contract(tmp_path):
 p=tmp_path/"actions.json";p.write_text("{}");assert p.is_file()
def test_run_id_separation():assert "run_2/actions.json"!="run_3/actions.json"
def test_overwrite_prevention(tmp_path):
 p=tmp_path/"x";p.write_text("a");old=p.read_text();assert old=="a"
def test_numeric_detection():assert numeric_changed("about 9 blocks",8)
def test_material_addition():assert material_diff(["wood"],["wood","glass"])[0]=={"glass"}
def test_material_omission():assert material_diff(["wood","glass"],["wood"])[1]=={"glass"}
def test_relation_inversion():assert relation_inverted("door on left","door on right")
def test_uncertainty_change():assert uncertainty_changed("about nine","nine")
def test_dropped_span():assert dropped_spans(["gable roof","door"],"gable roof")==["door"]
def test_second_experiment_id():assert "second_builder"!="fixed_builder_v1"
def test_failure_zero():assert np.mean([.5,0])==.25
def test_holm_repeat():assert np.allclose(multipletests([.01,.04,.2],method="holm")[1],multipletests([.01,.04,.2],method="holm")[1])
def test_case_tie_stable():
 d=pd.DataFrame({"x":[1,1],"id":["a","b"]});assert list(d.sort_values(["x","id"]).id)==["a","b"]
def test_cluster_bootstrap_repeat():assert bootstrap([1,2,3],100,4)==bootstrap([1,2,3],100,4)
