from __future__ import annotations
import argparse,csv,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np,pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from offline_evaluation.executor.canonical import load_npy
from offline_evaluation.fixed_builder.analyze import bootstrap
from offline_evaluation.metrics import complexity
from tools.evaluate_description_quality import _completeness
from .common import load,csvw

def safe(n,d,empty=0):return empty if not d else n/d
def details(grid):
 b=grid.bbox();c=grid.centroid();n=len(grid.voxels)
 if not b:return {"min_x":0,"min_y":0,"min_z":0,"centroid_x":0,"centroid_y":0,"centroid_z":0,"voxel_count":0,"bbox_volume":0,"occupancy_density":0}
 vol=(b["xmax"]-b["xmin"]+1)*(b["ymax"]-b["ymin"]+1)*(b["zmax"]-b["zmin"]+1)
 return {"min_x":b["xmin"],"min_y":b["ymin"],"min_z":b["zmin"],"centroid_x":c[0],"centroid_y":c[1],"centroid_z":c[2],"voxel_count":n,"bbox_volume":vol,"occupancy_density":safe(n,vol)}
def corr(x,y):
 z=pd.DataFrame({"x":x,"y":y}).dropna()
 if len(z)<3 or z.x.nunique()<2 or z.y.nunique()<2:return (len(z),np.nan,np.nan)
 q=stats.spearmanr(z.x,z.y);return len(z),q.statistic,q.pvalue
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="configs/arr_followup_v1.yaml");a=p.parse_args();c=load(a.config);parent=c["_parent"];met=pd.read_csv(parent/"evaluation"/"per_scene_metrics.csv");scores=pd.read_csv(parent/"analysis"/"description_model"/"description_scores.csv")
 inp=pd.read_csv(parent/"input_manifest"/"all_inputs.csv");actions=parent/"builder_outputs"/"parsed_actions";rows=[];comp_rows=[]
 for (scene,model),z in met.groupby(["scene_id","description_model"]):
  if set(z.representation)!={"free_form","structured"}:continue
  ds,bid=scene.split(":");gt=load_npy(c["_root"]/f"datasets/buildings_100_{ds}"/bid/"gt"/"voxels.npy",c["_root"]/f"datasets/buildings_100_{ds}"/bid/"gt"/"bbox.json");gd=details(gt);r={"scene_id":scene,"building_id":bid,"dataset_version":ds,"description_model":model}
  for rep,label in (("free_form","free"),("structured","structured")):
   m=z[z.representation==rep].iloc[0];pred=load_npy(parent/"evaluation"/"voxels"/ds/bid/model/rep/"voxels.npy",parent/"evaluation"/"voxels"/ds/bid/model/rep/"bbox.json");d=details(pred)
   d["centroid_distance"]=math.dist((d["centroid_x"],d["centroid_y"],d["centroid_z"]),(gd["centroid_x"],gd["centroid_y"],gd["centroid_z"]))
   for k,v in d.items():r[f"{label}_{k}"]=v
   r[f"{label}_centroid_distance"]=d["centroid_distance"]
   for k in ("absolute_iou","translation_aligned_iou","bbox_dimension_mae","alignment_shift_x","alignment_shift_y","alignment_shift_z","alignment_shift_magnitude","normalized_add_cost","normalized_delete_cost","normalized_replace_cost","total_normalized_repair_cost"):r[f"{label}_{k}"]=m[k]
   r[f"{label}_action_count"]=len(json.loads((actions/ds/bid/model/rep/"actions.json").read_text())["operations"])
  for k in ("absolute_iou","translation_aligned_iou","bbox_dimension_mae","centroid_distance","alignment_shift_magnitude","voxel_count","bbox_volume","occupancy_density"):r["delta_"+k]=r["structured_"+k]-r["free_"+k]
  r["free_voxel_count_ratio"]=safe(r["free_voxel_count"],gd["voxel_count"]);r["structured_voxel_count_ratio"]=safe(r["structured_voxel_count"],gd["voxel_count"]);r["delta_voxel_count_ratio"]=r["structured_voxel_count_ratio"]-r["free_voxel_count_ratio"]
  r["delta_shift_l1"]=sum(abs(r[f"structured_alignment_shift_{x}"]-r[f"free_alignment_shift_{x}"]) for x in "xyz");r["delta_shift_l2"]=math.sqrt(sum((r[f"structured_alignment_shift_{x}"]-r[f"free_alignment_shift_{x}"])**2 for x in "xyz"))
  for k,v in gd.items():r["gt_"+k]=v
  for label in ("free","structured"):
   r[label+"_position_grounding_loss"]=r[label+"_translation_aligned_iou"]-r[label+"_absolute_iou"];r[label+"_residual_intrinsic_geometry_error"]=1-r[label+"_translation_aligned_iou"]
  r["delta_position_grounding_loss"]=r["structured_position_grounding_loss"]-r["free_position_grounding_loss"];r["delta_residual_intrinsic_geometry_error"]=r["structured_residual_intrinsic_geometry_error"]-r["free_residual_intrinsic_geometry_error"]
  rows.append(r)
  sr=scores[(scores.scene_id==scene)&(scores.description_model==model)].iloc[0];desc_path=Path(inp[(inp.scene_id==scene)&(inp.description_model==model)&(inp.representation=="free_form")].iloc[0].description_source_path);desc=json.loads(desc_path.read_text());text=json.dumps(desc,ensure_ascii=False);r2={**r,**sr.to_dict(),"description_char_count":len(text),"description_token_count":len(text.split()),"detail_count":len(desc.get("elements",[]))+len(desc.get("rebuild_hints",[])),"gt_material_count":len({b.material for b in gt.voxels.values()})};comp_rows.append(r2)
 out=c["_out"]/"analysis"/"structured_absolute_iou";out.mkdir(parents=True,exist_ok=True);df=pd.DataFrame(rows);df.to_csv(out/"per_scene.csv",index=False)
 metrics=["delta_absolute_iou","delta_translation_aligned_iou","delta_bbox_dimension_mae","delta_centroid_distance","delta_alignment_shift_magnitude","delta_voxel_count_ratio","delta_occupancy_density","delta_position_grounding_loss","delta_residual_intrinsic_geometry_error"]
 agg=[]
 for model in ["pooled","openai","claude","gemini"]:
  z=df if model=="pooled" else df[df.description_model==model]
  for m in metrics:
   lo,hi=bootstrap(z[m]);agg.append({"model":model,"metric":m,"n":len(z),"mean":z[m].mean(),"median":z[m].median(),"ci95_low":lo,"ci95_high":hi})
 pd.DataFrame(agg).to_csv(out/"aggregate.csv",index=False)
 reg=df.merge(scores,on=["scene_id","building_id","dataset_version","description_model"]);reg["complexity"]=np.log1p(reg.gt_voxel_count);formula="delta_absolute_iou ~ delta_bbox_dimension_mae + delta_centroid_distance + delta_alignment_shift_magnitude + delta_voxel_count_ratio + delta_occupancy_density + C(description_model) + complexity";fit=smf.ols(formula,reg).fit(cov_type="cluster",cov_kwds={"groups":reg.scene_id})
 rr=[{"term":k,"estimate":fit.params[k],"se":fit.bse[k],"p":fit.pvalues[k],"ci95_low":fit.conf_int().loc[k,0],"ci95_high":fit.conf_int().loc[k,1]} for k in fit.params.index]
 X=pd.DataFrame({"delta_bbox_mae":reg.delta_bbox_dimension_mae,"delta_centroid_distance":reg.delta_centroid_distance,"delta_shift":reg.delta_alignment_shift_magnitude,"delta_voxel_ratio":reg.delta_voxel_count_ratio,"delta_density":reg.delta_occupancy_density,"complexity":reg.complexity});X=X.replace([np.inf,-np.inf],np.nan).dropna();vif=[{"term":k,"vif":variance_inflation_factor(X.values,i)} for i,k in enumerate(X.columns)];pd.DataFrame(rr).merge(pd.DataFrame(vif),on="term",how="left").to_csv(out/"regression_results.csv",index=False)
 cats={"bbox_improved_absolute_worse":(df.delta_bbox_dimension_mae<0)&(df.delta_absolute_iou<0),"aligned_better_absolute_worse":(df.delta_translation_aligned_iou>0)&(df.delta_absolute_iou<0),"all_improved":(df.delta_absolute_iou>0)&(df.delta_translation_aligned_iou>0)&(df.delta_bbox_dimension_mae<0),"all_worse":(df.delta_absolute_iou<0)&(df.delta_translation_aligned_iou<0)&(df.delta_bbox_dimension_mae>0),"shift_changed":df.delta_shift_l2>df.delta_shift_l2.quantile(.9),"voxel_over":df.structured_voxel_count_ratio>df.structured_voxel_count_ratio.quantile(.9),"voxel_under":df.structured_voxel_count_ratio<df.structured_voxel_count_ratio.quantile(.1)}
 cases=[]
 for name,mask in cats.items():
  sort="delta_absolute_iou" if "absolute" in name else ("delta_shift_l2" if name=="shift_changed" else "structured_voxel_count_ratio");asc=name not in {"shift_changed","voxel_over"}
  for rank,(_,q) in enumerate(df[mask].sort_values(sort,ascending=asc).head(20).iterrows(),1):cases.append({"category":name,"rank":rank,**q.to_dict()})
 pd.DataFrame(cases).to_csv(out/"high_priority_cases.csv",index=False)
 (out/"summary.md").write_text("# Structured Absolute-IoU decomposition\n\nPosition-grounding loss is defined diagnostically as aligned IoU minus absolute IoU. Residual intrinsic geometry error is 1 minus aligned IoU. These are descriptive diagnostics, not a causal decomposition.\n\nSee aggregate.csv, regression_results.csv, and high_priority_cases.csv.\n")
 analyze_completeness(c,pd.DataFrame(comp_rows))
def analyze_completeness(c,d):
 out=c["_out"]/"analysis"/"completeness_repair";out.mkdir(parents=True,exist_ok=True);rows=[]
 targets=["free_normalized_add_cost","free_normalized_delete_cost","free_normalized_replace_cost","free_total_normalized_repair_cost","free_voxel_count","free_voxel_count_ratio","free_action_count","description_char_count","description_token_count","gt_voxel_count","gt_bbox_volume","gt_material_count"]
 for model in ["pooled","openai","claude","gemini"]:
  z=d if model=="pooled" else d[d.description_model==model]
  for y in targets:
   n,r,p=corr(z.completeness_score,z[y]);rows.append({"model":model,"outcome":y,"n":n,"spearman_rho":r,"raw_p":p})
 pd.DataFrame(rows).to_csv(out/"correlations.csv",index=False)
 formula="free_total_normalized_repair_cost ~ completeness_score + C(description_model) + np.log1p(gt_voxel_count) + np.log1p(gt_bbox_volume) + gt_material_count + description_char_count";fit=smf.ols(formula,d).fit(cov_type="cluster",cov_kwds={"groups":d.scene_id});pd.DataFrame([{"term":k,"estimate":fit.params[k],"se":fit.bse[k],"p":fit.pvalues[k],"ci95_low":fit.conf_int().loc[k,0],"ci95_high":fit.conf_int().loc[k,1]} for k in fit.params.index]).to_csv(out/"adjusted_models.csv",index=False)
 d=d.copy();d["completeness_quantile"]=pd.qcut(d.completeness_score.rank(method="first"),3,labels=["low","mid","high"]);d.groupby("completeness_quantile",observed=True)[["free_voxel_count_ratio","free_normalized_add_cost","free_normalized_delete_cost","free_normalized_replace_cost","free_total_normalized_repair_cost","free_translation_aligned_iou","free_bbox_dimension_mae"]].agg(["count","mean","median"]).to_csv(out/"quantile_results.csv")
 sens=[]
 for _,r in d.iterrows():
  gt=r.gt_voxel_count;pr=r.free_voxel_count;add=r.free_normalized_add_cost*gt;delete=r.free_normalized_delete_cost*gt;replace=r.free_normalized_replace_cost*gt;overlap=max(0,min(gt,pr)-replace)
  sens.append({**{k:r[k] for k in ("scene_id","description_model","completeness_score")},"original":safe(add+delete+replace,gt),"max_norm":safe(add+delete+replace,max(gt,pr)),"sum_norm":safe(add+delete+replace,gt+pr),"delete_pred_norm":safe(delete,pr),"add_gt_norm":safe(add,gt),"replace_overlap_norm":safe(replace,overlap)})
 sd=pd.DataFrame(sens);outrows=[]
 for model in ["pooled","openai","claude","gemini"]:
  z=sd if model=="pooled" else sd[sd.description_model==model]
  for y in ("original","max_norm","sum_norm","delete_pred_norm","add_gt_norm","replace_overlap_norm"):
   n,r,p=corr(z.completeness_score,z[y]);outrows.append({"model":model,"normalization":y,"n":n,"spearman_rho":r,"raw_p":p})
 pd.DataFrame(outrows).to_csv(out/"normalization_sensitivity.csv",index=False);pd.DataFrame(rows)[pd.DataFrame(rows).model!="pooled"].to_csv(out/"per_model_results.csv",index=False)
 (out/"summary.md").write_text("# Completeness and repair burden\n\nResults are associations, not causal effects. The adjusted model uses OLS with building-cluster-robust standard errors because each building occurs under three description models. Quantile cut points are empirical tertiles with deterministic rank tie-breaking. Alternative repair normalizations define zero denominators as zero.\n")
if __name__=="__main__":main()
