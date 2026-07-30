from __future__ import annotations
import argparse,json,os
from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from offline_evaluation.executor.canonical import load_npy
from .common import load,csvw
def save(fig,p):
 p.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(p.with_suffix(".png"),dpi=180);fig.savefig(p.with_suffix(".pdf"));plt.close(fig)
def projection(c):
 cases=pd.read_csv(c["_out"]/"analysis"/"structured_absolute_iou"/"high_priority_cases.csv");out=c["_out"]/"analysis"/"structured_absolute_iou"/"projections"
 for _,r in cases.iterrows():
  p=out/r.category/f"{int(r['rank']):02d}_{r.dataset_version}_{r.building_id}_{r.description_model}"
  if p.with_suffix(".png").is_file():continue
  gt=load_npy(c["_root"]/f"datasets/buildings_100_{r.dataset_version}"/r.building_id/"gt"/"voxels.npy",c["_root"]/f"datasets/buildings_100_{r.dataset_version}"/r.building_id/"gt"/"bbox.json");free=load_npy(c["_parent"]/"evaluation"/"voxels"/r.dataset_version/r.building_id/r.description_model/"free_form"/"voxels.npy",c["_parent"]/"evaluation"/"voxels"/r.dataset_version/r.building_id/r.description_model/"free_form"/"bbox.json");st=load_npy(c["_parent"]/"evaluation"/"voxels"/r.dataset_version/r.building_id/r.description_model/"structured"/"voxels.npy",c["_parent"]/"evaluation"/"voxels"/r.dataset_version/r.building_id/r.description_model/"structured"/"bbox.json")
  fig,axs=plt.subplots(3,3,figsize=(9,9));views=[("Top view (x,z)",0,2),("Front view (x,y)",0,1),("Side view (z,y)",2,1)]
  for col,(name,a,b) in enumerate(views):
   coords=[q for g in (gt,free,st) for q in g.voxels];lo=(min(q[a] for q in coords),min(q[b] for q in coords));hi=(max(q[a] for q in coords),max(q[b] for q in coords))
   for row,(label,g) in enumerate((("GT",gt),("Free-form",free),("Structured",st))):
    q=np.array(list(g.voxels))
    if q.size:axs[row,col].scatter(q[:,a],q[:,b],s=2,marker="s",c="black")
    axs[row,col].set(xlim=(lo[0]-1,hi[0]+1),ylim=(lo[1]-1,hi[1]+1),aspect="equal");axs[row,col].set_title(f"{label}: {name}")
  fig.suptitle(f"{r.scene_id} / {r.description_model} / {r.category}");save(fig,p)
def latex(d):
 cols=list(d.columns);return "\\begin{tabular}{%s}\n%s \\\\\\hline\n%s\n\\end{tabular}\n"%("l"*len(cols)," & ".join(cols),"\n".join(" & ".join(str(x).replace("_","\\_") for x in q)+" \\\\" for q in d.itertuples(index=False,name=None)))
def paper(c):
 td=c["_out"]/"paper_tables";fd=c["_out"]/"paper_figures";td.mkdir(parents=True,exist_ok=True);fd.mkdir(parents=True,exist_ok=True)
 phase=pd.read_csv(c["_out"]/"analysis"/"structured_absolute_iou"/"aggregate.csv");comp=pd.read_csv(c["_out"]/"analysis"/"completeness_repair"/"adjusted_models.csv");ground=pd.read_csv(c["_out"]/"analysis"/"grounding_pipeline"/"stage_correlations.csv");direct=pd.read_csv(c["_out"]/"analysis"/"direct_image_to_build"/"aggregate_metrics.csv")
 tables={"error_decomposition_position_intrinsic":phase,"completeness_adjusted_regression":comp,"material_recognition_to_grounding":ground,"direct_image_comparison":direct}
 sp=c["_out"]/"stochastic_repeat"/"analysis"/"stability_metrics.csv"
 if sp.is_file():tables["stochastic_repeat_summary"]=pd.read_csv(sp)
 for n,d in tables.items():d.to_csv(td/f"{n}.csv",index=False);(td/f"{n}.tex").write_text(latex(d))
 per=pd.read_csv(c["_out"]/"analysis"/"structured_absolute_iou"/"per_scene.csv");fig,ax=plt.subplots(figsize=(6,4));ax.scatter(per.delta_centroid_distance,per.delta_absolute_iou,s=9,alpha=.5);ax.set(xlabel="Structured − Free-form centroid distance (blocks)",ylabel="Structured − Free-form Absolute Occupancy IoU");save(fig,fd/"structured_absolute_iou_failure")
 fig,ax=plt.subplots(figsize=(6,4));vals=per[["free_position_grounding_loss","structured_position_grounding_loss","free_residual_intrinsic_geometry_error","structured_residual_intrinsic_geometry_error"]];vals.boxplot(ax=ax,rot=20);ax.set_ylabel("Diagnostic error");save(fig,fd/"position_vs_intrinsic_geometry")
 ds=pd.read_csv(c["_out"]/"analysis"/"direct_image_to_build"/"per_scene_metrics.csv");fig,ax=plt.subplots(figsize=(8,4));ds.boxplot(column="translation_aligned_iou",by="condition",ax=ax,rot=45);fig.suptitle("");ax.set_ylabel("Translation-Aligned Occupancy IoU");save(fig,fd/"direct_image_to_build_comparison")
 scores=pd.read_csv(c["_parent"]/"analysis"/"description_model"/"description_scores.csv");pm=pd.read_csv(c["_parent"]/"evaluation"/"per_scene_metrics.csv");z=pm[pm.representation=="free_form"].merge(scores,on=["scene_id","building_id","dataset_version","description_model"]);fig,ax=plt.subplots(figsize=(6,4));ax.scatter(z.dimension_score,z.bbox_dimension_mae,s=8,alpha=.5);ax.set(xlabel="Dimension description score",ylabel="Bounding-box dimension MAE (blocks)");save(fig,fd/"dimension_score_to_bbox_mae")
 fig,ax=plt.subplots(figsize=(7,4));means=z.groupby(pd.qcut(z.material_description_score.rank(method="first"),3))[["material_set_f1","material_histogram_similarity","material_aware_iou","exact_position_material_f1"]].mean();means.plot(ax=ax,marker="o");ax.set(xlabel="Material-description score tertile",ylabel="Mean metric");save(fig,fd/"material_selection_to_grounding")
 fig,ax=plt.subplots(figsize=(9,3));ax.axis("off");ax.text(.5,.5,"Multi-view images → description attributes → Fixed Builder actions → local 3D geometry → global position/material grounding",ha="center",va="center");save(fig,fd/"experimental_pipeline")
 fig,ax=plt.subplots(figsize=(9,3));ax.axis("off");ax.text(.5,.5,"Attribute recognition  →  selection/count fidelity  →  translation-aligned geometry  →  absolute coordinate and material grounding",ha="center",va="center");save(fig,fd/"description_to_grounding_stage_diagram")
 fig,ax=plt.subplots(figsize=(6,4));ax.scatter(pm.absolute_iou,pm.translation_aligned_iou,s=8,alpha=.4);ax.set(xlabel="Absolute Occupancy IoU",ylabel="Translation-Aligned Occupancy IoU");save(fig,fd/"absolute_vs_translation_aligned")
 (td/"README.md").write_text("# Table placement\n\nMain candidates: error decomposition, direct comparison, material grounding stages, and stochastic stability. Full regression, API reliability, provenance, axis shifts, normalization sensitivity, and per-model tests belong in the appendix.\n")
def api_usage(c):
 parts=[]
 for condition,path in (("direct_image_to_build",c["_out"]/"direct_image_to_build"/"manifests"/"calls.csv"),("stochastic_repeat",c["_out"]/"stochastic_repeat"/"manifests"/"calls.csv")):
  if path.is_file():
   d=pd.read_csv(path);d["experiment_condition"]=condition;parts.append(d)
 if not parts:return
 d=pd.concat(parts,ignore_index=True,sort=False);out=c["_out"]/"api_usage";d.to_csv(out/"per_call.csv",index=False) if out.mkdir(parents=True,exist_ok=True) is None else None
 a=d.groupby(["experiment_condition","model_id"],dropna=False).agg(attempts=("scene_id","size"),successful_calls=("validation_status",lambda x:(x=="success").sum()),failed_calls=("validation_status",lambda x:(x!="success").sum()),retry_calls=("attempt",lambda x:(pd.to_numeric(x)>1).sum()),input_tokens=("input_tokens","sum"),output_tokens=("output_tokens","sum"),total_tokens=("total_tokens","sum"),elapsed_time=("elapsed_time","sum")).reset_index();a.to_csv(out/"aggregate.csv",index=False);(out/"summary.md").write_text("# Follow-up API usage\n\nOnly new arr_followup_v1 calls are included. No monetary cost is inferred because responses did not provide authoritative price metadata.\n\n"+a.to_csv(index=False))
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="configs/arr_followup_v1.yaml");p.add_argument("--skip-projections",action="store_true");a=p.parse_args();c=load(a.config)
 if not a.skip_projections:projection(c)
 paper(c);api_usage(c)
if __name__=="__main__":main()
