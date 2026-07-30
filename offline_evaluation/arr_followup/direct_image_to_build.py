from __future__ import annotations
import argparse,csv,json,math,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
from tools.generate_building_descriptions import _pick_images
from tools.llm_client import complete_multimodal_with_meta,extract_json_object
from tools.llm_config import load_llm_config
from offline_evaluation.fixed_builder.common import require_key,redact,text_sha
from offline_evaluation.fixed_builder.run_builder import validate
from offline_evaluation.executor import execute
from offline_evaluation.executor.canonical import load_npy,write_npy
from offline_evaluation.metrics import occupancy,materials,geometry,repair,best_alignment
from offline_evaluation.fixed_builder.evaluate import align,extra_material
from offline_evaluation.fixed_builder.analyze import bootstrap,perm_p
from statsmodels.stats.multitest import multipletests
from scipy import stats
from .common import load,sha,dump,csvw

def prepare(c):
 rows=[];rng=np.random.default_rng(c["seed"]);n=c["direct_image_to_build"]["subset_size"]
 for ds in ("v1","v4"):
  vals=[]
  for i in range(100):
   bid=f"building_{i:03d}";g=load_npy(c["_root"]/f"datasets/buildings_100_{ds}"/bid/"gt"/"voxels.npy",c["_root"]/f"datasets/buildings_100_{ds}"/bid/"gt"/"bbox.json");vals.append((bid,len(g.voxels)))
  d=pd.DataFrame(vals,columns=["building_id","gt_voxel_count"]);d["stratum"]=pd.qcut(d.gt_voxel_count.rank(method="first"),5,labels=False)
  for s,z in d.groupby("stratum"):
   take=max(1,(n//2)//5);idx=rng.choice(z.index,take,replace=False)
   for _,q in z.loc[idx].sort_values("building_id").iterrows():
    b=c["_root"]/f"datasets/buildings_100_{ds}"/q.building_id;meta=json.loads((b/"meta.json").read_text());images=_pick_images(meta,b,c["direct_image_to_build"]["max_images"])
    rows.append({"scene_id":f"{ds}:{q.building_id}","dataset_version":ds,"building_id":q.building_id,"stratum":int(s),"gt_voxel_count":q.gt_voxel_count,"model_id":c["direct_image_to_build"]["model_id"],"prompt_version":c["direct_image_to_build"]["prompt_version"],"image_paths":json.dumps([str(x) for x in images]),"image_hashes":json.dumps([sha(x) for x in images]),"view_order":json.dumps([x.name for x in images]),"status":"ready"})
 csvw(c["_out"]/"direct_image_to_build"/"manifests"/"sample_manifest.csv",rows);return rows
def run(c,scene=""):
 require_key("openai");cfg=load_llm_config(None);cfg.provider="openai";cfg.openai_model=c["direct_image_to_build"]["model_id"];base=c["_out"]/"direct_image_to_build";mp=base/"manifests"/"sample_manifest.csv"
 rows=list(csv.DictReader(mp.open()));rows=[r for r in rows if not scene or r["scene_id"]==scene];sy=(c["_root"]/"offline_evaluation"/"arr_followup"/"prompts"/"direct_system_prompt.txt").read_text();us=(c["_root"]/"offline_evaluation"/"arr_followup"/"prompts"/"direct_user_prompt.txt").read_text()
 for r in rows:
  ok=base/"parsed_actions"/r["dataset_version"]/r["building_id"]/"actions.json"
  if ok.is_file():continue
  rawdir=base/"raw_responses"/r["dataset_version"]/r["building_id"];attempt=max([int(x.stem.split("_")[-1]) for x in rawdir.glob("attempt_*.json")],default=0)+1;t=time.monotonic();start=datetime.now(timezone.utc);rec={**r,"experiment_id":c["experiment_id"],"experiment_condition":"direct_image_to_build","prompt_hash":text_sha(sy+"\n"+us),"attempt":attempt,"request_timestamp":start.isoformat()}
  try:
   comp=complete_multimodal_with_meta(cfg,sy,us,[Path(x) for x in json.loads(r["image_paths"])],c["direct_image_to_build"]["temperature"],c["direct_image_to_build"]["max_tokens"]);raw=rawdir/f"attempt_{attempt:03d}.json";dump(raw,comp.raw_response);obj=extract_json_object(comp.text);errs=validate(obj,c["direct_image_to_build"]["max_actions"],c["direct_image_to_build"]["max_coordinate"]);ex=execute(obj.get("operations",[])) if not errs else None
   rec.update(response_timestamp=datetime.now(timezone.utc).isoformat(),response_id=comp.raw_response.get("id",""),input_tokens=comp.usage.get("input_tokens",0),output_tokens=comp.usage.get("output_tokens",0),total_tokens=comp.usage.get("total_tokens",0),finish_reason=comp.raw_response.get("status",""),elapsed_time=time.monotonic()-t,api_status="success",parse_status="success",validation_status="success" if not errs else "failure",execution_status=ex.status.value if ex else "not_run",error_type="" if not errs else "ValidationError",error_message=";".join(errs),raw_response_path=str(raw),parsed_action_path=str(ok))
   if not errs:dump(ok,obj);dump(base/"validation"/r["dataset_version"]/r["building_id"]/"validation.json",{"errors":[],"execution_status":ex.status.value})
  except Exception as e:rec.update(response_timestamp=datetime.now(timezone.utc).isoformat(),response_id="",input_tokens=0,output_tokens=0,total_tokens=0,finish_reason="",elapsed_time=time.monotonic()-t,api_status="failure",parse_status="failure",validation_status="not_run",execution_status="not_run",error_type=type(e).__name__,error_message=redact(e),raw_response_path="",parsed_action_path="");dump(base/"errors"/r["dataset_version"]/r["building_id"]/f"attempt_{attempt:03d}.json",rec)
  dump(base/"manifests"/"per_call"/r["dataset_version"]/r["building_id"]/f"attempt_{attempt:03d}.json",rec)
def collect(c):
 rows=[json.loads(x.read_text()) for x in (c["_out"]/"direct_image_to_build"/"manifests"/"per_call").glob("**/*.json")];csvw(c["_out"]/"direct_image_to_build"/"manifests"/"calls.csv",rows)
def evaluate(c):
 base=c["_out"]/"direct_image_to_build";sample=pd.read_csv(base/"manifests"/"sample_manifest.csv");rows=[];fails=[]
 for _,r in sample.iterrows():
  act=base/"parsed_actions"/r.dataset_version/r.building_id/"actions.json"
  if not act.is_file():fails.append({"scene_id":r.scene_id,"reason":"missing_output"});continue
  ex=execute(json.loads(act.read_text())["operations"]);gtp=c["_root"]/f"datasets/buildings_100_{r.dataset_version}"/r.building_id/"gt";gt=load_npy(gtp/"voxels.npy",gtp/"bbox.json");pred=ex.grid;shift,al=align(gt,pred,c["alignment"]["xz"],c["alignment"]["y"]);_,diag,_=best_alignment(gt,pred,c["alignment"]["diagnostic"]);ab=occupancy(gt,pred);aa=occupancy(gt,al);m=materials(gt,pred);rep=repair(gt,al);em=extra_material(gt,pred);gs=gt.size();ps=pred.size();vox=base/"evaluation"/"voxels"/r.dataset_version/r.building_id;write_npy(pred,vox/"voxels.npy",vox/"bbox.json")
  rows.append({"scene_id":r.scene_id,"dataset_version":r.dataset_version,"building_id":r.building_id,"condition":"direct_image_to_build","evaluation_status":"success","absolute_iou":ab["occupancy_iou"],"translation_aligned_iou":aa["occupancy_iou"],"absolute_f1":ab["f1"],"translation_aligned_f1":aa["f1"],"bbox_dimension_mae":sum(abs(ps[i]-gs[i]) for i in range(3))/3,"centroid_distance":math.dist(gt.centroid(),pred.centroid()),"diagnostic_aligned_iou_10":occupancy(gt,diag)["occupancy_iou"],"material_aware_iou":m["material_aware_iou"],"exact_position_material_f1":2*m["exact_voxel_count"]/(len(gt.voxels)+len(pred.voxels)),**em,**rep})
 pd.DataFrame(rows).to_csv(base/"evaluation"/"per_scene_metrics.csv",index=False);pd.DataFrame(fails).to_csv(base/"evaluation"/"failures.csv",index=False)
 compare(c,pd.DataFrame(rows))
def compare(c,direct):
 parent=pd.read_csv(c["_parent"]/"evaluation"/"per_scene_metrics.csv");parent["condition"]=parent.representation+"_"+parent.description_model;all_=pd.concat([direct,parent[parent.scene_id.isin(direct.scene_id)]],ignore_index=True,sort=False);out=c["_out"]/"analysis"/"direct_image_to_build";out.mkdir(parents=True,exist_ok=True);all_.to_csv(out/"per_scene_metrics.csv",index=False);metrics=["absolute_iou","translation_aligned_iou","bbox_dimension_mae","material_set_f1","material_histogram_similarity","material_aware_iou","exact_position_material_f1","normalized_add_cost","normalized_delete_cost","normalized_replace_cost","total_normalized_repair_cost"];tests=[];wtl=[]
 for model in ("openai","claude","gemini"):
  for rep in ("free_form","structured"):
   lang=parent[(parent.description_model==model)&(parent.representation==rep)&parent.scene_id.isin(direct.scene_id)]
   for metric in metrics:
    w=direct[["scene_id",metric]].merge(lang[["scene_id",metric]],on="scene_id",suffixes=("_direct","_language"));diff=w[metric+"_direct"]-w[metric+"_language"];lo,hi=bootstrap(diff);tests.append({"comparison":f"direct_vs_{rep}_{model}","metric":metric,"n":len(diff),"mean_difference":diff.mean(),"median_difference":diff.median(),"ci95_low":lo,"ci95_high":hi,"wilcoxon_p":stats.wilcoxon(diff).pvalue if np.any(diff) else 1,"permutation_p":perm_p(diff),"effect_size_dz":diff.mean()/diff.std(ddof=1) if diff.std(ddof=1) else 0});wtl.append({"comparison":f"direct_vs_{rep}_{model}","metric":metric,"wins":sum(diff>1e-12),"ties":sum(abs(diff)<=1e-12),"losses":sum(diff< -1e-12)})
 for col in ("wilcoxon_p","permutation_p"):
  adj=multipletests([x[col] for x in tests],method="holm")[1]
  for x,v in zip(tests,adj):x[col+"_holm"]=v
 pd.DataFrame(tests).to_csv(out/"paired_tests.csv",index=False);pd.DataFrame(wtl).to_csv(out/"win_tie_loss.csv",index=False);all_.groupby(["condition"],dropna=False)[metrics].mean().to_csv(out/"aggregate_metrics.csv")
 (out/"summary.md").write_text("# Direct Image-to-Build\n\nThis is a stratified subset analysis. The same `gpt-5-mini` model family, action schema, coordinate range, validator, executor, and alignment definition are used; the direct multimodal prompt necessarily differs from the language-input prompt.\n")
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="configs/arr_followup_v1.yaml");p.add_argument("command",choices=["prepare","run","collect","evaluate"]);p.add_argument("--scene-id",default="");a=p.parse_args();c=load(a.config)
 if a.command=="prepare":prepare(c)
 elif a.command=="run":run(c,a.scene_id)
 elif a.command=="collect":collect(c)
 else:evaluate(c)
if __name__=="__main__":main()
