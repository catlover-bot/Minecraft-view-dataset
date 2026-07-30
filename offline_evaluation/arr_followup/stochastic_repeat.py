from __future__ import annotations
import argparse,csv,json,time,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
from scipy import stats
from tools.llm_client import complete_multimodal_with_meta,extract_json_object
from tools.llm_config import load_llm_config
from offline_evaluation.fixed_builder.common import require_key,redact,text_sha
from offline_evaluation.fixed_builder.run_builder import validate
from offline_evaluation.fixed_builder.evaluate import align,extra_material
from offline_evaluation.executor import execute
from offline_evaluation.executor.canonical import load_npy,write_npy
from offline_evaluation.metrics import occupancy,materials,repair
from .common import load,dump,csvw,sha
def prepare(c):
 phase=pd.read_csv(c["_out"]/"analysis"/"structured_absolute_iou"/"per_scene.csv");base=phase.drop_duplicates("scene_id").copy();base["complexity_bin"]=base.groupby("dataset_version").gt_voxel_count.transform(lambda x:pd.qcut(x.rank(method="first"),5,labels=False));rng=np.random.default_rng(c["seed"]);chosen=[]
 for (ds,q),z in base.groupby(["dataset_version","complexity_bin"]):
  take=c["stochastic_repeat"]["sample_size"]//10;chosen += list(rng.choice(z.scene_id,take,replace=False))
 inp=pd.read_csv(c["_parent"]/"input_manifest"/"all_inputs.csv");rows=[]
 for scene in sorted(chosen):
  for model in ("openai","claude","gemini"):
   for rep in ("free_form","structured"):
    r=inp[(inp.scene_id==scene)&(inp.description_model==model)&(inp.representation==rep)].iloc[0]
    for run in c["stochastic_repeat"]["additional_runs"]:rows.append({"scene_id":scene,"dataset_version":r.dataset_version,"building_id":r.building_id,"description_model":model,"representation":rep,"run_id":run,"input_path":r.input_path,"input_hash":r.input_hash,"model_id":c["stochastic_repeat"]["model_id"],"status":"ready"})
 csvw(c["_out"]/"stochastic_repeat"/"sample_manifest.csv",rows);dump(c["_out"]/"stochastic_repeat"/"expected_calls.json",{"selected_buildings":len(chosen),"selection_seed":c["seed"],"additional_calls":len(rows),"runs":[2,3],"scene_ids":sorted(chosen)})
def run(c,key=""):
 require_key("openai");cfg=load_llm_config(None);cfg.provider="openai";cfg.openai_model=c["stochastic_repeat"]["model_id"];base=c["_out"]/"stochastic_repeat";rows=list(csv.DictReader((base/"sample_manifest.csv").open()));sy=(c["_root"]/"offline_evaluation"/"fixed_builder"/"prompts"/"builder_system_prompt.txt").read_text();ut=(c["_root"]/"offline_evaluation"/"fixed_builder"/"prompts"/"builder_user_template.txt").read_text()
 if key:rows=[r for r in rows if f"{r['scene_id']}|{r['description_model']}|{r['representation']}|{r['run_id']}"==key]
 for r in rows:
  dest=base/"parsed_actions"/r["dataset_version"]/r["building_id"]/r["description_model"]/r["representation"]/f"run_{r['run_id']}"/"actions.json"
  if dest.is_file():continue
  rawdir=base/"raw_responses"/r["dataset_version"]/r["building_id"]/r["description_model"]/r["representation"]/f"run_{r['run_id']}";attempt=max([int(x.stem.split("_")[-1]) for x in rawdir.glob("attempt_*.json")],default=0)+1;user=ut.replace("{input}",Path(r["input_path"]).read_text());t=time.monotonic();rec={**r,"experiment_id":c["experiment_id"],"experiment_condition":"stochastic_repeat","prompt_hash":text_sha(sy+"\n"+ut),"attempt":attempt,"request_timestamp":datetime.now(timezone.utc).isoformat()}
  try:
   comp=complete_multimodal_with_meta(cfg,sy,user,[],0,12000);raw=rawdir/f"attempt_{attempt:03d}.json";dump(raw,comp.raw_response);obj=extract_json_object(comp.text);errs=validate(obj,2000,64);ex=execute(obj.get("operations",[])) if not errs else None;rec.update(response_timestamp=datetime.now(timezone.utc).isoformat(),response_id=comp.raw_response.get("id",""),input_tokens=comp.usage.get("input_tokens",0),output_tokens=comp.usage.get("output_tokens",0),total_tokens=comp.usage.get("total_tokens",0),elapsed_time=time.monotonic()-t,api_status="success",parse_status="success",validation_status="success" if not errs else "failure",execution_status=ex.status.value if ex else "not_run",error_type="" if not errs else "ValidationError",error_message=";".join(errs))
   if not errs:dump(dest,obj)
  except Exception as e:rec.update(response_timestamp=datetime.now(timezone.utc).isoformat(),response_id="",input_tokens=0,output_tokens=0,total_tokens=0,elapsed_time=time.monotonic()-t,api_status="failure",parse_status="failure",validation_status="not_run",execution_status="not_run",error_type=type(e).__name__,error_message=redact(e))
  dump(base/"manifests"/"per_call"/r["dataset_version"]/r["building_id"]/r["description_model"]/r["representation"]/f"run_{r['run_id']}"/f"attempt_{attempt:03d}.json",rec)
def evaluate(c):
 base=c["_out"]/"stochastic_repeat";man=pd.read_csv(base/"sample_manifest.csv");rows=[]
 for _,r in man.iterrows():
  act=base/"parsed_actions"/r.dataset_version/r.building_id/r.description_model/r.representation/f"run_{r.run_id}"/"actions.json"
  if not act.is_file():continue
  obj=json.loads(act.read_text());pred=execute(obj["operations"]).grid;gtp=c["_root"]/f"datasets/buildings_100_{r.dataset_version}"/r.building_id/"gt";gt=load_npy(gtp/"voxels.npy",gtp/"bbox.json");shift,al=align(gt,pred,c["alignment"]["xz"],c["alignment"]["y"]);ab=occupancy(gt,pred);aa=occupancy(gt,al);m=materials(gt,pred);rep=repair(gt,al);gs=gt.size();ps=pred.size();vox=base/"evaluation"/"voxels"/r.dataset_version/r.building_id/r.description_model/r.representation/f"run_{r.run_id}";write_npy(pred,vox/"voxels.npy",vox/"bbox.json")
  rows.append({"scene_id":r.scene_id,"building_id":r.building_id,"dataset_version":r.dataset_version,"description_model":r.description_model,"representation":r.representation,"run_id":r.run_id,"absolute_iou":ab["occupancy_iou"],"translation_aligned_iou":aa["occupancy_iou"],"bbox_dimension_mae":sum(abs(ps[i]-gs[i]) for i in range(3))/3,"material_set_f1":extra_material(gt,pred)["material_set_f1"],"material_aware_iou":m["material_aware_iou"],"total_normalized_repair_cost":rep["total_normalized_repair_cost"],"action_count":len(obj["operations"]),"voxel_count":len(pred.voxels)})
 d=pd.DataFrame(rows);d.to_csv(base/"evaluation"/"per_scene_metrics.csv",index=False);analyze(c,d)
def analyze(c,d):
 parent=pd.read_csv(c["_parent"]/"evaluation"/"per_scene_metrics.csv");sel=set(d.scene_id);p=parent[parent.scene_id.isin(sel)].copy();p["run_id"]=1;cols=list(d.columns);all_=pd.concat([p[[x for x in cols if x in p.columns]],d],ignore_index=True);out=c["_out"]/"stochastic_repeat"/"analysis";out.mkdir(parents=True,exist_ok=True);all_.to_csv(out/"all_runs.csv",index=False);metrics=["absolute_iou","translation_aligned_iou","bbox_dimension_mae","material_set_f1","material_aware_iou","total_normalized_repair_cost"];summ=[]
 for keys,z in all_.groupby(["description_model","representation"]):
  for m in metrics:summ.append({"description_model":keys[0],"representation":keys[1],"metric":m,"within_condition_sd":z.groupby("scene_id")[m].std().mean(),"coefficient_of_variation":z.groupby("scene_id")[m].std().mean()/z[m].mean(),"icc":icc(z,"scene_id","run_id",m)})
 pd.DataFrame(summ).to_csv(out/"stability_metrics.csv",index=False);effects=[]
 for run in (1,2,3):
  z=all_[all_.run_id==run]
  for model in ("openai","claude","gemini"):
   w=z[z.description_model==model].pivot(index="scene_id",columns="representation",values=metrics)
   for m in metrics:
    diff=w[m]["structured"]-w[m]["free_form"];effects.append({"run_id":run,"description_model":model,"metric":m,"mean_effect":diff.mean(),"positive_scene_fraction":(diff>0).mean(),"negative_scene_fraction":(diff<0).mean()})
 pd.DataFrame(effects).to_csv(out/"representation_effects_by_run.csv",index=False)
 bbox=pd.DataFrame(effects);bi=bbox[bbox.metric=="bbox_dimension_mae"];ai=bbox[bbox.metric=="absolute_iou"];al=bbox[bbox.metric=="translation_aligned_iou"]
 (c["_out"]/"stochastic_repeat"/"summary.md").write_text(f"# Stochastic repeat\n\nSelected buildings: {d.scene_id.nunique()}. Runs 1–3 are distinguished explicitly; run 1 reuses the parent experiment. Bbox improvement means a negative Structured−Free-form effect; Absolute-IoU reduction means a negative effect. See analysis tables for scene-, run-, model-, and pooled denominators.\n")
def icc(d,subject,run,value):
 q=d.pivot_table(index=subject,columns=run,values=value).dropna().values
 if q.shape[0]<2 or q.shape[1]<2:return np.nan
 n,k=q.shape;gm=q.mean();msb=k*((q.mean(1)-gm)**2).sum()/(n-1);msw=((q-q.mean(1,keepdims=True))**2).sum()/(n*(k-1));return (msb-msw)/(msb+(k-1)*msw) if msb+(k-1)*msw else np.nan
def collect(c):
 rows=[json.loads(x.read_text()) for x in (c["_out"]/"stochastic_repeat"/"manifests"/"per_call").glob("**/*.json")];csvw(c["_out"]/"stochastic_repeat"/"manifests"/"calls.csv",rows)
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="configs/arr_followup_v1.yaml");p.add_argument("command",choices=["prepare","run","collect","evaluate"]);p.add_argument("--key",default="");a=p.parse_args();c=load(a.config);{"prepare":prepare,"collect":collect,"evaluate":evaluate}.get(a.command,lambda c:run(c,a.key))(c)
if __name__=="__main__":main()
