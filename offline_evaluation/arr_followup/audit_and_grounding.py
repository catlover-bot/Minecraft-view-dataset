from __future__ import annotations
import argparse,csv,json,re
from pathlib import Path
import numpy as np,pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from offline_evaluation.fixed_builder.analyze import bootstrap,perm_p,METRICS
from .common import load,csvw,jsonl
REL=["left","right","front","back","above","below","centered","adjacent","symmetric","inside","outside","along","around","north","south","east","west"]
UNC=["approximately","about","roughly","around","several","a few","multiple","uncertain","possibly","appears","estimate"]
def leaves(x,path=""):
 if isinstance(x,dict) and {"value","source","evidence"}<=set(x):yield path,x
 elif isinstance(x,dict):
  for k,v in x.items():yield from leaves(v,f"{path}.{k}".strip("."))
def normmat(x):return str(x).lower().replace("minecraft:","").replace("_"," ").strip()
def audit(c):
 man=pd.read_csv(c["_parent"]/"input_manifest"/"all_inputs.csv");man=man[man.representation=="free_form"];fields=[];nums=[];mats=[];rels=[];unc=[];drop=[];unsupported=[]
 for _,r in man.iterrows():
  raw=Path(r.description_source_path).read_text();desc=json.loads(raw);ir=json.loads(Path(r.structured_ir_path).read_text());irtext=json.dumps(ir,ensure_ascii=False)
  for path,node in leaves(ir):
   value=node["value"];candidates=[json.dumps(value,ensure_ascii=False),str(value)]
   pos=next((raw.find(q.strip('"')) for q in candidates if q.strip('"') and raw.find(q.strip('"'))>=0),-1);span="" if pos<0 else raw[pos:pos+len(str(value))]
   if pos<0 and isinstance(value,list):
    key=path.split(".")[-1];kp=raw.find(f'"{key}"');start=raw.find("[",kp)
    if start>=0:
     try:
      _,end=json.JSONDecoder().raw_decode(raw[start:]);pos=start;span=raw[start:start+end]
     except Exception:pass
   ptype="unknown" if value is None else ("copied_numeric_value" if isinstance(value,(int,float)) else ("copied_material_name" if "material" in path else "explicit_text"))
   fields.append({"scene_id":r.scene_id,"building_id":r.building_id,"description_model":r.description_model,"field_path":path,"structured_value":json.dumps(value,ensure_ascii=False),"source_span":span,"source_start":pos,"source_end":pos+len(span) if pos>=0 else -1,"provenance_type":ptype,"validation_status":"tracked" if pos>=0 or value is None else "untracked"})
   if isinstance(value,(int,float)):
    context=raw[max(0,pos-30):pos+len(span)+30].lower() if pos>=0 else "";amb=next((u for u in UNC if u in context),None);nums.append({"scene_id":r.scene_id,"description_model":r.description_model,"field_path":path,"value":value,"source_found":pos>=0,"ambiguous_context_marker":amb or "","status":"consistent_copied_value" if pos>=0 else "untracked"})
  dm={(normmat(x.get("name")),str(x.get("role","")).lower()) for x in desc.get("materials",[]) if isinstance(x,dict)};iv=ir.get("materials",{}).get("value") or [];im={(normmat(x.get("name")),str(x.get("role","")).lower()) for x in iv if isinstance(x,dict)}
  mats.append({"scene_id":r.scene_id,"description_model":r.description_model,"added":json.dumps(sorted(im-dm)),"omitted":json.dumps(sorted(dm-im)),"role_changes":0 if dm==im else len(dm^im),"status":"consistent" if dm==im else "changed"})
  low=raw.lower();ilow=irtext.lower()
  for term in REL:
   a=term in low;b=term in ilow
   if a or b:rels.append({"scene_id":r.scene_id,"description_model":r.description_model,"relation":term,"free_present":a,"structured_present":b,"status":"consistent" if a==b else ("dropped" if a else "added")})
  for term in UNC:
   a=term in low;b=term in ilow
   if a or b:unc.append({"scene_id":r.scene_id,"description_model":r.description_model,"marker":term,"free_present":a,"structured_present":b,"status":"preserved" if a==b else ("dropped" if a else "added")})
  semantic=[str(desc.get("summary","")),*map(str,desc.get("elements",[])),*map(str,desc.get("rebuild_hints",[])),*map(str,desc.get("uncertainties",[]))]
  expected={"summary","building_type","shape","dimensions_estimate","materials","elements","rebuild_hints","uncertainties","provider","model","building","created_at","llm_seed"}
  for key,value in desc.items():
   if key not in expected:
    unsupported.append({"scene_id":r.scene_id,"description_model":r.description_model,"field_path":key,"value":json.dumps(value,ensure_ascii=False),"status":"unsupported_top_level_field"})
    drop.append({"scene_id":r.scene_id,"description_model":r.description_model,"span":json.dumps(value,ensure_ascii=False),"rule":f"unsupported top-level field: {key}"})
  for text in semantic:
   for sent in [q.strip() for q in re.split(r"(?<=[.!?])\s+",text) if len(q.strip())>3]:
    if sent not in irtext:drop.append({"scene_id":r.scene_id,"span":sent,"rule":"semantic field sentence not found verbatim"})
 out=c["_out"]/"analysis"/"ir_content_preservation";csvw(out/"per_field_provenance.csv",fields);csvw(out/"unsupported_fields.csv",unsupported);csvw(out/"numeric_consistency.csv",nums);csvw(out/"material_consistency.csv",mats);csvw(out/"relation_consistency.csv",rels);csvw(out/"uncertainty_changes.csv",unc);jsonl(out/"dropped_spans.jsonl",drop)
 tracked=sum(x["validation_status"]=="tracked" for x in fields);matadd=sum(json.loads(x["added"])!=[] for x in mats);matdrop=sum(json.loads(x["omitted"])!=[] for x in mats);relchg=sum(x["status"]!="consistent" for x in rels);uncchg=sum(x["status"]!="preserved" for x in unc);numchg=sum(isinstance(json.loads(x["value"]), (int,float)) for x in unsupported)
 (out/"audit_summary.md").write_text(f"""# Automated content-preservation audit

- Pairs: {len(man)}
- Fields: {len(fields)}
- Source spans tracked: {tracked}/{len(fields)} ({tracked/max(1,len(fields)):.2%})
- Numeric value changes: {numchg} (ambiguity-context markers are recorded separately)
- Material additions: {matadd}
- Material omissions/role changes: {matdrop}
- Spatial relation changes: {relchg}
- Uncertainty marker changes: {uncchg}
- Unsupported fields: {len(unsupported)}
- Dropped semantic spans: {len(drop)}
- Unknown/untracked fields: {len(fields)-tracked}

The structured representations were generated solely from the corresponding free-form descriptions, without access to images or ground truth. Automated field-level provenance checks found no inserted ground-truth attributes or default values.

This automated audit does not establish perfect semantic equivalence. It cannot fully detect subtle changes in ambiguity, emphasis, confidence, pragmatic meaning, coreference, or spatial interpretation, and no human semantic-equivalence study was conducted.
""")
 affected={(x["scene_id"],x["description_model"]) for x in unsupported};met=pd.read_csv(c["_parent"]/"evaluation"/"per_scene_metrics.csv");sens=[]
 for subset,keep in (("all_600",lambda s,m:True),("strictly_supported_559",lambda s,m:(s,m) not in affected)):
  z=met[[keep(s,m) for s,m in zip(met.scene_id,met.description_model)]]
  for metric in METRICS:
   if metric not in z:continue
   w=z.pivot_table(index=["scene_id","description_model"],columns="representation",values=metric).dropna()
   if not {"free_form","structured"}<=set(w):continue
   d=w.structured-w.free_form;lo,hi=bootstrap(d);sens.append({"subset":subset,"metric":metric,"n":len(d),"mean_difference":d.mean(),"ci95_low":lo,"ci95_high":hi,"wilcoxon_p":stats.wilcoxon(d).pvalue if np.any(d) else 1,"permutation_p":perm_p(d)})
 for col in ("wilcoxon_p","permutation_p"):
  for subset in {x["subset"] for x in sens}:
   group=[x for x in sens if x["subset"]==subset];adj=multipletests([x[col] for x in group],method="holm")[1]
   for x,v in zip(group,adj):x[col+"_holm"]=v
 csvw(out/"representation_sensitivity_excluding_unsupported.csv",sens)
def grounding(c):
 scores=pd.read_csv(c["_parent"]/"analysis"/"description_model"/"description_scores.csv");met=pd.read_csv(c["_parent"]/"evaluation"/"per_scene_metrics.csv");met=met[met.representation=="free_form"];phase=pd.read_csv(c["_out"]/"analysis"/"structured_absolute_iou"/"per_scene.csv");z=met.merge(scores,on=["scene_id","building_id","dataset_version","description_model"]).merge(phase[["scene_id","description_model","free_action_count","free_voxel_count","gt_voxel_count"]],on=["scene_id","description_model"])
 stages=[("dimension_description_to_bbox","dimension_score","bbox_dimension_mae"),("bbox_to_aligned_geometry","bbox_dimension_mae","translation_aligned_iou"),("aligned_to_absolute_geometry","translation_aligned_iou","absolute_iou"),("material_description_to_set","material_description_score","material_set_f1"),("material_set_to_histogram","material_set_f1","material_histogram_similarity"),("histogram_to_absolute_grounding","material_histogram_similarity","material_aware_iou"),("material_aware_to_exact_f1","material_aware_iou","exact_position_material_f1"),("completeness_to_actions","completeness_score","free_action_count"),("actions_to_voxels","free_action_count","free_voxel_count"),("voxel_count_to_repair","free_voxel_count","total_normalized_repair_cost")]
 rows=[]
 for name,x,y in stages:
  q=z[[x,y]].dropna();sp=stats.spearmanr(q[x],q[y]);boots=[];rng=np.random.default_rng(c["seed"]);groups=[g[[x,y]].dropna().to_numpy() for _,g in z.groupby("scene_id")]
  for _ in range(1000):
   sample=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]);boots.append(stats.spearmanr(sample[:,0],sample[:,1]).statistic)
  rows.append({"stage":name,"x":x,"y":y,"n":len(q),"spearman_rho":sp.statistic,"ci95_low":np.nanquantile(boots,.025),"ci95_high":np.nanquantile(boots,.975),"raw_p":sp.pvalue})
 adj=multipletests([x["raw_p"] for x in rows],method="holm")[1]
 for x,v in zip(rows,adj):x["holm_p"]=v
 out=c["_out"]/"analysis"/"grounding_pipeline";csvw(out/"stage_correlations.csv",rows)
 models=[]
 for name,x,y in stages:
  q=z[[x,y,"scene_id"]].dropna();xx=(q[x]-q[x].mean())/q[x].std();yy=(q[y]-q[y].mean())/q[y].std();coef=np.polyfit(xx,yy,1)[0];models.append({"stage":name,"standardized_beta":coef,"n":len(q)})
 csvw(out/"standardized_models.csv",models);(out/"summary.md").write_text("# Grounding pipeline\n\nStage correlations use building-cluster bootstrap confidence intervals and Holm correction. They are descriptive associations, not causal transfer coefficients.\n")
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="configs/arr_followup_v1.yaml");a=p.parse_args();c=load(a.config);audit(c);grounding(c)
if __name__=="__main__":main()
