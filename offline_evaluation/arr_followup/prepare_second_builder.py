from __future__ import annotations
import argparse,pandas as pd
from .common import load,csvw,dump,sha
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="configs/arr_followup_v1.yaml");a=p.parse_args();c=load(a.config);phase=pd.read_csv(c["_out"]/"analysis"/"structured_absolute_iou"/"per_scene.csv").drop_duplicates("scene_id");phase["bin"]=pd.qcut(phase.gt_voxel_count.rank(method="first"),5,labels=False);sample=phase.groupby(["dataset_version","bin"],group_keys=False).head(5);inp=pd.read_csv(c["_parent"]/"input_manifest"/"all_inputs.csv");rows=[]
 for scene in sample.scene_id:
  for _,r in inp[inp.scene_id==scene].iterrows():rows.append({"experiment_id":"arr_followup_v1_second_builder_dryrun","scene_id":scene,"building_id":r.building_id,"description_model":r.description_model,"representation":r.representation,"input_path":r.input_path,"input_hash":r.input_hash,"builder_provider":"anthropic","builder_model_id":"claude-haiku-4-5-20251001","prompt_version":"second-fixed-builder-v1","status":"dry_run_not_executed","reason":"deferred until direct and stochastic primary priorities complete"})
 out=c["_out"]/"second_builder";csvw(out/"manifest.csv",rows);dump(out/"expected_calls.json",{"subset_buildings":len(sample),"expected_calls":len(rows),"api_called":False,"model_id":"claude-haiku-4-5-20251001"})
 (out/"README.md").write_text("# Second fixed builder\n\nPrepared as a separate experiment ID and not executed. Model capability and prompt behavior would be a cross-builder factor; absolute performance would not be treated as directly interchangeable with the OpenAI builder.\n")
if __name__=="__main__":main()
