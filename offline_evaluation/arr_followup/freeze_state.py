from __future__ import annotations
import argparse,platform,subprocess,sys
from pathlib import Path
from .common import load,sha,csvw,dump
def run(*x):return subprocess.run(x,cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True).stdout
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="configs/arr_followup_v1.yaml");a=p.parse_args();c=load(a.config);out=c["_out"]/"frozen_state";out.mkdir(parents=True,exist_ok=True)
 (out/"git_commit.txt").write_text(run("git","rev-parse","HEAD"));(out/"git_diff.patch").write_text(run("git","diff","--binary")+"\n# Untracked files\n"+run("git","ls-files","--others","--exclude-standard"))
 (out/"environment.txt").write_text(f"python={platform.python_version()}\nplatform={platform.platform()}\n"+run(sys.executable,"-m","pip","freeze"))
 roots=[c["_parent"],c["_root"]/"configs"/"fixed_builder_v1.yaml",c["_root"]/"offline_evaluation"/"metrics.py",c["_root"]/"offline_evaluation"/"fixed_builder"]
 files=[]
 for r in roots:
  files += [r] if r.is_file() else list(r.rglob("*"))
 rows=[{"path":str(x.relative_to(c["_root"])),"sha256":sha(x),"bytes":x.stat().st_size} for x in sorted(set(files)) if x.is_file() and "__pycache__" not in x.parts]
 csvw(out/"file_hashes.csv",rows);dump(out/"parent_experiment.json",{"experiment_id":c["experiment_id"],"parent_experiment":"fixed_builder_v1","parent_path":str(c["_parent"]),"parent_hash_manifest":"frozen_state/file_hashes.csv","fixed_builder_outputs":1200})
if __name__=="__main__":main()
