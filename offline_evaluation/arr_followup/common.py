from __future__ import annotations
import csv,json,hashlib
from pathlib import Path
from typing import Any
import yaml
ROOT=Path(__file__).resolve().parents[2]
def load(path):
 c=yaml.safe_load((ROOT/path).read_text());c["_root"]=ROOT;c["_out"]=ROOT/c["output_root"];c["_parent"]=ROOT/c["parent_experiment"];return c
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for x in iter(lambda:f.read(1024*1024),b""):h.update(x)
 return h.hexdigest()
def dump(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")
def csvw(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fields=sorted({k for r in rows for k in r})
 with p.open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def jsonl(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows))
