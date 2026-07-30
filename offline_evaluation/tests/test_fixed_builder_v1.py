import json,os
import numpy as np
from offline_evaluation.fixed_builder.common import redact,text_sha
from offline_evaluation.fixed_builder.prepare_inputs import canonical
from offline_evaluation.fixed_builder.run_builder import validate
from offline_evaluation.fixed_builder.analyze import bootstrap,perm_p

def test_api_key_redaction(monkeypatch):
 monkeypatch.setenv("OPENAI_API_KEY","secret-123")
 assert "secret-123" not in redact("Authorization: secret-123")
def test_prompt_hash_stable():
 assert text_sha("x")==text_sha("x") and text_sha("x")!=text_sha("y")
def test_invalid_and_unsupported_actions():
 assert validate({"operations":[{"op":"clone"}]},10,64)
 assert validate({"operations":[{"op":"set","x":0,"y":0,"z":0}]},10,64)
def test_empty_action_list_valid():
 assert validate({"operations":[]},10,64)==[]
def test_structured_ir_provenance_and_no_defaults():
 x=canonical({"summary":"gable","dimensions_estimate":{"width":9}})
 assert x["dimensions_estimate"]["width"]["value"]==9
 assert x["dimensions_estimate"]["depth"]["value"] is None
 assert x["dimensions_estimate"]["depth"]["source"]=="unknown"
def test_building_cluster_bootstrap_deterministic():
 x=[1,2,3,4];assert bootstrap(x,100,7)==bootstrap(x,100,7)
def test_paired_permutation_identical():
 assert perm_p([0,0,0],100,1)==1
