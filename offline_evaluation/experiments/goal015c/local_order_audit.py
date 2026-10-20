"""Local, order-aware schema provenance. Never opens a live HTTP connection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import resource
import socket
import time
from types import SimpleNamespace
from unittest.mock import patch

from offline_evaluation.experiments.goal015 import adapter as old
from offline_evaluation.resmoke_v2 import adapter as native
from offline_evaluation.resmoke_v2.schema import action_schema
from offline_evaluation.server_v2 import providers
from offline_evaluation.server_v2.generation import object_hash

COMPARISON = Path("/root/minecraft-runs/goal015b-20261020T090529Z-5f01c85f/phase4/COMPARISON_REPLAY.json")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def descriptor(path):
    path = Path(path)
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": sha(raw)}


def ordered_representation(value):
    """Object order is semantic to this representation; scalar types are tagged."""
    if isinstance(value, dict):
        return ["object", [[k, ordered_representation(v)] for k, v in value.items()]]
    if isinstance(value, list):
        return ["array", [ordered_representation(v) for v in value]]
    return [type(value).__name__, value]


def schema_from_payload(builder, payload):
    return payload["text"]["format"]["schema"] if builder == "openai" else payload["output_config"]["format"]["schema"]


def schema_orders(schema):
    branches = schema["properties"]["operations"]["items"]["anyOf"]
    return {
        "anyOf_order": [b["properties"]["op"]["enum"][0] for b in branches],
        "branches": [{"operation": b["properties"]["op"]["enum"][0],
                      "properties_order": list(b["properties"]),
                      "required_order": b["required"]} for b in branches],
        "ordered_representation_sha256": sha(json.dumps(ordered_representation(schema),
                                                       ensure_ascii=False, separators=(",", ":")).encode()),
        "typed_tree_sha256": sha(old.canonical_json(schema).encode()),
    }


def capture_old_fixture(builder, payload, *, fixture_only=False):
    """Exercise the real old adapter -> providers -> Request.data, with dummy auth.

    Credentials are supplied by replacing only providers' os reference, so no
    existing environment value is inspected. No header values are returned.
    """
    if fixture_only is not True:
        raise ValueError("isolated fixture only")
    captured = []
    def transport(request, timeout):
        if request.full_url != providers.ENDPOINTS[builder] or request.get_method() != "POST":
            raise AssertionError("unexpected fixture endpoint/method")
        captured.append(bytes(request.data))
        return b'{"isolated_transport_fixture":true}'
    dummy_os = SimpleNamespace(environ={name: "ISOLATED_FIXTURE_DUMMY" for name in providers.KEY_NAMES.values()})
    with patch.object(providers, "os", dummy_os), \
         patch.object(socket.socket, "connect", side_effect=AssertionError("network prohibited")), \
         patch.object(socket, "getaddrinfo", side_effect=AssertionError("network prohibited")):
        old.send_request(builder, payload, api_allowed=True, permission=True, timeout=600, transport=transport)
    if len(captured) != 1:
        raise AssertionError("fixture requires exactly one Request")
    return captured[0]


def _write(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return descriptor(path)


def run_audit(run_root, comparison_path=COMPARISON):
    """Read only the two already-selected historical cases and current payloads."""
    start, cpu = time.monotonic(), time.process_time()
    root = Path(run_root)
    comparison = json.loads(Path(comparison_path).read_bytes())
    selected = comparison["comparisons"]
    if len(selected) != 2 or {r["builder"] for r in selected} != {"openai", "anthropic"}:
        raise ValueError("requires exactly the two fixed Goal015B comparisons")
    inputs = {str(Path(comparison_path)): descriptor(comparison_path)}
    for row in selected:
        for d in [row["old_original_description"], row["current_full_calls"][0]["payload"]]:
            actual = descriptor(d["path"])
            if actual != d:
                raise ValueError("selected source descriptor mismatch")
            inputs[d["path"]] = actual
    _write(root / "LOCAL_ORDER_INPUTS_BEFORE.json", {"files": list(inputs.values())})
    records = []
    for row in selected:
        builder = row["builder"]
        saved_desc = row["current_full_calls"][0]["payload"]
        saved_bytes = Path(saved_desc["path"]).read_bytes()
        saved = json.loads(saved_bytes)
        input_text = saved["input"] if builder == "openai" else saved["messages"][0]["content"]
        built = old.build_payload(builder, input_text)
        clone = json.loads(old.canonical_json(saved))
        body = json.dumps(clone, ensure_ascii=False, allow_nan=False).encode("utf-8")
        # The capture is synthetic and distinct from the historical body proof.
        fixture = old.build_payload(builder, '{"summary":"Isolated schema ordering fixture."}')
        captured = capture_old_fixture(builder, fixture, fixture_only=True)
        captured_schema = schema_from_payload(builder, json.loads(captured))
        if schema_orders(captured_schema) != schema_orders(schema_from_payload(builder, clone)):
            raise AssertionError("fixture actual Request schema order differs from reconstruction")
        original = action_schema()
        saved_schema = schema_from_payload(builder, saved)
        clone_schema = schema_from_payload(builder, clone)
        if original != saved_schema or original != clone_schema:
            raise AssertionError("typed schema tree changed")
        original_description = Path(row["old_original_description"]["path"]).read_text()
        settings = row["old_requested_settings"]
        past = native.build_payload(builder, settings["model"], old.fixed_prompt(), original_description, 32768)
        if object_hash(past) != row["old_reconstructed_payload_frozen_object_sha256"]:
            raise AssertionError("historical reconstructed object hash differs")
        past_body = json.dumps(past, ensure_ascii=False, allow_nan=False).encode("utf-8")
        historical_schema = schema_from_payload(builder, json.loads(past_body))
        stages = {
            "original_schema_construction": schema_orders(original),
            "goal015_pure_payload_build": schema_orders(schema_from_payload(builder, built)),
            "saved_payload_loaded_without_sorting": schema_orders(saved_schema),
            "goal015_canonical_clone": schema_orders(clone_schema),
            "final_Request_data_reconstructed_from_saved_payload": schema_orders(schema_from_payload(builder, json.loads(body))),
            "isolated_synthetic_actual_Request_data": schema_orders(captured_schema),
            "selected_old_O_reconstructed_body": schema_orders(historical_schema),
        }
        arrays = [(s["anyOf_order"], [b["required_order"] for b in s["branches"]]) for s in stages.values()]
        if not all(a == arrays[0] for a in arrays):
            raise AssertionError("required or anyOf order changed")
        if stages["original_schema_construction"]["ordered_representation_sha256"] == stages["goal015_canonical_clone"]["ordered_representation_sha256"]:
            raise AssertionError("expected local order difference absent")
        records.append({
            "builder": builder, "selected_scene_id": row["scene_id"], "selected_attempt_id": row["selected_attempt_id"],
            "stages": stages, "saved_payload": saved_desc,
            "saved_payload_body_sha256": sha(saved_bytes),
            "reconstructed_goal015_final_body_sha256": sha(body), "reconstructed_goal015_final_body_bytes": len(body),
            "isolated_synthetic_captured_Request_data_sha256": sha(captured),
            "historical_object_hash_matches_selected_evidence": True,
            "historical_reconstructed_object_hash": object_hash(past),
            "historical_reconstructed_body_sha256": sha(past_body), "historical_reconstructed_body_bytes": len(past_body),
            "historical_wire_evidence": "RECONSTRUCTED_FROM_FROZEN_CODE",
            "historical_original_wire_bytes_available": False,
            "typed_schema_tree_equal_at_all_stages": True,
            "required_and_anyOf_array_orders_unchanged": True,
            "output_raw_key_order_used_as_schema_evidence": False,
            "remote_delivery_attested": False,
        })
    for d in inputs.values():
        if descriptor(d["path"]) != d:
            raise ValueError("local audit input changed")
    report = {
        "schema": "goal015c_local_order_audit_v1", "status": "PASS", "records": records,
        "conclusion": "Goal015 canonical clone sorts object keys, including operation properties, while required and anyOf arrays retain their order. This code behavior does not establish why a model generated carve-only output.",
        "credentials_read_from_environment": False, "actual_network_requests": 0, "research_api_requests_sent": 0,
        "parent_responses_reanalyzed": 0, "selected_old_O_cases": 2,
        "resources": {"cpu_seconds": time.process_time()-cpu, "elapsed_seconds": time.monotonic()-start,
                      "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "input_snapshot": descriptor(root / "LOCAL_ORDER_INPUTS_BEFORE.json"),
    }
    return _write(root / "LOCAL_ORDER_AUDIT.json", report)
