"""Goal015C fixed mini-task and order-preserving single POST adapter.

Construction/validation have no credentials or network access. The production
sender accepts only the six exact pre-frozen bodies; it never sorts or rebuilds
the body sent by urllib. Mock injection belongs to isolated runtime fixtures.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import socket
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from offline_evaluation.experiments.goal015 import adapter as prior
from offline_evaluation.resmoke_v2.schema import action_schema
from offline_evaluation.server_v2.cohort_v4 import _serialize_exact
from offline_evaluation.server_v2 import providers
from offline_evaluation.server_v2.core import strict_json, Limits

TASK_SUMMARY = (
    "Starting from an empty world, return exactly three operations in this order: "
    "first, fill the inclusive cuboid from (0,0,0) to (1,1,1) with stone; second, "
    "carve only the single voxel at (0,0,0), using that same coordinate for both "
    "endpoints; third, set the voxel at (3,0,0) to glass. Leave every other coordinate "
    "empty. Do not add, omit, or repeat operations. Return only the operations JSON object."
)
EXPECTED_DOCUMENT = {"operations": [
    {"op": "fill", "x1": 0, "y1": 0, "z1": 0, "x2": 1, "y2": 1, "z2": 1, "block": "stone"},
    {"op": "carve", "x1": 0, "y1": 0, "z1": 0, "x2": 0, "y2": 0, "z2": 0},
    {"op": "set", "x": 3, "y": 0, "z": 0, "block": "glass"},
]}
CONDITIONS = ("NATIVE_CURRENT_ORDER", "NATIVE_ORIGINAL_PROPERTY_ORDER",
              "TEXT_JSON_INSTRUCTION_ONLY")
MODELS = prior.MODELS
MAX_OUTPUT_TOKENS = 32768
TIMEOUT = 600
ENDPOINTS = dict(providers.ENDPOINTS)
ProviderError = prior.ProviderError
TransportError = prior.TransportError
parse_response = prior.parse_response
classify_usage = prior.classify_usage
validate_output = prior.validate_output
fixed_prompt = prior.fixed_prompt
canonical_json = prior.canonical_json


def input_text():
    return _serialize_exact({"summary": TASK_SUMMARY})


def wire_bytes(payload):
    """Match the old final serializer, preserving insertion order."""
    return json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256(body):
    return hashlib.sha256(body).hexdigest()


def ordered_representation(value):
    """Type-explicit representation: object enumeration order is significant."""
    if isinstance(value, dict):
        return ["object", [[k, ordered_representation(v)] for k, v in value.items()]]
    if isinstance(value, list):
        return ["array", [ordered_representation(v) for v in value]]
    return [type(value).__name__, value]


def schema_of(builder, payload):
    return payload["text" if builder == "openai" else "output_config"]["format"]["schema"]


def build_payload(builder, condition):
    if builder not in MODELS or condition not in CONDITIONS:
        raise ProviderError("configuration_failure")
    # A reproduces the entire old send-time canonical clone, including all
    # non-properties order. B changes exactly three properties dictionaries.
    payload = json.loads(canonical_json(prior.build_payload(builder, input_text())))
    if condition == CONDITIONS[1]:
        branches = schema_of(builder, payload)["properties"]["operations"]["items"]["anyOf"]
        original = action_schema()["properties"]["operations"]["items"]["anyOf"]
        for branch, old in zip(branches, original, strict=True):
            branch["properties"] = {k: branch["properties"][k] for k in old["properties"]}
    elif condition == CONDITIONS[2]:
        key = "text" if builder == "openai" else "output_config"
        del payload[key]["format"]
        if not payload[key]:
            del payload[key]
    return payload


def order_diff(left, right, path=()):
    """Find value/type/key-order differences, preserving array indices."""
    diffs = []
    if type(left) is not type(right):
        return [{"path": list(path), "kind": "type"}]
    if isinstance(left, dict):
        if set(left) != set(right):
            diffs.append({"path": list(path), "kind": "membership"})
        elif list(left) != list(right):
            diffs.append({"path": list(path), "kind": "object_order"})
        for k in left:
            if k in right:
                diffs += order_diff(left[k], right[k], path + (k,))
    elif isinstance(left, list):
        if len(left) != len(right):
            diffs.append({"path": list(path), "kind": "length"})
        for i, (a, b) in enumerate(zip(left, right)):
            diffs += order_diff(a, b, path + (i,))
    elif left != right:
        diffs.append({"path": list(path), "kind": "value"})
    return diffs


def verify_conditions(builder):
    a, b, c = [build_payload(builder, condition) for condition in CONDITIONS]
    key = "text" if builder == "openai" else "output_config"
    expected = [{"path": [key, "format", "schema", "properties", "operations",
                          "items", "anyOf", i, "properties"], "kind": "object_order"}
                for i in range(3)]
    if canonical_json(a) != canonical_json(b) or order_diff(a, b) != expected:
        raise ProviderError("condition_difference_failure")
    removed = json.loads(wire_bytes(a))
    del removed[key]["format"]
    if not removed[key]:
        del removed[key]
    if wire_bytes(removed) != wire_bytes(c):
        raise ProviderError("condition_difference_failure")
    return {"typed_A_B_equal": True, "allowed_A_B_order_differences": expected,
            "C_format_removal_only": True, "accepted_schema_set_A_B_identical": True}


def validate_wire(builder, condition, body):
    if not isinstance(body, bytes):
        raise ProviderError("wire_mismatch")
    expected = wire_bytes(build_payload(builder, condition))
    if body != expected:
        raise ProviderError("wire_mismatch")
    # Strict duplicate-key rejection as well as exact byte equality.
    strict_json(body.decode("utf-8"), Limits(max_json_bytes=providers.MAX_RESPONSE_BYTES))
    return {"sha256": sha256(body), "bytes": len(body),
            "object_sha256": sha256(canonical_json(json.loads(body)).encode()),
            "ordered_sha256": sha256(canonical_json(ordered_representation(json.loads(body))).encode())}


def _dispatch(request, timeout):
    """Exactly one POST; no redirects/retries, safe metadata only."""
    status = None
    try:
        with urllib.request.build_opener(providers._NoRedirect()).open(request, timeout=timeout) as response:
            status = response.getcode()
            request_id = response.headers.get("x-request-id") or response.headers.get("request-id")
            raw = response.read(providers.MAX_RESPONSE_BYTES + 1)
        if len(raw) > providers.MAX_RESPONSE_BYTES:
            raise TransportError("response_too_large", status, http_response_received=True,
                                 raw_response=raw, raw_complete=False)
        if status != 200:
            raise TransportError("http_failure", status, http_response_received=True,
                                 raw_response=raw, raw_complete=True)
        return {"raw": raw, "http_status": status, "request_id": request_id}
    except urllib.error.HTTPError as exc:
        raw, complete = None, False
        try:
            raw = exc.read(providers.MAX_RESPONSE_BYTES + 1)
            complete = len(raw) <= providers.MAX_RESPONSE_BYTES
        except (OSError, http.client.HTTPException):
            pass
        error = TransportError(prior.classify_http_error(exc.code, raw or b""), exc.code,
                               http_response_received=True, raw_response=raw, raw_complete=complete)
        error.request_id = exc.headers.get("x-request-id") or exc.headers.get("request-id") if exc.headers else None
        raise error from None
    except http.client.IncompleteRead as exc:
        raise TransportError("transport_failure", status, http_response_received=status is not None,
                             raw_response=exc.partial, raw_complete=False) from None
    except (TimeoutError, socket.timeout):
        raise TransportError("timeout", status, http_response_received=status is not None) from None
    except urllib.error.URLError:
        raise TransportError("transport_failure", status, http_response_received=status is not None) from None
    except (OSError, http.client.HTTPException):
        raise TransportError("transport_failure", status, http_response_received=status is not None) from None


def send_wire(builder, condition, wire, *, expected_sha256, expected_bytes,
              api_allowed=False, permission=False, timeout=600, before_send=None):
    if api_allowed is not True or permission is not True:
        raise ProviderError("api_disabled")
    metadata = validate_wire(builder, condition, wire)
    if metadata["sha256"] != expected_sha256 or metadata["bytes"] != expected_bytes or timeout != TIMEOUT:
        raise ProviderError("wire_mismatch")
    endpoint = ENDPOINTS[builder]
    if endpoint != providers.ENDPOINTS[builder] or urlsplit(endpoint).scheme != "https":
        raise ProviderError("endpoint_mismatch")
    key = os.environ.get(providers.KEY_NAMES[builder])
    if not key:
        raise ProviderError("missing_api_key")
    headers = {"Content-Type": "application/json"}
    if builder == "openai":
        headers["Authorization"] = "Bearer " + key
    else:
        headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
    request = urllib.request.Request(endpoint, data=wire, headers=headers, method="POST")
    proof = {"request_data_sha256": sha256(request.data), "request_data_bytes": len(request.data),
             "request_data_equal": request.data == wire, "endpoint_host": urlsplit(endpoint).hostname}
    if proof["request_data_sha256"] != expected_sha256 or not proof["request_data_equal"]:
        raise ProviderError("wire_mismatch")
    if not callable(before_send):
        raise ProviderError("durable_send_proof_required")
    before_send(proof)
    return {**_dispatch(request, timeout), **proof}
