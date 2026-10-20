"""Goal015D native A/B transport with unchanged FULL_RECORD input bytes."""
import json
import os
import urllib.request
from urllib.parse import urlsplit

from ..goal015 import adapter as prior
from ..goal015 import planning as input_rules
from ..goal015c import adapter as order
from ...server_v2 import providers
from ...server_v2.core import strict_json, Limits
from ...resmoke_v2.schema import action_schema

CONDITIONS = order.CONDITIONS[:2]
MODELS = order.MODELS
MAX_OUTPUT_TOKENS, TIMEOUT = 32768, 600
ENDPOINTS = dict(order.ENDPOINTS)
ProviderError, TransportError = order.ProviderError, order.TransportError
canonical_json, wire_bytes = order.canonical_json, order.wire_bytes
ordered_representation, order_diff = order.ordered_representation, order.order_diff
sha256, fixed_prompt = order.sha256, order.fixed_prompt
parse_response, classify_usage, validate_output = order.parse_response, order.classify_usage, order.validate_output
_dispatch = order._dispatch


def validate_input(text):
    if not isinstance(text, str):
        raise ProviderError("input_mismatch")
    data = strict_json(text, Limits(max_json_bytes=providers.MAX_RESPONSE_BYTES))
    if not isinstance(data, dict) or set(data) != set(input_rules.FIELDS):
        raise ProviderError("full_record_fields_required")
    if not isinstance(data["summary"], str) or not data["summary"].strip():
        raise ProviderError("full_record_summary_required")
    return data


def input_from_payload(builder, payload):
    if builder == "openai":
        return payload["input"]
    if builder == "anthropic":
        return payload["messages"][0]["content"]
    raise ProviderError("configuration_failure")


def build_payload(builder, condition, input_text):
    if builder not in MODELS or condition not in CONDITIONS:
        raise ProviderError("configuration_failure")
    validate_input(input_text)
    payload = json.loads(canonical_json(prior.build_payload(builder, input_text)))
    if condition == CONDITIONS[1]:
        branches = order.schema_of(builder, payload)["properties"]["operations"]["items"]["anyOf"]
        original = action_schema()["properties"]["operations"]["items"]["anyOf"]
        for branch, old in zip(branches, original, strict=True):
            branch["properties"] = {k: branch["properties"][k] for k in old["properties"]}
    if input_from_payload(builder, payload).encode("utf-8") != input_text.encode("utf-8"):
        raise ProviderError("input_mismatch")
    return payload


def verify_conditions(builder, input_text):
    a, b = [build_payload(builder, c, input_text) for c in CONDITIONS]
    key = "text" if builder == "openai" else "output_config"
    expected = [{"path": [key, "format", "schema", "properties", "operations",
                          "items", "anyOf", i, "properties"], "kind": "object_order"} for i in range(3)]
    if canonical_json(a) != canonical_json(b) or order_diff(a, b) != expected:
        raise ProviderError("condition_difference_failure")
    if input_from_payload(builder, a) != input_from_payload(builder, b):
        raise ProviderError("input_mismatch")
    return {"typed_A_B_equal": True, "allowed_A_B_order_differences": expected,
            "full_input_bytes_equal": True, "accepted_schema_set_A_B_identical": True}


def validate_wire(builder, condition, body, input_text=None):
    if not isinstance(body, bytes):
        raise ProviderError("wire_mismatch")
    payload = strict_json(body.decode("utf-8"), Limits(max_json_bytes=providers.MAX_RESPONSE_BYTES))
    text = input_from_payload(builder, payload)
    if input_text is not None and text != input_text:
        raise ProviderError("input_mismatch")
    if body != wire_bytes(build_payload(builder, condition, text)):
        raise ProviderError("wire_mismatch")
    return {"sha256": sha256(body), "bytes": len(body),
            "input_sha256": sha256(text.encode("utf-8")),
            "object_sha256": sha256(canonical_json(payload).encode()),
            "ordered_sha256": sha256(canonical_json(ordered_representation(payload)).encode())}


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
    if not callable(before_send):
        raise ProviderError("durable_send_proof_required")
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
    before_send(proof)
    return {**_dispatch(request, timeout), **proof}
