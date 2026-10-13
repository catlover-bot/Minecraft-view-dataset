"""CPU-only contracts; injected transports never contact either provider."""
import io
import json
import urllib.error

import pytest

from offline_evaluation.resmoke_v2 import adapter, schema
from offline_evaluation.server_v2 import core, providers
from offline_evaluation.server_v2.generation import object_hash


def envelope(provider, text='{"operations":[]}', *, stop=None, refusal=False):
    if provider == "openai":
        block = {"type": "refusal", "refusal": "fixture refusal"} if refusal else {
            "type": "output_text", "text": text}
        return json.dumps({"id": "fixture", "model": "gpt-5-mini-2025-08-07",
                           "status": "incomplete" if stop else "completed",
                           "incomplete_details": {"reason": stop} if stop else None,
                           "output": [{"type": "message", "content": [block]}],
                           "usage": {"input_tokens": 10, "output_tokens": 2}}).encode()
    return json.dumps({"id": "fixture", "model": adapter.MODELS[provider],
                       "stop_reason": "refusal" if refusal else (stop or "end_turn"),
                       "content": [{"type": "text", "text": text}],
                       "usage": {"input_tokens": 10, "output_tokens": 2}}).encode()


def test_native_payload_same_schema_hash_no_input_derived_constraints():
    sentinel = "DO_NOT_LEAK_GT_MATERIAL_8877"
    payloads = {p: adapter.build_payload(p, m, "common prompt", sentinel)
                for p, m in adapter.MODELS.items()}
    left = payloads["openai"]["text"]["format"]
    right = payloads["anthropic"]["output_config"]["format"]
    assert left["strict"] is True
    assert left["type"] == right["type"] == "json_schema"
    assert left["schema"] == right["schema"] == schema.action_schema()
    assert object_hash(left["schema"]) == object_hash(right["schema"]) == schema.schema_sha256()
    assert sentinel not in json.dumps(left)
    assert sentinel not in json.dumps(right)
    left["schema"]["required"].append(sentinel)
    assert sentinel not in json.dumps(schema.action_schema())


@pytest.mark.parametrize("text,stage", [
    ('{"operations":[{"op":"carve","x1":0,"y1":0,"z1":0,"x2":1,"y2":1,"z2":1,"block":"air"}]}', "schema"),
    ('{"operations":[{"op":"set","x":true,"y":0,"z":0,"block":"stone"}]}', "schema"),
    ('{"operations":[{"op":"set","x":0,"x":1,"y":0,"z":0,"block":"stone"}]}', "json"),
    ('```json\n{"operations":[]}\n```', "json"),
    ('[]', "schema"),
    ('{"operations":[],"other":null}', "schema"),
    ('{"operations":[{"op":"SET","x":0,"y":0,"z":0,"block":"stone"}]}', "schema"),
    ('{"operations":[{"op":"set","x":0,"y":0,"z":0,"block":"Stone"}]}', "local"),
    ('{"operations":[{"op":"set","x":100001,"y":0,"z":0,"block":"stone"}]}', "local"),
])
def test_no_schema_repair_and_separate_validation_stages(text, stage):
    if stage == "json":
        with pytest.raises(core.ValidationError):
            core.strict_json(text)
        return
    document = core.strict_json(text)
    if stage == "schema":
        with pytest.raises(core.ValidationError):
            schema.validate_document(document)
        return
    operations = schema.validate_document(document)
    with pytest.raises(core.ValidationError):
        core.validate_actions(operations)


def test_schema_allows_empty_and_final_empty_without_repair():
    for operations in ([], [{"op": "carve", "x1": 0, "y1": 0, "z1": 0,
                              "x2": 4, "y2": 4, "z2": 4}]):
        parsed = schema.validate_document({"operations": operations})
        assert parsed is operations
        assert not core.execute(core.validate_actions(parsed)).voxels
    good = {"operations": [{"op": "set", "x": 0, "y": 0, "z": 0, "block": "stone"}]}
    assert len(core.execute(schema.validate_document(good)).voxels) == 1


@pytest.mark.parametrize("provider", adapter.MODELS)
def test_refusal_and_truncation_are_not_completion(provider):
    response = adapter.parse_response(provider, envelope(provider, refusal=True))
    assert response.failure_kind == "refusal" and not response.complete
    assert response.usage == {"input_tokens": 10, "output_tokens": 2}
    stop = "max_output_tokens" if provider == "openai" else "max_tokens"
    response = adapter.parse_response(provider, envelope(provider, stop=stop))
    assert response.failure_kind == "truncation" and not response.complete
    assert adapter.parse_response(provider, envelope(provider)).complete


@pytest.mark.parametrize("provider", adapter.MODELS)
def test_single_fixture_request_gated_and_unchanged_model(monkeypatch, provider):
    payload = adapter.build_payload(provider, adapter.MODELS[provider], "prompt", "input")
    calls = []
    def transport(request, timeout):
        calls.append((json.loads(request.data), timeout))
        return envelope(provider)
    monkeypatch.setenv(providers.KEY_NAMES[provider], "fixture-not-a-real-key")
    for options in ({}, {"api_allowed": True}, {"permission": True}):
        with pytest.raises(providers.ProviderError, match="api_disabled"):
            adapter.send_request(provider, payload, transport=transport, **options)
    assert not calls
    raw = adapter.send_request(provider, payload, api_allowed=True, permission=True, transport=transport)
    assert adapter.parse_response(provider, raw).complete
    assert calls == [(payload, 60.0)]
    payload["tools"] = [{"type": "unsupported-paid-feature"}]
    with pytest.raises(providers.ProviderError, match="configuration_failure"):
        adapter.send_request(provider, payload, api_allowed=True, permission=True, transport=transport)
    assert len(calls) == 1
    with pytest.raises(providers.ProviderError, match="configuration_failure"):
        adapter.build_payload(provider, "substitute-model", "prompt", "input")


def test_http_schema_rejection_never_includes_body_and_never_retries(monkeypatch):
    body = b'{"error":{"type":"invalid_request_error","message":"output_config.format schema unsupported PRIVATE_SENTINEL"}}'
    calls = []
    class Opener:
        def open(self, request, timeout):
            calls.append(1)
            raise urllib.error.HTTPError("https://fixture.invalid", 400, "bad", {}, io.BytesIO(body))
    monkeypatch.setattr(adapter.urllib.request, "build_opener", lambda *args: Opener())
    with pytest.raises(providers.ProviderError) as caught:
        adapter.http_transport(object(), 1)
    assert caught.value.kind == "schema_unsupported"
    assert caught.value.status_code == 400
    assert "PRIVATE_SENTINEL" not in str(caught.value)
    assert calls == [1]
    assert adapter.classify_http_error(401, body) == "auth_failure"
    assert adapter.classify_http_error(402, body) == "billing_failure"
    assert adapter.classify_http_error(400, b"not json") == "http_failure"


@pytest.mark.parametrize("provider", adapter.MODELS)
def test_missing_usage_is_unknown_and_never_zero(provider):
    raw = json.loads(envelope(provider))
    raw.pop("usage")
    response = adapter.parse_response(provider, json.dumps(raw).encode())
    with pytest.raises(providers.ProviderError, match="unknown_usage"):
        providers.usage_upper_cost(provider, response.usage,
                                   {"input_usd_per_million": 1, "output_usd_per_million": 5})


def test_duplicate_response_envelope_keys_fail_closed():
    with pytest.raises(providers.ProviderError, match="response_parse_failure"):
        adapter.parse_response("openai", b'{"status":"completed","status":"incomplete"}')
