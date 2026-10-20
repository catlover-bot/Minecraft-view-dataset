import copy
import datetime as dt
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from offline_evaluation.experiments.goal015d import adapter, pricing


def full_input():
    return json.dumps({"summary": "Synthetic FULL fixture", "building_type": "fixture",
                       "shape": "cube", "dimensions_estimate": {"x": 2, "y": 1, "z": 2},
                       "materials": ["stone"], "elements": [], "rebuild_hints": [],
                       "uncertainties": ["original fixture uncertainty"]},
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@pytest.mark.parametrize("builder", adapter.MODELS)
def test_full_input_exact_AB_only_three_property_orders(builder):
    text = full_input()
    evidence = adapter.verify_conditions(builder, text)
    assert evidence["full_input_bytes_equal"]
    a, b = [adapter.build_payload(builder, c, text) for c in adapter.CONDITIONS]
    assert adapter.canonical_json(a) == adapter.canonical_json(b)
    assert len(adapter.order_diff(a, b)) == 3
    assert adapter.wire_bytes(a) != adapter.wire_bytes(b)
    assert adapter.input_from_payload(builder, a).encode() == text.encode()
    assert adapter.input_from_payload(builder, b).encode() == text.encode()


@pytest.mark.parametrize("builder", adapter.MODELS)
def test_resorting_B_rejected_before_key_read(builder, monkeypatch):
    class NoKeys:
        def get(self, *args):
            raise AssertionError("credentials accessed before wire validation")
    monkeypatch.setattr(adapter, "os", SimpleNamespace(environ=NoKeys()))
    body = adapter.canonical_json(adapter.build_payload(builder, adapter.CONDITIONS[1], full_input())).encode()
    with pytest.raises(adapter.ProviderError):
        adapter.send_wire(builder, adapter.CONDITIONS[1], body,
                          expected_sha256=adapter.sha256(body), expected_bytes=len(body),
                          api_allowed=True, permission=True, before_send=lambda proof: None)


@pytest.mark.parametrize("builder", adapter.MODELS)
@pytest.mark.parametrize("condition", adapter.CONDITIONS)
def test_wire_direct_to_Request_data_and_durable_proof(builder, condition, monkeypatch):
    body = adapter.wire_bytes(adapter.build_payload(builder, condition, full_input()))
    events = []
    monkeypatch.setattr(adapter, "os", SimpleNamespace(environ={
        "OPENAI_API_KEY": "isolated-fixture-only", "ANTHROPIC_API_KEY": "isolated-fixture-only"}))
    def transport(request, timeout):
        assert events == ["durable"]
        assert request.data is body and timeout == 600
        assert request.full_url == adapter.ENDPOINTS[builder]
        events.append("transport")
        return {"raw": b"{}", "http_status": 200}
    monkeypatch.setattr(adapter, "_dispatch", transport)
    def record(proof):
        assert proof["request_data_equal"]
        assert proof["request_data_sha256"] == adapter.sha256(body)
        assert proof["request_data_bytes"] == len(body)
        assert not any("key" in k or "header" in k for k in proof)
        events.append("durable")
    adapter.send_wire(builder, condition, body, expected_sha256=adapter.sha256(body),
                      expected_bytes=len(body), api_allowed=True, permission=True, before_send=record)
    assert events == ["durable", "transport"]


def test_non_native_and_summary_only_refused():
    with pytest.raises(adapter.ProviderError):
        adapter.build_payload("openai", "TEXT_JSON_INSTRUCTION_ONLY", full_input())
    with pytest.raises(adapter.ProviderError):
        adapter.build_payload("openai", adapter.CONDITIONS[0], '{"summary":"only"}')


def test_wire_must_match_bound_FULL_input():
    text = full_input()
    body = adapter.wire_bytes(adapter.build_payload("openai", adapter.CONDITIONS[0], text))
    with pytest.raises(adapter.ProviderError):
        adapter.validate_wire("openai", adapter.CONDITIONS[0], body, text.replace("cube", "pavilion"))


def fixture_evidence(directory):
    directory.mkdir()
    source = pricing.source
    acquired = source.now() - dt.timedelta(minutes=2)
    documents = {}
    for role, url in source.SOURCES.items():
        path = directory/(role+".body")
        path.write_text("Synthetic evidence fixture; not an actual document review.")
        documents[role] = {"requested_url": url, "final_url": url, "http_status": 200,
                           "request_started_utc": acquired.isoformat(), "retrieved_utc": acquired.isoformat(),
                           "body": source.descriptor(path)}
    capture = directory/"capture.json"; review = directory/"review.json"; evidence = directory/"evidence.json"
    source.write_new(capture, {"schema": "goal015c_public_capture_v1", "reviewed": False,
                              "request_method": "GET", "documents": documents})
    source.write_new(review, {"reviewed": True, "read_actual_bodies": True, "contract": source.contract(),
                             "sources": {k: "Explicit synthetic fixture review only." for k in documents}})
    source.seal(capture, review, evidence)
    return evidence


def test_C_evidence_reused_without_dates_or_contract_changes(tmp_path):
    path = fixture_evidence(tmp_path/"first")
    data = json.loads(path.read_text())
    plan = {"pricing_contract": pricing.contract(), "pricing_evidence": pricing.descriptor(path), "fixture_only": True}
    before = copy.deepcopy(plan)
    checked = pricing.verify_evidence(plan)
    assert plan == before and checked["documents"] == data["documents"]
    assert checked["valid_until_utc"] == data["valid_until_utc"]
    assert set(plan["pricing_contract"]["conditions"]) == set(adapter.CONDITIONS)
    assert checked["contract"] == pricing.source.contract()
    renewed = fixture_evidence(tmp_path/"renewed")
    assert pricing.verify_evidence(plan, renewed)["descriptor"]["path"] == str(renewed)
    assert plan == before


def test_D_contract_mutation_and_expired_C_evidence_refused(tmp_path):
    path = fixture_evidence(tmp_path/"first")
    plan = {"pricing_contract": pricing.contract(), "pricing_evidence": pricing.descriptor(path), "fixture_only": True}
    checked = pricing.verify_evidence(plan)
    with pytest.raises(pricing.source.EvidenceError):
        pricing.verify_evidence(plan, current_time=pricing.source.date(checked["valid_until_utc"]))
    plan["pricing_contract"]["conditions"]["TEXT_JSON_INSTRUCTION_ONLY"] = {"native": False}
    with pytest.raises(pricing.source.EvidenceError):
        pricing.verify_evidence(plan)


def test_sixteen_full_context_reservations_are_separate():
    assert 8*pricing.reservation("openai") == Decimal("1.324288")
    assert 8*pricing.reservation("anthropic") == Decimal("2.910720")
    assert 8*(pricing.reservation("openai")+pricing.reservation("anthropic")) == Decimal("4.235008")
    assert pricing.contract()["providers"] == pricing.source.contract()["providers"]
