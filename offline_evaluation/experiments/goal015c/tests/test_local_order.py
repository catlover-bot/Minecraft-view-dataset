import copy
import json
from types import SimpleNamespace

import pytest

from offline_evaluation.experiments.goal015c import adapter as new
from offline_evaluation.experiments.goal015c import local_order_audit as audit
from offline_evaluation.resmoke_v2.schema import action_schema, _matches


@pytest.mark.parametrize("builder", new.MODELS)
def test_actual_old_request_data_sorts_properties_but_not_arrays(builder):
    payload = audit.old.build_payload(builder, '{"summary":"Isolated fixture."}')
    before = audit.schema_orders(audit.schema_from_payload(builder, payload))
    body = audit.capture_old_fixture(builder, payload, fixture_only=True)
    after = audit.schema_orders(audit.schema_from_payload(builder, json.loads(body)))
    assert before["anyOf_order"] == after["anyOf_order"] == ["set", "fill", "carve"]
    assert [b["required_order"] for b in before["branches"]] == [b["required_order"] for b in after["branches"]]
    assert [b["properties_order"] for b in after["branches"]] == [
        ["block", "op", "x", "y", "z"],
        ["block", "op", "x1", "x2", "y1", "y2", "z1", "z2"],
        ["op", "x1", "x2", "y1", "y2", "z1", "z2"]]
    assert before["typed_tree_sha256"] == after["typed_tree_sha256"]
    assert before["ordered_representation_sha256"] != after["ordered_representation_sha256"]


def test_old_capture_requires_explicit_fixture_scope():
    with pytest.raises(ValueError, match="isolated fixture"):
        audit.capture_old_fixture("openai", {})


@pytest.mark.parametrize("builder", new.MODELS)
def test_A_B_only_properties_order_and_C_only_format_removal(builder):
    assert new.verify_conditions(builder)["typed_A_B_equal"]
    a, b, c = [new.build_payload(builder, condition) for condition in new.CONDITIONS]
    assert a == b and new.wire_bytes(a) != new.wire_bytes(b)
    assert audit.ordered_representation(a) != audit.ordered_representation(b)
    key = "text" if builder == "openai" else "output_config"
    expected = copy.deepcopy(a)
    del expected[key]
    assert new.wire_bytes(expected) == new.wire_bytes(c)
    assert audit.schema_orders(new.schema_of(builder, b))["branches"] == audit.schema_orders(action_schema())["branches"]


@pytest.mark.parametrize("builder", new.MODELS)
@pytest.mark.parametrize("document,accepted", [
    (new.EXPECTED_DOCUMENT, True), ({"operations": []}, True),
    ({"operations": [new.EXPECTED_DOCUMENT["operations"][1]]}, True),
    ({"operations": [{"op": "set", "x": True, "y": 0, "z": 0, "block": "stone"}]}, False),
    ({"operations": [{"op": "set", "x": 0.0, "y": 0, "z": 0, "block": "stone"}]}, False),
    ({"operations": [], "extra": 1}, False),
    ({"operations": [{"op": "carve", "x1": 0, "y1": 0, "z1": 0, "x2": 0, "y2": 0, "z2": 0, "block": "air"}]}, False),
])
def test_schema_acceptance_unchanged_by_property_order(builder, document, accepted):
    schemas = [new.schema_of(builder, new.build_payload(builder, c)) for c in new.CONDITIONS[:2]]
    assert [_matches(document, s) for s in schemas] == [accepted, accepted]


@pytest.mark.parametrize("builder", new.MODELS)
def test_resorting_B_before_send_is_rejected_before_credentials(builder, monkeypatch):
    b = new.build_payload(builder, new.CONDITIONS[1])
    body = new.wire_bytes(json.loads(new.canonical_json(b)))
    monkeypatch.setattr(new, "os", SimpleNamespace(environ=SimpleNamespace(
        get=lambda *args: pytest.fail("credentials accessed for invalid wire"))))
    with pytest.raises(new.ProviderError, match="wire_mismatch"):
        new.send_wire(builder, new.CONDITIONS[1], body, expected_sha256=new.sha256(body),
                      expected_bytes=len(body), api_allowed=True, permission=True,
                      before_send=lambda proof: pytest.fail("invalid wire accepted"))


@pytest.mark.parametrize("builder", new.MODELS)
@pytest.mark.parametrize("condition", new.CONDITIONS)
def test_new_sender_uses_exact_wire_and_saves_proof_before_transport(builder, condition, monkeypatch):
    body = new.wire_bytes(new.build_payload(builder, condition))
    events = []
    monkeypatch.setattr(new, "os", SimpleNamespace(environ={k: "ISOLATED_FIXTURE_DUMMY"
                                                         for k in audit.providers.KEY_NAMES.values()}))
    def dispatch(req, timeout):
        events.append("transport")
        assert events == ["durable_proof", "transport"]
        assert req.data == body and timeout == 600 and req.full_url == new.ENDPOINTS[builder]
        return {"raw": b"{}", "http_status": 200, "request_id": "fixture"}
    monkeypatch.setattr(new, "_dispatch", dispatch)
    def proof(data):
        assert data["request_data_sha256"] == new.sha256(body) and data["request_data_equal"]
        events.append("durable_proof")
    result = new.send_wire(builder, condition, body, expected_sha256=new.sha256(body),
                           expected_bytes=len(body), api_allowed=True, permission=True, before_send=proof)
    assert result["request_data_equal"] and result["http_status"] == 200


def test_durable_proof_failure_prevents_dispatch(monkeypatch):
    builder, condition = "openai", new.CONDITIONS[1]
    body = new.wire_bytes(new.build_payload(builder, condition))
    monkeypatch.setattr(new, "os", SimpleNamespace(environ={"OPENAI_API_KEY": "ISOLATED_FIXTURE_DUMMY"}))
    monkeypatch.setattr(new, "_dispatch", lambda *args: pytest.fail("sent without durable proof"))
    def fail(proof):
        raise OSError("fixture disk failure")
    with pytest.raises(OSError, match="fixture"):
        new.send_wire(builder, condition, body, expected_sha256=new.sha256(body), expected_bytes=len(body),
                      api_allowed=True, permission=True, before_send=fail)
