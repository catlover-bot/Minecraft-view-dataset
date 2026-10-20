"""Synthetic evidence only; no public GET or credential use in tests."""
import copy
import datetime as dt
from decimal import Decimal
import hashlib
import json

import pytest

from offline_evaluation.experiments.goal015c import adapter, pricing


def evidence(tmp_path):
    docs = {}
    acquired = pricing.now() - dt.timedelta(minutes=1)
    for role, url in pricing.SOURCES.items():
        body = tmp_path / (role + ".body")
        body.write_bytes(b"SYNTHETIC DOCUMENT FIXTURE; NOT ACTUAL CURRENT API EVIDENCE")
        docs[role] = {"requested_url": url, "final_url": url, "http_status": 200,
                      "request_started_utc": acquired.isoformat(),
                      "retrieved_utc": acquired.isoformat(), "body": pricing.descriptor(body)}
    capture = tmp_path/"capture.json"
    review = tmp_path/"review.json"
    pricing.write_new(capture, {"schema": "goal015c_public_capture_v1",
                               "reviewed": False, "request_method": "GET", "documents": docs})
    pricing.write_new(review, {"reviewed": True, "read_actual_bodies": True,
                              "contract": pricing.contract(),
                              "sources": {k: "Synthetic review test, no actual contract verification." for k in docs}})
    path = tmp_path/"evidence.json"
    pricing.seal(capture, review, path)
    plan = {"fixture_only": True, "pricing_evidence": pricing.descriptor(path),
            "pricing_contract": pricing.contract()}
    return plan, path, capture, review


def rewrite(path, data):
    path.write_text(json.dumps(data))


def test_current_and_boundary_expiry(tmp_path):
    plan, path, _, _ = evidence(tmp_path)
    d = pricing.verify_evidence(plan)
    expiry = pricing.date(d["valid_until_utc"])
    pricing.verify_evidence(plan, current_time=expiry-dt.timedelta(microseconds=1))
    with pytest.raises(pricing.EvidenceError, match="expired"):
        pricing.verify_evidence(plan, current_time=expiry)


def test_future_date_rejected(tmp_path):
    plan, path, _, _ = evidence(tmp_path)
    d = json.loads(path.read_text())
    with pytest.raises(pricing.EvidenceError, match="future"):
        pricing.verify_evidence(plan, current_time=pricing.date(d["reviewed_utc"])-dt.timedelta(seconds=1))


def test_same_contract_explicit_renewal_does_not_change_plan(tmp_path):
    first = tmp_path/"first"; first.mkdir()
    second = tmp_path/"second"; second.mkdir()
    plan, _, _, _ = evidence(first)
    before = copy.deepcopy(plan)
    _, renewed, _, _ = evidence(second)
    assert pricing.verify_evidence(plan, renewed)["descriptor"]["path"] == str(renewed)
    assert plan == before


def test_changed_price_requires_new_design(tmp_path):
    plan, path, _, _ = evidence(tmp_path)
    d = json.loads(path.read_text())
    d["contract"]["providers"]["openai"]["input_usd_per_million"] = "0.24"
    rewrite(path, d)
    with pytest.raises(pricing.EvidenceError, match="contract"):
        pricing.verify_evidence(plan, str(path))


def test_real_body_tamper_rejected(tmp_path):
    plan, path, _, _ = evidence(tmp_path)
    d = json.loads(path.read_text())
    p = d["documents"]["openai_model"]["body"]["path"]
    from pathlib import Path
    Path(p).write_bytes(b"changed")
    with pytest.raises(pricing.EvidenceError, match="changed"):
        pricing.verify_evidence(plan)


@pytest.mark.parametrize("field", ["reviewed", "read_actual_bodies"])
def test_capture_without_explicit_read_review_cannot_seal(tmp_path, field):
    _, _, capture, review = evidence(tmp_path)
    d = json.loads(review.read_text()); d[field] = False; rewrite(review, d)
    with pytest.raises(pricing.EvidenceError, match="explicit"):
        pricing.seal(capture, review, tmp_path/"second.json")
    assert not (tmp_path/"second.json").exists()


def test_missing_document_or_nonofficial_url_rejected(tmp_path):
    plan, path, _, _ = evidence(tmp_path)
    d = json.loads(path.read_text())
    d["documents"]["openai_model"]["final_url"] = "https://example.com/fake"
    rewrite(path, d)
    with pytest.raises(pricing.EvidenceError, match="source"):
        pricing.verify_evidence(plan, path)
    del d["documents"]["openai_model"]; rewrite(path, d)
    with pytest.raises(pricing.EvidenceError, match="coverage"):
        pricing.verify_evidence(plan, path)


def test_reservation_is_full_context_and_output_cap():
    assert pricing.reservation("openai") == Decimal("0.165536")
    assert pricing.reservation("anthropic") == Decimal("0.363840")
    assert 6*(pricing.reservation("openai")+pricing.reservation("anthropic")) == Decimal("3.176256")
    assert pricing.contract()["providers"]["openai"]["context_window_tokens"] == 400000
    assert pricing.contract()["providers"]["openai"]["documented_max_input_tokens"] == 272000


def test_fixture_does_not_skip_contract_or_freshness(tmp_path):
    plan, path, _, _ = evidence(tmp_path)
    plan["pricing_contract"]["conditions"][adapter.CONDITIONS[2]]["native"] = True
    with pytest.raises(pricing.EvidenceError, match="contract"):
        pricing.verify_evidence(plan)


def test_redirect_official_allowlist():
    redirect = pricing.PublicRedirect()
    with pytest.raises(pricing.EvidenceError, match="redirect"):
        redirect.redirect_request(None, None, 302, "", {}, "https://example.com/stolen")


def test_contract_hash_and_claimed_expiry_bound_to_actual_capture(tmp_path):
    plan, path, _, _ = evidence(tmp_path)
    d = json.loads(path.read_text())
    d["valid_until_utc"] = (pricing.now()+dt.timedelta(days=5)).isoformat()
    rewrite(path, d)
    with pytest.raises(pricing.EvidenceError, match="expired"):
        pricing.verify_evidence(plan, path)
    d["contract_sha256"] = "0"*64; rewrite(path, d)
    with pytest.raises(pricing.EvidenceError, match="hash"):
        pricing.verify_evidence(plan, path)


@pytest.mark.parametrize("change", ["future_start", "future_retrieved", "insecure", "userinfo", "missing"])
def test_seal_and_verify_check_every_document(tmp_path, change):
    plan, path, capture, review = evidence(tmp_path)
    cap = json.loads(capture.read_text())
    doc = cap["documents"]["anthropic_model"]
    if change == "future_start":
        doc["request_started_utc"] = (pricing.now()+dt.timedelta(hours=1)).isoformat()
    elif change == "future_retrieved":
        doc["retrieved_utc"] = (pricing.now()+dt.timedelta(hours=1)).isoformat()
    elif change == "insecure":
        doc["final_url"] = doc["final_url"].replace("https:", "http:")
    elif change == "userinfo":
        doc["final_url"] = doc["final_url"].replace("https://", "https://user@")
    else:
        del cap["documents"]["anthropic_model"]
    rewrite(capture, cap)
    with pytest.raises(pricing.EvidenceError):
        pricing.seal(capture, review, tmp_path/"invalid.json")
    assert not (tmp_path/"invalid.json").exists()
    value = json.loads(path.read_text()); value["documents"] = cap["documents"]; rewrite(path, value)
    with pytest.raises(pricing.EvidenceError):
        pricing.verify_evidence(plan, path)
