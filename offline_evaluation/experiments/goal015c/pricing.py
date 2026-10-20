"""Public-document acquisition, explicit review seal, and 24-hour freshness.

Capture never declares documents reviewed. A separate human/agent-authored
review must identify the read evidence and the exact unchanged contract.
Renewal does not create a plan or modify a ledger/budget.
"""
from __future__ import annotations

import argparse
import datetime as dt
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import urllib.request
from urllib.parse import urlsplit

from .adapter import CONDITIONS, MODELS, ENDPOINTS, MAX_OUTPUT_TOKENS, canonical_json

SOURCES = {
    "openai_model": "https://developers.openai.com/api/docs/models/gpt-5-mini.md",
    "openai_structured": "https://developers.openai.com/api/docs/guides/structured-outputs.md",
    "openai_reasoning": "https://developers.openai.com/api/docs/guides/reasoning.md",
    "openai_gpt5": "https://developers.openai.com/api/docs/guides/gpt-5.md",
    "openai_deprecations": "https://developers.openai.com/api/docs/deprecations.md",
    "openai_responses": "https://developers.openai.com/api/reference/resources/responses/methods/create.md",
    "anthropic_model": "https://platform.claude.com/docs/en/models/haiku-4-5/overview.md",
    "anthropic_pricing": "https://platform.claude.com/docs/en/about-claude/pricing.md",
    "anthropic_structured": "https://platform.claude.com/docs/en/build-with-claude/structured-outputs.md",
    "anthropic_messages": "https://platform.claude.com/docs/en/api/messages/create.md",
    "json_schema": "https://json-schema.org/understanding-json-schema/reference/combining",
}
HOSTS = {"developers.openai.com", "platform.claude.com", "json-schema.org"}


class EvidenceError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise EvidenceError(message)


def now():
    return dt.datetime.now(dt.timezone.utc)


def date(value):
    value = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(value.tzinfo is not None, "timezone required")
    return value


def descriptor(path):
    path = Path(path).absolute()
    require(not path.is_symlink() and path.is_file(), "regular evidence file required")
    body = path.read_bytes()
    return {"path": str(path), "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def check_descriptor(value):
    require(isinstance(value, dict), "descriptor required")
    actual = descriptor(value["path"])
    require(all(actual[k] == value[k] for k in actual), "evidence file changed")
    return actual


def write_new(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def verify_documents(documents, reviewed_at, current_time):
    require(isinstance(documents, dict) and set(documents) == set(SOURCES),
            "incomplete documentation coverage")
    for role, doc in documents.items():
        check_descriptor(doc["body"])
        final = urlsplit(doc["final_url"])
        require(doc["requested_url"] == SOURCES[role] and final.scheme == "https"
                and final.hostname in HOSTS and not final.username and not final.password
                and doc["http_status"] == 200, "unapproved documentation source")
        started, retrieved = date(doc["request_started_utc"]), date(doc["retrieved_utc"])
        require(started <= retrieved <= reviewed_at <= current_time,
                "documentation timestamps inconsistent or future dated")
    return min(date(d["request_started_utc"]) for d in documents.values())


def contract():
    """Fixed diagnostic contract. If current docs disagree, do not seal it."""
    return {
        "schema": "goal015c_contract_v1",
        "providers": {
            "openai": {"model": MODELS["openai"], "endpoint": ENDPOINTS["openai"],
                       "context_window_tokens": 400000, "max_output_tokens": MAX_OUTPUT_TOKENS,
                       "documented_max_input_tokens": 272000,
                       "documented_max_output_tokens": 128000, "input_usd_per_million": "0.25",
                       "output_usd_per_million": "2.00", "service_tier": "default",
                       "reasoning_effort": "medium", "sampling_omitted": ["temperature", "top_p", "seed"],
                       "documented_shutdown_date": "2026-12-11"},
            "anthropic": {"model": MODELS["anthropic"], "endpoint": ENDPOINTS["anthropic"],
                          "context_window_tokens": 200000, "max_output_tokens": MAX_OUTPUT_TOKENS,
                          "documented_max_output_tokens": 64000, "input_usd_per_million": "1.00",
                          "output_usd_per_million": "5.00", "service_tier": "standard_only",
                          "temperature": 1.0, "thinking": "disabled"},
        },
        "conditions": {CONDITIONS[0]: {"native": True, "properties_order": "sorted"},
                       CONDITIONS[1]: {"native": True, "properties_order": "original"},
                       CONDITIONS[2]: {"native": False, "text_output": True,
                                       "format": None, "JSON_mode": False}},
        "timeout_seconds": 600, "concurrency": 1, "retries": 0,
        "cache_discount_applied": False, "tools": False, "images": False,
        "reservation": "full_context_input_plus_output_cap",
        "fees_scope": "standard first-party token prices; no service upgrades or paid tools",
    }


def reservation(builder, terms=None):
    p = (terms or contract())["providers"][builder]
    return (Decimal(p["context_window_tokens"]) * Decimal(p["input_usd_per_million"])
            + Decimal(p["max_output_tokens"]) * Decimal(p["output_usd_per_million"])) / Decimal(1_000_000)


class PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parts = urlsplit(newurl)
        require(parts.scheme == "https" and parts.hostname in HOSTS and not parts.username,
                "non-official redirect refused")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def capture(output):
    output = Path(output).absolute()
    output.mkdir(parents=False, exist_ok=False)
    documents = {}
    opener = urllib.request.build_opener(PublicRedirect())
    for role, url in SOURCES.items():
        started = now().isoformat()
        with opener.open(urllib.request.Request(url, method="GET",
                         headers={"User-Agent": "Goal015C-public-contract-review/1.0"}), timeout=30) as response:
            body = response.read(12_000_001)
            require(len(body) <= 12_000_000, "documentation body limit exceeded")
            require(response.status == 200, "documentation GET failed")
            final = response.url
            require(urlsplit(final).hostname in HOSTS, "unexpected documentation host")
        p = output / (role + ".body")
        with p.open("xb") as stream:
            stream.write(body)
        documents[role] = {"requested_url": url, "final_url": final, "retrieved_utc": now().isoformat(),
                           "request_started_utc": started, "http_status": 200, "body": descriptor(p)}
        write_new(output / (role + ".receipt.json"), documents[role])
    result = {"schema": "goal015c_public_capture_v1", "reviewed": False, "documents": documents,
              "research_api_requests": 0, "request_method": "GET"}
    write_new(output / "CAPTURE.json", result)
    return result


def seal(capture_path, review_path, output):
    capture_doc = json.loads(Path(capture_path).read_text())
    review = json.loads(Path(review_path).read_text())
    require(capture_doc["schema"] == "goal015c_public_capture_v1", "wrong capture")
    require(capture_doc.get("request_method") == "GET" and capture_doc.get("reviewed") is False,
            "raw public GET capture required")
    require(review.get("reviewed") is True and review.get("read_actual_bodies") is True,
            "explicit actual-body review required; capture alone is not reviewed")
    require(review.get("contract") == contract(), "contract changed; new design required")
    require(set(review.get("sources", {})) == set(SOURCES), "review must cover every document")
    reviewed = now()
    acquired = verify_documents(capture_doc["documents"], reviewed, reviewed)
    for role in capture_doc["documents"]:
        note = review["sources"][role]
        require(isinstance(note, str) and len(note) >= 20, "substantive reviewed evidence note required")
    require(acquired <= reviewed < acquired + dt.timedelta(hours=24), "capture expired before seal")
    result = {"schema": "goal015c_reviewed_evidence_v1", "reviewed": True,
              "capture": descriptor(capture_path), "review": descriptor(review_path),
              "documents": capture_doc["documents"], "contract": contract(),
              "contract_sha256": hashlib.sha256(canonical_json(contract()).encode()).hexdigest(),
              "reviewed_utc": reviewed.isoformat(),
              "valid_until_utc": (acquired + dt.timedelta(hours=24)).isoformat()}
    write_new(output, result)
    return result


def verify_evidence(plan, evidence=None, *, current_time=None):
    value = evidence or plan["pricing_evidence"]
    path = value["path"] if isinstance(value, dict) else str(value)
    if isinstance(value, dict):
        check_descriptor(value)
    data = json.loads(Path(path).read_text())
    require(data.get("schema") == "goal015c_reviewed_evidence_v1" and data.get("reviewed") is True,
            "reviewed pricing evidence required")
    require(data.get("contract") == plan["pricing_contract"] == contract(), "pricing/model contract changed")
    require(data["contract_sha256"] == hashlib.sha256(canonical_json(contract()).encode()).hexdigest(),
            "contract hash mismatch")
    fixture = plan.get("fixture_only") is True
    if not fixture:
        check_descriptor(data["capture"]); check_descriptor(data["review"])
        capture_doc = json.loads(Path(data["capture"]["path"]).read_text())
        review = json.loads(Path(data["review"]["path"]).read_text())
        require(review.get("reviewed") is True and review.get("read_actual_bodies") is True
                and review.get("contract") == contract(), "review seal mismatch")
        require(data["documents"] == capture_doc["documents"], "capture binding mismatch")
    t = current_time or now()
    acquired = verify_documents(data["documents"], date(data["reviewed_utc"]), t)
    expiry = acquired + dt.timedelta(hours=24)
    require(acquired <= date(data["reviewed_utc"]) <= t < expiry
            and date(data["valid_until_utc"]) == expiry, "pricing evidence expired or future dated")
    require(t.date().isoformat() < contract()["providers"]["openai"]["documented_shutdown_date"],
            "documented model shutdown reached; no fallback")
    return {**data, "descriptor": descriptor(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture"); cap.add_argument("--output", required=True)
    se = sub.add_parser("seal")
    for name in ("capture", "review", "output"):
        se.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.command == "capture":
        result = capture(args.output)
        print(json.dumps({"reviewed": False, "documents": len(result["documents"])}))
    else:
        result = seal(args.capture, args.review, args.output)
        print(json.dumps({"valid_until_utc": result["valid_until_utc"]}))


if __name__ == "__main__":
    main()
