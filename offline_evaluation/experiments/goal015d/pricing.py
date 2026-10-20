"""Reuse authenticated Goal015C price evidence without changing acquisition UTC."""
import copy
from ..goal015c import pricing as source

descriptor, write_new, now = source.descriptor, source.write_new, source.now
reservation = source.reservation
capture, seal = source.capture, source.seal


def contract():
    value = copy.deepcopy(source.contract())
    value["schema"] = "goal015d_contract_v1"
    del value["conditions"]["TEXT_JSON_INSTRUCTION_ONLY"]
    return value


def verify_evidence(plan, evidence=None, *, current_time=None):
    source.require(plan["pricing_contract"] == contract(), "Goal015D native A/B contract changed")
    # The source verifier still enforces its entire original contract, actual
    # captured bodies, explicit review, hashes, each document's UTC, and 24h TTL.
    # D uses the exact same provider prices/settings and only its native subset.
    base = {"pricing_contract": source.contract(), "pricing_evidence": plan["pricing_evidence"],
            "fixture_only": plan.get("fixture_only") is True}
    checked = source.verify_evidence(base, evidence, current_time=current_time)
    return {**checked, "goal015d_contract": contract(),
            "reuse_basis": "Unchanged Goal015C provider contract; native A/B subset only; original acquisition UTC retained"}
