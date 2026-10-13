"""Predeclared paired v4 inference with buildings as the independent units.

One row is a final input outcome, not an API attempt. Retries never increase n.
Failed MAE/repair values remain null. Only definitive attempted failures receive
zero in the companion bounded-IoU analysis; planned/unmeasurable cases do not.
"""
from __future__ import annotations

import math
from pathlib import Path
from statistics import mean

from .cohort_v4 import CounterRNG, SEED
from .statistics import holm_adjust

CONDITIONS = ("O", "J", "P", "E")
PROVIDERS = ("openai", "anthropic")
PRIMARY_METRICS = ("aligned_occupancy_iou", "aligned_dimension_mae")
BOUNDED_IOU = {"aligned_occupancy_iou", "absolute_occupancy_iou", "aligned_material_aware_iou", "absolute_material_aware_iou"}
SECONDARY_METRICS = ("absolute_occupancy_iou", "absolute_material_set_f1", "aligned_material_set_f1",
                     "absolute_material_aware_iou", "aligned_material_aware_iou",
                     "absolute_add_count", "absolute_delete_count", "absolute_replace_count",
                     "aligned_add_count", "aligned_delete_count", "aligned_replace_count",
                     "absolute_total_normalized_repair_cost", "aligned_total_normalized_repair_cost")
DEFINITIVE_OUTPUT_FAILURES = {"parse_failure", "validation_failure", "execution_failure", "truncation"}
NEVER_FAILURE_ZERO = {"NOT_RUN", "not_run", "budget_stop", "budget_exhausted", "MISSING", "missing_input",
                      "GT_MISSING", "ALIGNMENT_RESOURCE_LIMIT", "reserved", "pending", "infeasible"}


def comparison_families():
    return [
        {"id": "primary_builder_valid_pair", "size": 8, "comparisons": [
            {"id": f"anthropic-minus-openai:{c}:{m}", "a_provider": "openai", "b_provider": "anthropic",
             "a_condition": c, "b_condition": c, "metric": m} for c in CONDITIONS for m in PRIMARY_METRICS],
         "Holm_scope": "all eight predeclared provider contrasts; nonestimable p-values retained as null, counted as1 for correction"},
        {"id": "secondary_lossless_J_minus_O", "size": 4, "comparisons": [
            {"id": f"J-minus-O:{p}:{m}", "a_provider": p, "b_provider": p,
             "a_condition": "O", "b_condition": "J", "metric": m} for p in PROVIDERS for m in PRIMARY_METRICS],
         "Holm_scope": "all four predeclared within-provider serialization contrasts"},
        {"id": "diagnostic_oracle_minus_original", "size": 8, "comparisons": [
            {"id": f"{c}-minus-O:{p}:{m}", "a_provider": p, "b_provider": p,
             "a_condition": "O", "b_condition": c, "metric": m} for p in PROVIDERS for c in ("P", "E") for m in PRIMARY_METRICS],
         "Holm_scope": "all eight GT-diagnostic contrasts; no pure language-information-loss interpretation"},
    ]


def _quantile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (index - lo) * (ordered[hi] - ordered[lo])


def paired_building_inference(pairs, *, seed=SEED, bootstrap_replicates=2000, signflip_replicates=10000, domain="paired"):
    """Average matched rows within building, then equally weight buildings."""
    if type(bootstrap_replicates) is not int or bootstrap_replicates <= 0 or type(signflip_replicates) is not int or signflip_replicates <= 0:
        raise ValueError("positive replicate counts required")
    clusters = {}
    for pair in pairs:
        a, b = pair["a"], pair["b"]
        if type(a) not in (int, float) or type(b) not in (int, float) or not math.isfinite(a) or not math.isfinite(b):
            raise ValueError("paired inferential values must be finite")
        clusters.setdefault(pair["scene_id"], []).append(b-a)
    values = [mean(clusters[s]) for s in sorted(clusters)]
    result = {"buildings": len(values), "paired_rows": len(pairs), "mean_delta": mean(values) if values else None,
              "estimand": "equal-weight building mean of within-building matched B-A differences",
              "seed": seed, "rng_implementation": "sha256-counter-rejection-fisher-yates-v1",
              "bootstrap_replicates": bootstrap_replicates, "signflip_replicates": signflip_replicates}
    if len(values) < 2:
        return dict(result, status="INSUFFICIENT_CLUSTERS", ci95=None, p_two_sided=None)
    rng = CounterRNG(seed, "bootstrap:" + domain)
    draws = [mean(values[rng.randbelow(len(values))] for _ in values) for _ in range(bootstrap_replicates)]
    observed = abs(mean(values))
    if not any(values):
        p = 1.0
    elif len(values) <= 16:
        # Independent exact sign enumeration on building means.
        exceed = sum(abs(mean(value if (bits >> i) & 1 else -value for i, value in enumerate(values))) >= observed - 1e-15
                     for bits in range(2**len(values)))
        p = exceed / 2**len(values)
    else:
        rng = CounterRNG(seed, "signflip:" + domain)
        exceed = sum(abs(mean(v if rng.randbelow(2) else -v for v in values)) >= observed - 1e-15
                     for _ in range(signflip_replicates))
        p = (exceed + 1) / (signflip_replicates + 1)
    return dict(result, status="success", ci95=[_quantile(draws, .025), _quantile(draws, .975)],
                p_two_sided=p, method="paired building percentile bootstrap and two-sided building sign-flip",
                signflip_method="exact" if len(values) <= 16 else "Monte Carlo plus-one")


def _metric(row, name):
    metrics = row.get("metrics")
    if row["status"] != "success":
        if metrics is not None:
            raise ValueError("failed/unexecuted metrics must be null, never favorable zeros")
        return None
    if not isinstance(metrics, dict):
        return None
    frame, field = name.split("_", 1)
    block = metrics.get(frame)
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError("metric frame must be an object or null")
    value = block.get(field)
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("non-finite or untyped metric")
    if (name in BOUNDED_IOU or field == "material_set_f1") and not 0 <= value <= 1:
        raise ValueError("bounded overlap/F1 must be inside [0,1]")
    if value < 0:
        raise ValueError("metrics must be nonnegative")
    return value


def attempted_failure(row):
    """Only identified, completed request outcomes with model-output failures.

    A transport attempt ID alone does not prove a model produced an output.
    Authentication, context, unavailable-model and unknown-usage outcomes are
    reported separately and never silently converted to model failure0.
    """
    return (row["status"] not in NEVER_FAILURE_ZERO and row["status"] != "success" and
            (row.get("attempted") is True or row.get("attempt_id") is not None) and
            row.get("completed_request_outcome") is True and
            row.get("failure_kind") in DEFINITIVE_OUTPUT_FAILURES and
            row["status"] in {"failed", *DEFINITIVE_OUTPUT_FAILURES})


def bounded_all_attempted_value(row, metric="aligned_occupancy_iou"):
    if metric not in BOUNDED_IOU:
        raise ValueError("failure0 is permitted only for bounded IoU")
    value = _metric(row, metric)
    return value if value is not None else 0.0 if attempted_failure(row) else None


def _validated(rows):
    result, seen = [], set()
    for original in rows:
        row = dict(original)
        if row.get("provider") not in PROVIDERS or row.get("condition") not in CONDITIONS or not isinstance(row.get("scene_id"), str) or not isinstance(row.get("status"), str):
            raise ValueError("new v4 provider/condition/scene/status required; old v5 results are not compatible")
        if row["condition"] in {"P", "E"} and row.get("description_model") is not None:
            raise ValueError("Oracle outcomes must not be replicated by description model")
        if row["condition"] in {"O", "J"} and row.get("description_model") not in {"openai", "anthropic", "gemini"}:
            raise ValueError("ordinary outcomes require saved description provider")
        key = (row["provider"], row["scene_id"], row["condition"], row.get("description_model"))
        if key in seen:
            raise ValueError("duplicate final input outcome; retries must be collapsed, not pooled")
        seen.add(key)
        if row["status"] != "success" and row.get("metrics") is not None:
            raise ValueError("failed/unexecuted metrics must be null")
        if row["status"] == "success" and isinstance(row.get("metrics"), dict):
            if row["metrics"].get("metric_version") != "server_v2.2_strict_translation":
                raise ValueError("new v4 analysis requires server_v2.2_strict_translation metrics; bounded/unknown versions cannot be relabelled")
        if row.get("provenance") is not None and row["provenance"] not in {"new_implementation", "new_v4_two_builder_experiment"}:
            raise ValueError("historical Builder provenance cannot be reused as new v4 outcomes")
        result.append(row)
    return result


def _paired(rows, comparison, all_attempted=False):
    # O/J are paired by description provider. Oracle-versus-O collapses the
    # available O models to one building value, keeping Oracle n at one.
    sides = []
    for prefix in ("a", "b"):
        selected = [r for r in rows if r["provider"] == comparison[prefix+"_provider"] and r["condition"] == comparison[prefix+"_condition"]]
        values = {}
        for row in selected:
            value = bounded_all_attempted_value(row, comparison["metric"]) if all_attempted else _metric(row, comparison["metric"])
            if value is not None:
                values[(row["scene_id"], row.get("description_model"))] = value
        sides.append(values)
    a, b = sides
    if comparison["a_condition"] in {"O", "J"} and comparison["b_condition"] in {"O", "J"}:
        keys = sorted(set(a) & set(b))
        return [{"scene_id": s, "description_model": m, "a": a[(s,m)], "b": b[(s,m)]} for s,m in keys]
    scene_ids = sorted({s for s,m in a} & {s for s,m in b})
    return [{"scene_id": scene, "a": mean(v for (s,m),v in a.items() if s == scene),
             "b": mean(v for (s,m),v in b.items() if s == scene)} for scene in scene_ids]


def _descriptive_metric(rows, metric):
    by_building = {}
    count = 0
    for row in rows:
        value = _metric(row, metric)
        if value is not None:
            by_building.setdefault(row["scene_id"], []).append(value)
            count += 1
    return {"valid_rows": count, "valid_buildings": len(by_building),
            "equal_building_mean": mean(mean(v) for v in by_building.values()) if by_building else None}


def _descriptive_pair(rows, comparison):
    pairs = _paired(rows, comparison)
    by_building = {}
    for pair in pairs:
        by_building.setdefault(pair["scene_id"], []).append(pair["b"]-pair["a"])
    return dict(comparison, common_valid_pair_count=len(pairs), common_valid_buildings=len(by_building),
                mean_delta=mean(mean(v) for v in by_building.values()) if by_building else None,
                interpretation="descriptive common-valid paired building mean; no additional significance claim")


def collect_planned_outcomes(directories, *, expected_identities=None):
    """Verified final outcomes for every planned input, including missing scores.

    This is read-only apart from cooperative lock files. Missing evaluations do
    not erase targets, and multiple preparations for one provider are rejected
    rather than pooled. Selection provenance binds full plans to their original
    cohort and planned-input file; enabled smoke copies bind their eight inputs
    directly through the approved generation identity.
    """
    from .cohort_v4 import digest, json_text
    from .generation import _load_run, _verify_artifact, object_hash, read_json, run_lock
    from .io_utils import sha256

    results, providers_seen = [], set()
    for directory in directories:
        directory = Path(directory)
        with run_lock(directory):
            manifest, config, records, _, state = _load_run(directory)
            identity = manifest["identity"]
            provider = identity["provider"]
            identity_hash = object_hash(identity)
            if provider not in PROVIDERS or provider in providers_seen:
                raise ValueError("exactly one immutable Builder preparation per provider is required; duplicate provider rejected")
            providers_seen.add(provider)
            if identity["provenance"] != "new_implementation":
                raise ValueError("new v4 summary requires new_implementation provenance")
            if expected_identities is not None and expected_identities.get(provider) != identity_hash:
                raise ValueError("approved campaign generation identity mismatch")
            if config.provider != provider or config.model != identity["model"] or config.provenance != identity["provenance"]:
                raise ValueError("Builder manifest/config provider/model/provenance mismatch")
            planned = records
            selection_path = directory / "selection.json"
            if selection_path.is_file():
                selection = read_json(selection_path)
                split = selection.get("split")
                if split not in {"initial_smoke", "smoke", "evaluation"}:
                    raise ValueError("invalid frozen selection split")
                planned_path = directory.parent / f"inputs.{split}.json"
                cohort = read_json(directory.parent / "cohort.v4.json")
                cohort_hash = cohort.get("cohort_sha256")
                if (cohort_hash != digest(json_text({k:v for k,v in cohort.items() if k != "cohort_sha256"}).encode()) or
                        selection.get("cohort_sha256") != cohort_hash or
                        selection.get("planned_inputs_sha256") != sha256(planned_path) or
                        selection.get("generation_identity_hash") != identity_hash):
                    raise ValueError("frozen selection/cohort/generation provenance mismatch")
                planned = read_json(planned_path)
                prepared = [r for r in planned if r.get("status") == "prepared"]
                infeasible = [r for r in planned if r.get("status") != "prepared"]
                if (prepared != records or selection.get("prepared_inputs") != len(prepared) or
                        selection.get("planned_inputs") != len(planned) or selection.get("infeasible_inputs") != infeasible):
                    raise ValueError("frozen planned/prepared/infeasible selection mismatch")
            planned_by_id = {}
            for record in planned:
                input_id = record.get("input_id")
                if (not isinstance(input_id, str) or input_id in planned_by_id or
                        record.get("status") not in {"prepared", "infeasible"} or
                        record.get("dataset_version") != "v4"):
                    raise ValueError("unique complete v4 planned-input identities required")
                planned_by_id[input_id] = record
            evaluated = {}
            evaluation_path = directory / "evaluation.json"
            if evaluation_path.is_file():
                evaluation = read_json(evaluation_path)
                if evaluation.get("identity_hash") != identity_hash or evaluation.get("alignment_mode") != "strict_translation":
                    raise ValueError("evaluation must bind the same generation identity and strict translation mode")
                for row in evaluation["rows"]:
                    input_id = row.get("input_id")
                    if input_id not in planned_by_id or input_id in evaluated:
                        raise ValueError("unknown or duplicate evaluation input identity")
                    evaluated[input_id] = row
            for input_id, record in planned_by_id.items():
                attempts = [a for a in state["attempts"] if a["input_id"] == input_id]
                success = next((a for a in attempts if a["status"] == "success"), None)
                last = success or (attempts[-1] if attempts else None)
                completed = bool(last and last.get("response_id") and last.get("response_model") == identity["model"]
                                 and last.get("usage") is not None and "raw.json" in last.get("artifacts", {}))
                status = "infeasible" if record["status"] == "infeasible" else "NOT_RUN"
                if last:
                    status = "EVALUATION_NOT_RUN" if last["status"] == "success" else last["status"]
                outcome = {"input_id":input_id, "scene_id":record["scene_id"], "condition":record["condition"],
                           "description_model":record.get("description_model"), "provider":provider,
                           "provenance":identity["provenance"], "planning_status":record["status"],
                           "generation_identity_hash":identity_hash, "status":status, "metrics":None,
                           "attempt_id":last["attempt_id"] if last else None, "attempted":bool(last),
                           "completed_request_outcome":completed, "failure_kind":last.get("failure_kind") if last else None}
                if record.get("reasons"):
                    outcome["input_infeasibility_reasons"] = record["reasons"]
                if input_id in evaluated:
                    row = evaluated[input_id]
                    if record["status"] != "prepared":
                        raise ValueError("infeasible input cannot have a generated evaluation outcome")
                    for field in ("input_id", "scene_id", "condition", "description_model", "provider", "provenance",
                                  "attempt_id", "attempted", "completed_request_outcome", "failure_kind"):
                        if row.get(field) != outcome.get(field):
                            raise ValueError("evaluation row does not match final planned input/attempt identity")
                    allowed_statuses = {"success", "GT_MISSING", "ALIGNMENT_RESOURCE_LIMIT"} if success else {status}
                    if row.get("status") not in allowed_statuses:
                        raise ValueError("evaluation status does not match authoritative final generation state")
                    if row.get("metrics") is not None or row.get("metrics_artifact") is not None:
                        if not last or last["status"] != "success" or not row.get("metrics_artifact"):
                            raise ValueError("scored evaluation requires a successful final attempt and verified metric artifact")
                        artifact = read_json(_verify_artifact(directory, row["metrics_artifact"]))
                        if (artifact.get("metrics") != row.get("metrics") or
                                artifact.get("run_identity_hash") != identity_hash or
                                artifact.get("attempt_id") != last["attempt_id"] or
                                artifact.get("alignment_mode") != "strict_translation"):
                            raise ValueError("evaluation metric artifact provenance mismatch")
                    outcome.update(row)
                results.append(outcome)
    if providers_seen != set(PROVIDERS):
        raise ValueError("both immutable Builder preparations are required; missing provider cannot be silently omitted")
    return _validated(results)


def summarize_v4(rows, *, seed=SEED, bootstrap_replicates=2000, signflip_replicates=10000, smoke=False):
    rows = _validated(rows)
    groups = []
    for provider in PROVIDERS:
        for condition in CONDITIONS:
            selected = [r for r in rows if r["provider"] == provider and r["condition"] == condition]
            valid = [_metric(r,"aligned_occupancy_iou") for r in selected]
            attempted = [bounded_all_attempted_value(r) for r in selected]
            by_building_valid, by_building_attempted = {}, {}
            for r,v,a in zip(selected,valid,attempted):
                if v is not None: by_building_valid.setdefault(r["scene_id"],[]).append(v)
                if a is not None: by_building_attempted.setdefault(r["scene_id"],[]).append(a)
            failure_kinds = {}
            for row in selected:
                if row.get("failure_kind") is not None:
                    kind = row["failure_kind"]
                    failure_kinds[kind] = failure_kinds.get(kind, 0) + 1
            groups.append({"provider":provider,"condition":condition,"planned_rows":len(selected),
                           "independent_buildings":len({r['scene_id'] for r in selected}),
                           "valid_rows":sum(v is not None for v in valid),
                           "definitive_attempted_failure_rows":sum(attempted_failure(r) for r in selected),
                           "all_attempted_rows":sum(a is not None for a in attempted),
                           "excluded_not_run_or_unmeasurable":sum(a is None for a in attempted),
                           "rows_with_attempt_identity":sum(r.get("attempt_id") is not None or r.get("attempted") is True for r in selected),
                           "operational_or_unmeasurable_attempt_rows":sum(a is None and (r.get("attempt_id") is not None or r.get("attempted") is True) for r,a in zip(selected,attempted)),
                           "failure_kind_counts":failure_kinds,
                           "failure_rate_denominator":sum(a is not None for a in attempted),
                           "failure_rate":sum(attempted_failure(r) for r in selected)/sum(a is not None for a in attempted) if any(a is not None for a in attempted) else None,
                           "valid_only_equal_building_aligned_iou":mean(mean(v) for v in by_building_valid.values()) if by_building_valid else None,
                           "all_attempted_failure0_equal_building_aligned_iou":mean(mean(v) for v in by_building_attempted.values()) if by_building_attempted else None,
                           "valid_only_metrics":{m:_descriptive_metric(selected,m) for m in PRIMARY_METRICS+SECONDARY_METRICS}})
    families = []
    if not smoke:
        for family in comparison_families():
            results = []
            for comparison in family["comparisons"]:
                pairs = _paired(rows,comparison)
                result = dict(comparison, **paired_building_inference(pairs,seed=seed,bootstrap_replicates=bootstrap_replicates,
                                      signflip_replicates=signflip_replicates,domain=family['id']+comparison['id']))
                result["common_valid_pair_count"] = len(pairs)
                if comparison["metric"] in BOUNDED_IOU:
                    companion = paired_building_inference(_paired(rows,comparison,True),seed=seed,
                                    bootstrap_replicates=bootstrap_replicates,signflip_replicates=signflip_replicates,
                                    domain="attempted:"+family['id']+comparison['id'])
                    companion.pop("p_two_sided",None)
                    companion["interpretation"] = "descriptive all-attempted bounded score; no additional significance claim"
                    result["all_attempted_failure0_companion"] = companion
                results.append(result)
            adjusted = holm_adjust([r['p_two_sided'] if r['p_two_sided'] is not None else 1.0 for r in results])
            for result,p in zip(results,adjusted):
                result['p_holm'] = p if result['p_two_sided'] is not None else None
            families.append(dict(id=family['id'],predeclared_size=family['size'],Holm_scope=family['Holm_scope'],results=results))
    descriptive_pairs = []
    if not smoke:
        for family in comparison_families():
            # Each provider/condition contrast appears once; metric-specific
            # common-valid sets may differ and have their own denominators.
            for original in family["comparisons"][::len(PRIMARY_METRICS)]:
                for metric in SECONDARY_METRICS:
                    comparison = dict(original, metric=metric, id=original["id"].rsplit(":",1)[0]+":"+metric)
                    descriptive_pairs.append(_descriptive_pair(rows, comparison))
    return {"provenance":"new_v4_two_builder_experiment","groups":groups,"families":families,
            "secondary_descriptive_common_valid_pairs":descriptive_pairs,
            "smoke_inference_suppressed":smoke,"independent_unit":"building","oracle_replication_per_description":False,
            "evaluation_version":"server_v2.2_strict_translation",
            "strict_score_semantics":"all-translation maximum occupancy IoU score bounded in [0,1]; strict translation search has no finite-radius restriction",
            "failure0_eligibility":"identified completed request outcomes with parse/validation/execution/truncation only; operational/unknown-usage outcomes excluded and separately counted",
            "failure_MAE_repair":"null, common-valid pairs only","research_requests_sent_by_analysis":0}
