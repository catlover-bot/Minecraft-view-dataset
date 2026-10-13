"""Frozen evaluation inputs and CPU planning; never enables or sends a request.

Reuse the saved Goal002 split and O/J/P bodies. Only E's old Z-run body is
replaced by the exact current resmoke cuboid body. No selection uses responses.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
import csv
import json
import statistics
import time

from offline_evaluation.resmoke_v2.cuboids import (
    compact_json, decode_exact, encode_exact, exact_builder_text, reference_actions,
)
from offline_evaluation.resmoke_v2.plan import MODELS, OUTPUT_CAP, PROMPT_PATH
from offline_evaluation.resmoke_v2.schema import action_schema, schema_sha256
from offline_evaluation.server_v2.cohort_v4 import DESCRIPTION_MODELS, canonicalize_saved_text, json_text
from offline_evaluation.server_v2.core import execute, grid_to_payload
from offline_evaluation.server_v2.data import load_scene, sha256_file
from offline_evaluation.server_v2.generation import GenerationError, digest, object_hash, read_json, write_json
from offline_evaluation.server_v2.io_utils import ROOT

PROTOCOL_ID = "goal005_fixed_evaluation60_native_cuboids_batch1_v1"
COHORT_PATH = Path("/root/minecraft-runs/goal002-20261013-141854/cohort.v4.json")
COHORT_FILE_SHA256 = "4015da18a189b597365d412fe21686139c62c809e4349e917b9ee4b971548ff0"
OLD_INPUTS_FILE_SHA256 = "6b4d7ce9e08289940f0dd6b7d95cc396e6f11219e9c8bd63b306d99d845cc846"
ORDER_RULE = ("Within each existing stratum sort evaluation rows by preserved stratum_draw_order, then scene_id; "
              "round-robin strata 0,1,2,3,4 at each draw position. First two rounds form batch1 (10 buildings). "
              "For building ordinal b, rotate [O/openai,J/openai,O/anthropic,J/anthropic,O/gemini,J/gemini,P,E] "
              "left by b modulo 8; alternate OpenAI then Anthropic for every semantic input. "
              "The same first 160 entries are the prefix of the full 960 schedule.")


def ordered_evaluation(cohort: dict) -> list[dict]:
    """Consume the fixed split, without RNG, ranking, redraw or result access."""
    scenes = cohort["scenes"]
    if (len(scenes) != 100 or len({s["scene_id"] for s in scenes}) != 100
            or Counter(s["split"] for s in scenes) != {"smoke": 10, "evaluation": 60, "rest": 30}):
        raise GenerationError("frozen 10/60/30 split required")
    groups = []
    for stratum in range(5):
        group = [s for s in scenes if s["size_rank_stratum"] == stratum]
        if Counter(s["split"] for s in group) != {"smoke": 2, "evaluation": 12, "rest": 6}:
            raise GenerationError("frozen per-stratum 2/12/6 split required")
        evaluation = sorted((s for s in group if s["split"] == "evaluation"),
                            key=lambda s: (s["stratum_draw_order"], s["scene_id"]))
        if sorted(s["stratum_draw_order"] for s in evaluation) != list(range(2, 14)):
            raise GenerationError("preserved evaluation draw positions required")
        groups.append(evaluation)
    return [group[position] for position in range(12) for group in groups]


def _source_path(root: Path, relative: str, expected_sha: str) -> Path:
    path = root / relative
    if (not path.is_relative_to(root) or any(p.is_symlink() for p in (path, *path.parents))
            or not path.resolve().is_relative_to(root.resolve()) or sha256_file(path) != expected_sha):
        raise GenerationError("frozen source changed or escaped: " + relative)
    return path


def _read_fixed_inputs(cohort_path: Path) -> list[dict]:
    loaded = []
    for provider in MODELS:
        directory = cohort_path.parent / ("builder_" + provider + "_evaluation")
        path = directory / "inputs.json"
        if sha256_file(path) != OLD_INPUTS_FILE_SHA256:
            raise GenerationError("frozen old evaluation input file changed")
        records = read_json(path)
        identity = read_json(directory / "manifest.json")["identity"]
        if (identity["inputs_hash"] != object_hash(records)
                or identity["inputs"] != [{"input_id": r["input_id"], "input_hash": object_hash(r)} for r in records]):
            raise GenerationError("old evaluation input manifest mismatch")
        loaded.append(records)
    if loaded[0] != loaded[1]:
        raise GenerationError("old provider input sets disagree")
    return loaded[0]


def _schedule(records: dict, ordered: list[dict]) -> list[dict]:
    schedule = []
    slots = [(condition, model) for model in DESCRIPTION_MODELS for condition in "OJ"] + [("P", None), ("E", None)]
    for ordinal, scene in enumerate(ordered):
        rotation = ordinal % len(slots)
        for condition, description_model in slots[rotation:] + slots[:rotation]:
            record = records[scene["scene_id"], condition, description_model]
            for provider, model in MODELS.items():
                schedule.append({**record, "provider": provider, "model": model, "schedule_index": len(schedule)})
    return schedule


def make_manifest(root=ROOT, cohort_path=COHORT_PATH) -> dict:
    """Create data in memory only; hash check all 60 sources and frozen bodies."""
    root, cohort_path = Path(root).resolve(), Path(cohort_path)
    if sha256_file(cohort_path) != COHORT_FILE_SHA256:
        raise GenerationError("original Goal002 cohort bytes changed; no redraw allowed")
    cohort = read_json(cohort_path)
    if digest(json_text({k: v for k, v in cohort.items() if k != "cohort_sha256"}).encode()) != cohort["cohort_sha256"]:
        raise GenerationError("cohort content hash mismatch")
    ordered = ordered_evaluation(cohort)
    if {"buildings_100_v4_building_060", "buildings_100_v4_building_075"} & {s["scene_id"] for s in ordered}:
        raise GenerationError("smoke buildings must not enter evaluation")
    old_records = _read_fixed_inputs(cohort_path)
    indexed = {(r["scene_id"], r["condition"], r["description_model"]): r for r in old_records}
    expected_keys = {(s["scene_id"], c, m) for s in ordered for c, m in
                     [(c, m) for m in DESCRIPTION_MODELS for c in "OJ"] + [("P", None), ("E", None)]}
    if set(indexed) != expected_keys or len(old_records) != 480:
        raise GenerationError("exact frozen evaluation semantic input scope required")
    prompt_hash = digest(PROMPT_PATH.read_bytes())
    records, gt, source_rows = {}, {}, []
    for scene in ordered:
        sid = scene["scene_id"]
        if digest(json_text(scene["source_files"]).encode()) != scene["source_manifest_sha256"]:
            raise GenerationError("scene source manifest hash mismatch")
        for source in scene["source_files"]:
            _source_path(root, source["path"], source["sha256"])
        source_rows.append({"scene_id": sid, "source_manifest_sha256": scene["source_manifest_sha256"],
                            "source_files": scene["source_files"]})
        directory = root / scene["scene_path"]
        grid = load_scene(directory).grid
        gt[sid] = {"building_directory": str(directory), "voxels_sha256": sha256_file(directory / "gt/voxels.npy"),
                   "bbox_sha256": sha256_file(directory / "gt/bbox.json"), "grid_sha256": object_hash(grid_to_payload(grid))}
        for condition, description_model in [(c, m) for m in DESCRIPTION_MODELS for c in "OJ"] + [("P", None), ("E", None)]:
            old = indexed[sid, condition, description_model]
            text = old["input_text"]
            if (old["cohort_sha256"] != cohort["cohort_sha256"] or digest(text.encode()) != old["input_sha256"]
                    or old["source_manifest_sha256"] != scene["source_manifest_sha256"]):
                raise GenerationError("old evaluation body or provenance mismatch")
            if condition in "OJ":
                source = scene["description_sources"][description_model]
                original = _source_path(root, source["path"], source["sha256"]).read_text("utf-8")
                canonical, _ = canonicalize_saved_text(original)
                if text != (original if condition == "O" else canonical) or old["source_sha256"] != source["sha256"]:
                    raise GenerationError("O/J body differs from original source or lossless serialization")
            else:
                if old["source_sha256"] != gt[sid]["voxels_sha256"] or old["source_GT_bbox_sha256"] != gt[sid]["bbox_sha256"]:
                    raise GenerationError("P/E frozen GT source mismatch")
                if condition == "E":
                    text = exact_builder_text(encode_exact(grid))
            record = {"scene_id": sid, "condition": condition,
                      "condition_label": {"O": "Original saved description", "J": "Lossless JSON reserialization",
                                          "P": "GT-attributes-only diagnostic", "E": "Complete discrete cuboid specification diagnostic"}[condition],
                      "description_model": description_model, "input_text": text,
                      "input_sha256": digest(text.encode()), "source_input_id": old["input_id"],
                      "source_input_sha256": old["input_sha256"], "source_sha256": old["source_sha256"],
                      "source_manifest_sha256": scene["source_manifest_sha256"],
                      "size_rank_stratum": scene["size_rank_stratum"], "stratum_draw_order": scene["stratum_draw_order"],
                      "cohort_sha256": cohort["cohort_sha256"], "protocol_id": PROTOCOL_ID,
                      "prompt_sha256": prompt_hash, "schema_sha256": schema_sha256(), "provenance": "new_implementation"}
            record["input_id"] = object_hash(record)
            records[sid, condition, description_model] = record
    schedule = _schedule(records, ordered)
    identity = {"protocol_id": PROTOCOL_ID, "cohort_path": str(cohort_path), "cohort_file_sha256": sha256_file(cohort_path),
                "cohort_sha256": cohort["cohort_sha256"], "source_dataset_hash": cohort["source_manifest_sha256"],
                "source_rows": source_rows, "old_evaluation_inputs_file_sha256": OLD_INPUTS_FILE_SHA256,
                "frozen_split": [{k: s[k] for k in ("scene_id", "split", "size_rank_stratum", "stratum_draw_order", "source_manifest_sha256")}
                                 for s in cohort["scenes"]],
                "evaluation_scene_ids": [s["scene_id"] for s in ordered],
                "batch_scene_ids": [s["scene_id"] for s in ordered[:10]],
                "full_schedule": schedule, "batch_schedule": schedule[:160], "gt_by_scene": gt,
                "order_rule": ORDER_RULE, "prompt_sha256": prompt_hash, "schema_sha256": schema_sha256(),
                "max_output_tokens": OUTPUT_CAP, "unseen_data": False,
                "conditions_changed_from_goal002": ["native shared schema", "resmoke prompt", "E cuboids instead of Z-runs", "output cap 16384 to 32768"],
                "conditions_unchanged_from_resmoke": ["models and returned-ID allowlists", "prompt", "schema", "E cuboid semantics", "validator/executor/metrics", "output cap32768"],
                "remaining_800_authorized": False, "population_960_authorized": False}
    return {"identity": identity, "identity_hash": object_hash(identity), "api_allowed": False, "research_api_additional_sends": 0}


def length_report(manifest: dict, pricing: dict) -> dict:
    """Measure all fixed60 inputs with cached local tokenizer; no download."""
    from offline_evaluation.resmoke_v2.cuboid_cpu import _tokenizer_status
    started = time.perf_counter()
    tokenizer, encoding = _tokenizer_status(True)
    identity = manifest["identity"]
    fixed_overhead = len(encoding.encode(PROMPT_PATH.read_text())) + len(encoding.encode(compact_json(action_schema())))
    inputs = []
    for record in identity["full_schedule"][::2]:
        count = len(encoding.encode(record["input_text"]))
        inputs.append({k: record[k] for k in ("scene_id", "condition", "description_model", "input_id", "input_sha256")}
                      | {"input_bytes": len(record["input_text"].encode()), "input_tokens_local_o200k_base": count,
                         "input_prompt_schema_tokens_local_o200k_base": count + fixed_overhead,
                         "openai_local_text_plus_output_cap_within_context": count + fixed_overhead + OUTPUT_CAP <= pricing["openai"]["context_window_tokens"],
                         "anthropic_token_count_and_context_fit": "UNKNOWN"})
    rows = []
    for scene in identity["evaluation_scene_ids"]:
        descriptor = identity["gt_by_scene"][scene]
        directory = Path(descriptor["building_directory"])
        if sha256_file(directory / "gt/voxels.npy") != descriptor["voxels_sha256"] or sha256_file(directory / "gt/bbox.json") != descriptor["bbox_sha256"]:
            raise GenerationError("GT changed during CPU length report")
        grid = load_scene(directory).grid
        spec = encode_exact(grid)
        reference = reference_actions(spec)
        reference_text = compact_json(reference)
        exact = decode_exact(spec).to_dict() == grid.to_dict()
        executed = execute(reference["operations"], origin=grid.origin).to_dict() == grid.as_world().to_dict()
        if not exact or not executed:
            raise GenerationError("current cuboid input/reference does not preserve GT")
        tokens = len(encoding.encode(reference_text))
        source = next(r for r in identity["full_schedule"] if r["scene_id"] == scene and r["condition"] == "E")
        rows.append({"scene_id": scene, "size_rank_stratum": source["size_rank_stratum"],
                     "batch1": scene in identity["batch_scene_ids"], "gt_voxels": len(grid.voxels),
                     "E_cuboid_count": len(spec["cuboids"]), "E_palette_size": len(spec["palette"]),
                     "E_input_bytes": len(source["input_text"].encode()), "E_input_tokens_local_o200k_base": len(encoding.encode(source["input_text"])),
                     "reference_action_bytes": len(reference_text.encode()), "reference_action_tokens_local_o200k_base": tokens,
                     "reference_body_within_output_cap_local_openai": tokens <= OUTPUT_CAP,
                     "reference_headroom_tokens": OUTPUT_CAP - tokens,
                     "anthropic_reference_tokens_and_fit": "UNKNOWN", "decode_exact": exact, "reference_executor_exact": executed})
    exceeded = [r["scene_id"] for r in rows if not r["reference_body_within_output_cap_local_openai"]]
    return {"schema": "goal005_fixed60_cpu_length_v1", "manifest_identity_hash": manifest["identity_hash"],
            "research_api_additional_sends": 0, "tokenizer": tokenizer, "rows": rows, "inputs": inputs,
            "summary": {"fixed_scenes": len(rows), "semantic_inputs": len(inputs), "reference_exceeds_cap_count": len(exceeded),
                        "reference_exceeds_cap_scenes": exceeded, "maximum_reference_tokens_local_o200k_base": max(r["reference_action_tokens_local_o200k_base"] for r in rows),
                        "maximum_input_prompt_schema_tokens_local_o200k_base": max(r["input_prompt_schema_tokens_local_o200k_base"] for r in inputs),
                        "wall_seconds": round(time.perf_counter() - started, 6)},
            "output_cap": OUTPUT_CAP, "targets_dropped_or_truncated": 0,
            "limitations": ["o200k_base counts text only; provider framing and injected schema prompts are not exact billed tokens.",
                            "Anthropic tokenization/context fit is not certified by OpenAI counts.",
                            "The reference is one feasible CPU sequence, not a generated output length guarantee; OpenAI reasoning also consumes output cap.",
                            "All sixty targets and cap32768 are retained. Truncation under this fixed resource limit is a possible model outcome.",
                            "Increasing cap or changing representation would define a new condition requiring explicit design/approval; it cannot be silently pooled with this fixed condition."]}


def extrapolate(plan: dict, ledger: dict, pricing: dict) -> dict:
    """Stratify measured16 by provider/condition; weight O/J three, P/E one."""
    schedule = plan["identity"]["schedule"]
    rows, groups = [], []
    if len(schedule) != 16 or len(ledger["entries"]) != 16:
        raise GenerationError("complete fixed16 empirical source required")
    for entry in ledger["entries"]:
        record = schedule[entry["schedule_index"]]
        if (entry["input_id"] != record["input_id"] or entry["provider"] != record["provider"]
                or entry["usage"] is None or entry["status"] == "reserved"):
            raise GenerationError("unmatched or incomplete measured usage")
        elapsed = (datetime.fromisoformat(entry["finished_at_utc"].replace("Z", "+00:00"))
                   - datetime.fromisoformat(entry["started_at_utc"].replace("Z", "+00:00"))).total_seconds()
        if elapsed < 0:
            raise GenerationError("negative measured elapsed time")
        rows.append({"provider": entry["provider"], "condition": record["condition"], "scene_id": record["scene_id"],
                     "attempt_id": entry["attempt_id"], "input_tokens": entry["charged_input_tokens"],
                     "output_tokens": entry["charged_output_tokens"], "cost_usd_upper_bound": entry["charged_cost_usd"],
                     "elapsed_seconds": elapsed, "outcome": entry["status"]})
    fields = ("input_tokens", "output_tokens", "cost_usd_upper_bound", "elapsed_seconds")
    for provider in MODELS:
        for condition in "OJPE":
            selected = [r for r in rows if r["provider"] == provider and r["condition"] == condition]
            if len(selected) != 2 or len({r["scene_id"] for r in selected}) != 2:
                raise GenerationError("exactly two buildings per provider-condition required")
            groups.append({"provider": provider, "condition": condition, "measured_n": 2,
                           "inputs_per_evaluation_building": 3 if condition in "OJ" else 1,
                           "means": {field: statistics.mean(r[field] for r in selected) for field in fields}})
    projections = []
    for buildings in (10, 60):
        by_provider = {}
        for provider in MODELS:
            selected = [g for g in groups if g["provider"] == provider]
            rate = pricing[provider]
            measured = {field: buildings * sum(g["inputs_per_evaluation_building"] * g["means"][field] for g in selected) for field in fields}
            reserve_input = buildings * 8 * rate["context_window_tokens"]
            reserve_output = buildings * 8 * OUTPUT_CAP
            by_provider[provider] = {"planned_calls": buildings * 8, "empirical_extrapolation": measured,
                                     "full_context_plus_cap_reservation": {"input_tokens": reserve_input, "output_tokens": reserve_output,
                                         "tokens": reserve_input + reserve_output,
                                         "cost_usd": (reserve_input * rate["input_usd_per_million"] + reserve_output * rate["output_usd_per_million"]) / 1_000_000}}
        projections.append({"buildings": buildings, "planned_calls": buildings * 16, "providers": by_provider,
                            "empirical_extrapolation": {field: sum(v["empirical_extrapolation"][field] for v in by_provider.values()) for field in fields},
                            "full_context_plus_cap_reservation": {field: sum(v["full_context_plus_cap_reservation"][field] for v in by_provider.values())
                                                                   for field in ("input_tokens", "output_tokens", "tokens", "cost_usd")}})
    return {"schema": "goal005_provider_condition_weighted_extrapolation_v1", "observations": rows, "groups": groups,
            "projections": projections, "weights_per_provider_per_building": {"O": 3, "J": 3, "P": 1, "E": 1},
            "research_api_additional_sends": 0, "proposed_cumulative_usd_cap": 10.0, "approved": False,
            "limitations": ["Only two smoke buildings; measured O/J use OpenAI descriptions, while evaluation also uses Anthropic/Gemini descriptions.",
                            "Building sizes, descriptions, response lengths, reasoning and latency differ; these are extrapolations without calibrated prediction intervals.",
                            "Elapsed time includes local post-response handling within each attempt; excludes pauses, CPU metric aggregation and checkpoint work.",
                            "Cost figures use conservative ledger accounting, not finalized bills. Measured model failures remain included.",
                            "Reservation totals sum complete documented input contexts plus full output caps; they are not completion cost forecasts.",
                            "USD10 cumulative cap is unapproved and may stop batch1 before160. Remaining800 are separately unauthorized."]}


def save_reports(destination, manifest, lengths, costs):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in (("manifest.json", manifest), ("lengths.json", lengths), ("costs.json", costs)):
        write_json(destination / name, value)
    projection_rows = [{"scope": "batch1" if p["buildings"] == 10 else "evaluation60", "buildings": p["buildings"],
                        "planned_calls": p["planned_calls"], **p["empirical_extrapolation"],
                        "reservation_tokens": p["full_context_plus_cap_reservation"]["tokens"],
                        "reservation_cost_usd": p["full_context_plus_cap_reservation"]["cost_usd"]} for p in costs["projections"]]
    manifest_rows = [{k: r[k] for k in ("schedule_index", "scene_id", "size_rank_stratum", "stratum_draw_order",
                                      "provider", "model", "condition", "description_model", "input_id", "input_sha256",
                                      "source_sha256", "source_manifest_sha256", "source_input_id", "source_input_sha256")}
                     | {"batch1": r["schedule_index"] < 160} for r in manifest["identity"]["full_schedule"]]
    for name, rows in (("lengths.csv", lengths["rows"]), ("input_lengths.csv", lengths["inputs"]),
                       ("empirical_usage16.csv", costs["observations"]), ("extrapolations.csv", projection_rows),
                       ("input_manifest960.csv", manifest_rows)):
        with (destination / name).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
