"""Frozen complete-description v4 cohort and strict lossless O/J inputs.

This module returns data and never makes HTTP requests or writes experiment
files. Old v1/v4 subsets and old Builder outputs are not consulted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path

from .core import Limits, ValidationError, execute
from .data import load_scene, sha256_file
from .io_utils import ROOT
from .oracle import decode_exact_spec, encode_exact_spec, gt_to_actions, partial_oracle

SEED = 20261013
COHORT_ID = "goal002_complete_v4_two_builder_20261013"
DESCRIPTION_MODELS = ("openai", "anthropic", "gemini")
CONDITION_LABELS = {"O": "Original saved description record", "J": "Lossless JSON reserialization",
                    "P": "Partial GT attributes Oracle", "E": "Exact-spec Oracle"}
RNG_IMPLEMENTATION = "sha256-counter-rejection-fisher-yates-v1"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_text(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class CounterRNG:
    """Portable deterministic unbiased integer draws; no process-global RNG."""
    def __init__(self, seed: int, domain: str):
        if type(seed) is not int or not isinstance(domain, str):
            raise ValueError("integer seed and string domain required")
        self.prefix = f"{RNG_IMPLEMENTATION}\0{seed}\0{domain}\0".encode("utf-8")
        self.counter = 0

    def randbelow(self, count: int) -> int:
        if type(count) is not int or not 1 <= count <= 2**256:
            raise ValueError("draw bound must be a positive bounded integer")
        ceiling = 2**256 - (2**256 % count)
        while True:
            value = int.from_bytes(hashlib.sha256(self.prefix + str(self.counter).encode("ascii")).digest(), "big")
            self.counter += 1
            if value < ceiling:
                return value % count

    def shuffled(self, values):
        result = list(values)
        for i in range(len(result) - 1, 0, -1):
            j = self.randbelow(i + 1)
            result[i], result[j] = result[j], result[i]
        return result


def strict_saved_json(text: str, max_bytes: int = 2_000_000):
    """Exact decimal numbers, duplicate-key rejection and no non-finite values."""
    if not isinstance(text, str) or len(text.encode("utf-8")) > max_bytes:
        raise ValidationError("saved JSON exceeds input byte limit")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValidationError("duplicate saved JSON key")
            result[key] = value
        return result

    def constant(_):
        raise ValidationError("non-finite saved JSON number")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_float=Decimal, parse_constant=constant)
    except (ValueError, RecursionError, InvalidOperation) as exc:
        raise ValidationError("invalid or duplicate-key saved JSON") from exc


def typed_tree(value):
    """Representation distinguishes boolean/int/decimal/null and array order."""
    if value is None:
        return ["null"]
    if type(value) is bool:
        return ["boolean", value]
    if type(value) is int:
        return ["integer", str(value)]
    if isinstance(value, Decimal):
        t = value.as_tuple()
        return ["decimal", t.sign, list(t.digits), t.exponent]
    if isinstance(value, str):
        return ["string", value]
    if isinstance(value, list):
        return ["array", [typed_tree(x) for x in value]]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return ["object", [[k, typed_tree(value[k])] for k in sorted(value)]]
    raise ValidationError("unsupported typed JSON tree")


def _serialize_exact(value) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        return str(value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValidationError("non-finite decimal")
        number = str(value)
        # A JSON exponent can parse as Decimal even when its value renders as
        # an integer. Keep its number type rather than serializing 1e0 as 1.
        return number if any(c in number for c in ".eE") else number + "e0"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ",".join(_serialize_exact(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + _serialize_exact(value[k])
                               for k in sorted(value)) + "}"
    raise ValidationError("unsupported JSON value")


def canonicalize_saved_text(text: str, max_bytes: int = 2_000_000):
    original = strict_saved_json(text, max_bytes)
    try:
        canonical = _serialize_exact(original)
        reparsed = strict_saved_json(canonical, max_bytes)
        tree = typed_tree(original)
        if tree != typed_tree(reparsed):
            raise ValidationError("typed JSON tree changed during reserialization")
        return canonical, {"typed_tree_equal": True,
                           "typed_tree_sha256": digest(json_text(tree).encode("utf-8")),
                           "transformation": "goal002_exact_decimal_lossless_json_v1"}
    except RecursionError as exc:
        raise ValidationError("saved JSON nesting exceeds safe transformation depth") from exc


def freeze_ranked_split(rows, seed: int = SEED):
    """Exactly 100 GT ranks; tie by scene ID, then 2/12/6 per quintile."""
    rows = [dict(r) for r in rows]
    if len(rows) != 100 or len({r["scene_id"] for r in rows}) != 100:
        raise ValueError("complete unique 100-scene v4 population required")
    if any(type(r.get("gt_occupied")) is not int or r["gt_occupied"] <= 0 for r in rows):
        raise ValueError("all 100 GT occupancy ranks must be known and non-empty")
    rows.sort(key=lambda r: (r["gt_occupied"], r["scene_id"]))
    result = []
    for stratum in range(5):
        group = [dict(r, rank=index, size_rank_stratum=stratum)
                 for index, r in enumerate(rows[stratum*20:(stratum+1)*20], stratum*20)]
        shuffled = CounterRNG(seed, f"v4-rank-stratum-{stratum}").shuffled(group)
        for index, row in enumerate(shuffled):
            row["split"] = "smoke" if index < 2 else "evaluation" if index < 14 else "rest"
            row["stratum_draw_order"] = index
            result.append(row)
    result.sort(key=lambda r: r["scene_id"])
    smoke = [r for r in result if r["split"] == "smoke"]
    initial = [min((r for r in smoke if r["size_rank_stratum"] == s), key=lambda r: r["scene_id"])["scene_id"]
               for s in (0, 4)]
    return {"seed": seed, "rng_implementation": RNG_IMPLEMENTATION,
            "rank_order": "GT occupied count ascending, then scene_id Unicode lexicographic",
            "split_counts": {s: sum(r["split"] == s for r in result) for s in ("smoke", "evaluation", "rest")},
            "initial_smoke_selection": "minimum scene_id among smoke in smallest/largest rank quintile",
            "initial_smoke_scene_ids": initial, "scenes": result}


@dataclass(frozen=True)
class InputBudgets:
    max_input_bytes: int = 2_000_000
    max_output_tokens: int = 16_384
    context_window_tokens: int | None = None
    framing_margin_bytes: int = 1024

    def __post_init__(self):
        for name in ("max_input_bytes", "max_output_tokens", "framing_margin_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError("input/output/framing budgets must be nonnegative integers")
        if self.context_window_tokens is not None and (type(self.context_window_tokens) is not int or self.context_window_tokens <= 0):
            raise ValueError("context ceiling must be positive or unconfigured")


def _budgets(value):
    return value if isinstance(value, InputBudgets) else InputBudgets(**(value or {}))


def _references(root: Path):
    path = root / "reports/server_only/originals_manifest.jsonl"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as stream:
        return {r["path"]: r for r in map(json.loads, stream)}


def _source(path: Path, root: Path, references: dict, kind: str):
    relative = path.relative_to(root).as_posix()
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValidationError("missing, symlink or escaped source: " + relative)
    sha = sha256_file(path)
    reference = references.get(relative)
    if reference is not None and reference["sha256"] != sha:
        raise ValidationError("source hash differs from preserved original: " + relative)
    return {"kind": kind, "path": relative, "sha256": sha, "size_bytes": path.stat().st_size,
            "repo": reference.get("repo") if reference else None,
            "ref": reference.get("ref") if reference else None,
            "blob_hash": reference.get("blob_hash") if reference else None,
            "original_reference_verified": reference is not None}


def audit_v4_scene(scene_path, root=ROOT, limits: Limits = Limits(), references=None):
    """Hash GT, three saved records and all 12 image references; CPU control."""
    root, scene_path = Path(root), Path(scene_path)
    references = _references(root) if references is None else references
    row = {"scene_id": f"buildings_100_v4_{scene_path.name}", "dataset_version": "v4",
           "scene_path": scene_path.relative_to(root).as_posix(), "status": "prepared",
           "errors": [], "source_files": [], "description_sources": {}}
    try:
        for name, kind in (("gt/voxels.npy", "GT"), ("gt/bbox.json", "GT_bbox")):
            row["source_files"].append(_source(scene_path / name, root, references, kind))
        scene = load_scene(scene_path, limits)
        row["gt_occupied"] = len(scene.grid.voxels)
        if not row["gt_occupied"]:
            raise ValidationError("GT is empty")
        row["source_files"].append(_source(scene_path / "meta.json", root, references, "scene_metadata"))
        meta = strict_saved_json((scene_path / "meta.json").read_bytes().decode("utf-8"), limits.max_json_bytes)
        if not isinstance(meta, dict):
            raise ValidationError("scene metadata must be an object")
        views = meta.get("views", [])
        if not isinstance(views, list) or len(views) != 12:
            raise ValidationError("v4 scene requires 12 image references")
        names = []
        for view in views:
            name = view.get("path") if isinstance(view, dict) else None
            if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValidationError("unsafe image reference")
            names.append(name)
            row["source_files"].append(_source(scene_path / name, root, references, "image_reference"))
        if len(set(names)) != 12:
            raise ValidationError("duplicate image references")
        spec = encode_exact_spec(scene.grid, limits)
        actions = gt_to_actions(scene.grid, limits)
        control = execute(actions, limits, origin=scene.grid.origin)
        row["control"] = {"nonempty_GT": True, "exact_roundtrip": decode_exact_spec(spec, limits).to_dict() == scene.grid.to_dict(),
                          "executor_control": control.to_dict() == scene.grid.to_dict(),
                          "reference_action_count": len(actions), "exact_spec_bytes": len(json_text(spec).encode()),
                          "reference_actions_json_bytes": len(json_text({"operations": actions}).encode()),
                          "origin_xyz": list(scene.grid.origin), "coordinate_frame": scene.grid.coordinate_frame,
                          "executor_limits": asdict(limits)}
        if not row["control"]["exact_roundtrip"] or not row["control"]["executor_control"]:
            raise ValidationError("reference exact/control mismatch")
    except (ValueError, OSError, UnicodeError) as exc:
        row.update(status="infeasible", gt_occupied=row.get("gt_occupied"), gt_error=str(exc))
        row["errors"].append("GT_SOURCE_OR_REFERENCE_INFEASIBLE")
    for model in DESCRIPTION_MODELS:
        try:
            folders = list(scene_path.glob(f"description_{model}_*/description.json"))
            folders += list((root / "outputs/i2t2b/buildings_100_v4" / scene_path.name).glob(f"description_{model}_*/description.json"))
            if len(folders) != 1:
                raise ValidationError("exactly one saved description source required")
            source = _source(folders[0], root, references, "saved_description")
            _, verification = canonicalize_saved_text(folders[0].read_bytes().decode("utf-8"))
            source.update(verification, status="prepared")
            row["description_sources"][model] = source
            row["source_files"].append(source)
        except (ValueError, OSError, UnicodeError) as exc:
            row["description_sources"][model] = {"status": "infeasible", "reason": str(exc)}
            row["status"] = "infeasible"
            row["errors"].append("DESCRIPTION_SOURCE_INFEASIBLE:" + model)
    row["source_manifest_sha256"] = digest(json_text(row["source_files"]).encode())
    return row


def build_v4_cohort(root=ROOT, seed: int = SEED, limits: Limits = Limits(), budgets=None):
    root = Path(root)
    references = _references(root)
    scenes = [audit_v4_scene(root / "datasets/buildings_100_v4" / f"building_{i:03d}", root, limits, references)
              for i in range(100)]
    cohort = {"cohort_id": COHORT_ID, "provenance": "new_implementation", "dataset_version": "v4",
              "unseen_data": False, "selection_uses_model_results": False, "past_i2t2b_outputs_used_as_builder_results": False,
              "historical_arr_replay": "BLOCKED_MISSING_ORIGINALS", "experiment_completion": "NOT_RUN",
              "executor_limits": asdict(limits), "input_budgets": asdict(_budgets(budgets)),
              "population_scenes": 100, "research_requests_sent": 0}
    if any(type(r.get("gt_occupied")) is not int or r["gt_occupied"] <= 0 for r in scenes):
        cohort.update(status="COHORT_BLOCKED_UNRANKABLE_GT", scenes=scenes, split_counts={}, initial_smoke_scene_ids=[])
    else:
        cohort.update(freeze_ranked_split(scenes, seed), status="CPU_READY" if all(r["status"] == "prepared" for r in scenes) else "INPUTS_PARTIAL")
    cohort["source_manifest_sha256"] = digest(json_text([{k: r[k] for k in ("scene_id", "source_manifest_sha256")} for r in scenes]).encode())
    cohort["cohort_sha256"] = digest(json_text(cohort).encode())
    return cohort


def _input_feasibility(text, budgets, prompt_bytes, reference=None):
    size = len(text.encode("utf-8"))
    upper = size + prompt_bytes + budgets.framing_margin_bytes
    reasons = []
    if size > budgets.max_input_bytes:
        reasons.append("INPUT_BYTE_LIMIT")
    context = "NOT_CONFIGURED"
    if budgets.context_window_tokens is not None:
        context = "WITHIN_CONSERVATIVE_BYTE_BOUND" if upper + budgets.max_output_tokens <= budgets.context_window_tokens else "NOT_FIT_CERTIFIED_BOUND"
        if context == "NOT_FIT_CERTIFIED_BOUND":
            reasons.append(context)
    result = {"input_bytes": size, "prompt_bytes": prompt_bytes,
              "framing_margin_bytes": budgets.framing_margin_bytes, "input_token_upper_bound_bytes": upper,
              "context_check": context, "actual_token_count": None,
              "framing_and_tokenizer_bound_certification": "UNVERIFIED_ASSUMPTION_REQUIRES_MODEL_ADAPTER_REVIEW",
              "configured_budgets": asdict(budgets)}
    if reference is not None:
        result.update(reference_actions_json_bytes=reference,
                      reference_output_token_upper_bound_bytes=reference,
                      reference_output_cap_status="FIT_CONSERVATIVE_BYTE_BOUND" if reference <= budgets.max_output_tokens else "REFERENCE_TOKENIZATION_UNVERIFIED",
                      exact_output_completion_guaranteed=False)
    return result, reasons


def build_inputs(cohort, split="evaluation", description_models=None, budgets=None, root=ROOT, prompt=None):
    """Retain planned invalid rows. P/E occur once per building, never per model."""
    root = Path(root)
    if split not in {"evaluation", "smoke", "rest", "initial_smoke"}:
        raise ValueError("unknown v4 split")
    if cohort["status"] == "COHORT_BLOCKED_UNRANKABLE_GT":
        raise ValueError("cannot replace targets or rank unknown GT")
    selected = [r for r in cohort["scenes"] if r["scene_id"] in cohort["initial_smoke_scene_ids"]] if split == "initial_smoke" else [r for r in cohort["scenes"] if r["split"] == split]
    models = tuple(description_models if description_models is not None else (("openai",) if split == "initial_smoke" else DESCRIPTION_MODELS))
    if split == "initial_smoke" and models != ("openai",):
        raise ValueError("initial smoke fixes saved OpenAI descriptions only")
    if not models or len(set(models)) != len(models) or any(m not in DESCRIPTION_MODELS for m in models):
        raise ValueError("invalid description models")
    budgets = _budgets(budgets or cohort.get("input_budgets"))
    limits = Limits(**cohort["executor_limits"])
    if prompt is None:
        prompt = (Path(__file__).parent / "prompts/builder.txt").read_text("utf-8")
    prompt_bytes = len(prompt.encode())
    rows = []
    for scene in selected:
        base = {"scene_id": scene["scene_id"], "dataset_version": "v4", "split": split,
                "cohort_id": cohort["cohort_id"], "cohort_sha256": cohort["cohort_sha256"],
                "source_manifest_sha256": scene["source_manifest_sha256"], "provenance": "new_implementation",
                "experiment_completion": "NOT_RUN"}
        for model in models:
            source = scene["description_sources"][model]
            for condition in ("O", "J"):
                record = dict(base, condition=condition, condition_label=CONDITION_LABELS[condition], description_model=model,
                              oracle=False, input_text=None, status="infeasible", reasons=[],
                              source_path=source.get("path"), source_sha256=source.get("sha256"))
                try:
                    if source["status"] != "prepared":
                        raise ValidationError("missing or invalid saved description")
                    path = root / source["path"]
                    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                        raise ValidationError("unsafe saved description source after cohort freeze")
                    raw = path.read_bytes()
                    if digest(raw) != source["sha256"]:
                        raise ValidationError("saved description changed after cohort freeze")
                    original = raw.decode("utf-8")
                    canonical, verification = canonicalize_saved_text(original)
                    text = original if condition == "O" else canonical
                    feasibility, reasons = _input_feasibility(text, budgets, prompt_bytes)
                    record.update(input_text=text, source_path=source["path"], source_sha256=source["sha256"],
                                  input_sha256=digest(text.encode()), typed_tree_sha256=verification["typed_tree_sha256"],
                                  typed_tree_equal=True, transformation="identity" if condition == "O" else verification["transformation"],
                                  feasibility=feasibility, status="infeasible" if reasons else "prepared", reasons=reasons)
                except (ValueError, OSError, UnicodeError) as exc:
                    record["reasons"] = [str(exc)]
                record["input_id"] = digest(json_text({k: record[k] for k in ("cohort_id", "scene_id", "description_model", "condition")}).encode())
                rows.append(record)
        try:
            gt_sources = {s["kind"]: s for s in scene["source_files"] if s["kind"] in {"GT", "GT_bbox"}}
            if set(gt_sources) != {"GT", "GT_bbox"}:
                raise ValidationError("both frozen GT source references are required")
            for source in scene["source_files"]:
                if source["kind"] in {"GT", "GT_bbox"}:
                    path = root / source["path"]
                    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or sha256_file(path) != source["sha256"]:
                        raise ValidationError("GT changed after cohort freeze")
            loaded = load_scene(root / scene["scene_path"], limits)
            spec = encode_exact_spec(loaded.grid, limits)
            actions = gt_to_actions(loaded.grid, limits)
            reference = len(json_text({"operations": actions}).encode())
            if decode_exact_spec(spec, limits).voxels != loaded.grid.voxels or execute(actions, limits).voxels != loaded.grid.voxels:
                raise ValidationError("Oracle/Executor reference infeasible")
            oracle_values = {"P": partial_oracle(loaded.grid), "E": spec}
            oracle_error = None
        except (ValueError, OSError) as exc:
            oracle_values, reference, oracle_error = {}, None, str(exc)
        gt_sources = {s["kind"]: s for s in scene["source_files"] if s["kind"] in {"GT", "GT_bbox"}}
        for condition in ("P", "E"):
            record = dict(base, condition=condition, condition_label=CONDITION_LABELS[condition], description_model=None,
                          oracle=True, diagnostic_only=True, status="infeasible", input_text=None,
                          reasons=[oracle_error] if oracle_error else [], source_sha256=gt_sources.get("GT", {}).get("sha256"),
                          source_GT_bbox_sha256=gt_sources.get("GT_bbox", {}).get("sha256"))
            if oracle_error is None:
                text = json_text(oracle_values[condition])
                feasibility, reasons = _input_feasibility(text, budgets, prompt_bytes, reference if condition == "E" else None)
                record.update(input_text=text, input_sha256=digest(text.encode()), feasibility=feasibility,
                              status="infeasible" if reasons else "prepared", reasons=reasons,
                              reference_executor_feasible=True, exact_roundtrip_feasible=True)
            record["input_id"] = digest(json_text({k: record[k] for k in ("cohort_id", "scene_id", "description_model", "condition")}).encode())
            rows.append(record)
    return rows


def input_counts(rows):
    return {"records": len(rows), "prepared": sum(r["status"] == "prepared" for r in rows),
            "infeasible": sum(r["status"] != "prepared" for r in rows),
            "conditions": {c: sum(r["condition"] == c for r in rows) for c in CONDITION_LABELS},
            "independent_buildings": len({r["scene_id"] for r in rows}), "research_requests_sent": 0}


def protocol():
    """Predeclared protocol; observed results and paid permission are absent."""
    from .analysis_v4 import comparison_families
    return {"schema": "goal002_v4_protocol_v1", "cohort_id": COHORT_ID, "seed": SEED,
            "population": "existing complete-description buildings_100_v4, 100 buildings; not unseen data",
            "rng": RNG_IMPLEMENTATION, "strata": "occupied GT count rank quintiles; ties by scene_id; 20 per stratum",
            "splits": {"smoke": 10, "evaluation": 60, "rest": 30},
            "per_stratum": {"smoke": 2, "evaluation": 12, "rest": 6},
            "initial_smoke": {"buildings": 2, "selection": "min scene_id in smoke quintiles0/4", "description_models": ["openai"],
                              "inputs_per_builder": 8, "combined_inputs": 16},
            "conditions": CONDITION_LABELS, "evaluation_inputs_per_builder": {"O": 180, "J": 180, "P": 60, "E": 60},
            "evaluation_inputs_combined": 960, "executor_control": "CPU only; once/building; never counted as model success",
            "normal_input_rule": "saved record text only; retain original metadata including original building filename; no extra GT/path/image/scene-specific information",
            "J_rule": "duplicate keys rejected; exact decimal/integer types, null, array order, all fields and uncertainty strings preserved; typed-tree equality and source SHA recorded",
            "evaluation_version": "server_v2.2_strict_translation",
            "alignment": "strict integer sparse/NTT maximum occupancy overlap across all XYZ translations; shared occupancy-selected shift for material and repair; tie L1 minimum then XYZ lexicographic; no rotation/scale/clamp; resource-limit cases are unmeasurable",
            "comparison_families": comparison_families(),
            "inference": "equal-weight building paired means; building-cluster percentile bootstrap2000 and seeded two-sided sign-flip10000 (exact at n<=16); sign-flip requires paired-difference exchangeability/symmetry under its null; predeclared family Holm; no smoke significance/generalization claims",
            "failure_policy": "valid-only conditional accuracy plus completed-request all-attempted failure0 for bounded IoU scores in[0,1]; eligible only identified completed request with known usage and parse/validation/execution/truncation failure; auth/billing/model unavailable/context/missing key/transport/unknown usage remain operational or unmeasurable and separate rates; NOT_RUN/budget stop/missing input/GT or alignment-resource failure excluded; MAE/repair failures null and metric-specific common-valid pairs",
            "failure0_requires": {"completed_request_outcome": True, "failure_kind": ["parse_failure", "validation_failure", "execution_failure", "truncation"],
                                  "note": "attempt_id alone and generic failed status do not prove a model output; failure0 bounds the score, not the translation search radius"},
            "smoke_caps_proposed_not_approved": {"global_calls": 20, "global_retries": 4, "attempts_per_input": 2,
                                                "global_cost_usd": 5, "max_output_tokens_per_provider": 16384,
                                                "global_tokens": 8327680, "concurrency": 1, "hidden_retries": 0},
            "model_ids": {"openai": "gpt-5-mini", "anthropic": "claude-haiku-4-5-20251001"},
            "model_availability": "NOT_RUN; no fallback; first model deprecated, second exact ID legacy Active per root official-doc check",
            "reasoning_policy": "OpenAI reasoning and visible output share output cap/billing; Anthropic adapter does not enable extended thinking; provider defaults differ",
            "budgets": {"input_bytes": 2000000, "output_tokens": 16384, "context_tokens": {"openai": 400000, "anthropic": 200000},
                        "executor": asdict(Limits()), "token_counts": "not measured; conservative UTF8-byte input upper plus prompt and framing1024; wrapper/tokenizer certification remains explicit"},
            "limitations": ["Original descriptions/images may not identify world origin; absolute errors are not solely description ability.",
                            "P/E expose nonvisible openings/interiors and exact GT attributes; advantages are diagnostic, not pure language information loss.",
                            "E additionally tests palette/Z-run decoding and expansion; output completion at16384 is not guaranteed; reference sizes and uncertified tokenization are reported.",
                            "J is a new lossless serialization contrast, not historical Structured IR or a general claim about structuring.",
                            "No previous i2t2b-v5 predictions are reused as new first Builder outputs; both providers require fresh matched-period generation.",
                            "No target replacement, silent truncation, v1 Gemini generation, automated full960 launch or paid request is authorized by Goal002."],
            "experiment_completion": "NOT_RUN", "research_requests_sent": 0, "budget_approval": "AWAITING_EXPLICIT_USER_APPROVAL"}
