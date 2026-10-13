from dataclasses import replace
import json

import numpy as np
import pytest

from offline_evaluation.server_v2.cohort_v4 import (
    CounterRNG, DESCRIPTION_MODELS, InputBudgets, audit_v4_scene, build_inputs,
    build_v4_cohort, canonicalize_saved_text, digest, freeze_ranked_split,
    input_counts, strict_saved_json, typed_tree,
)
from offline_evaluation.server_v2.core import Limits, ValidationError
from offline_evaluation.server_v2.providers import build_payload


@pytest.fixture(scope="module")
def v4_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("v4-population")
    for index in range(100):
        scene = root / "datasets/buildings_100_v4" / f"building_{index:03d}"
        (scene / "gt").mkdir(parents=True)
        count = 1 + index % 3
        np.save(scene / "gt/voxels.npy", np.full((1, 1, count), "stone", dtype="U8"), allow_pickle=False)
        (scene / "gt/bbox.json").write_text(json.dumps({"xmin": -4, "xmax": -4, "ymin": 7,
            "ymax": 7, "zmin": 10, "zmax": 9 + count, "voxel_axis_order": "Y,X,Z"}))
        views = []
        for view in range(12):
            name = f"images/view_{view:02d}.png"
            (scene / name).parent.mkdir(exist_ok=True)
            (scene / name).write_bytes(b"fixture-reference" + bytes([index, view]))
            views.append({"path": name})
        (scene / "meta.json").write_text(json.dumps({"views": views}))
        for model in DESCRIPTION_MODELS:
            folder = scene / f"description_{model}_saved_model"
            folder.mkdir()
            # Retained saved metadata is permissible; it is never supplemented.
            (folder / "description.json").write_bytes((f'{{\r\n "building":"building_{index:03d}",'
                '"summary":"stone wall","uncertainties":["opening unknown",null],'
                '"array":[2,1],"integer":1,"decimal":1e0,"boolean":true}\r\n').encode())
    return root, build_v4_cohort(root)


def test_portable_rng_and_rank_ties_do_not_depend_on_input_order():
    population = [{"scene_id": f"scene_{i:03d}", "gt_occupied": 10} for i in range(100)]
    a = freeze_ranked_split(population)
    b = freeze_ranked_split(list(reversed(population)))
    assert a == b
    assert a["split_counts"] == {"smoke": 10, "evaluation": 60, "rest": 30}
    assert len({row["scene_id"] for row in a["scenes"]}) == 100
    for stratum in range(5):
        group = [r for r in a["scenes"] if r["size_rank_stratum"] == stratum]
        assert {s: sum(r["split"] == s for r in group) for s in a["split_counts"]} == {"smoke": 2, "evaluation": 12, "rest": 6}
        assert sorted(r["rank"] for r in group) == list(range(stratum * 20, (stratum + 1) * 20))
    # Golden vector derived independently from the documented hash draws.
    assert CounterRNG(20261013, "test").shuffled(range(8)) == [5, 3, 1, 6, 0, 7, 4, 2]
    assert a != freeze_ranked_split(population, seed=20261014)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "empty", "unknown"])
def test_unrankable_population_is_never_silently_replaced(mutation):
    rows = [{"scene_id": str(i), "gt_occupied": 1} for i in range(100)]
    if mutation == "duplicate": rows[-1]["scene_id"] = rows[0]["scene_id"]
    if mutation == "missing": rows.pop()
    if mutation == "empty": rows[0]["gt_occupied"] = 0
    if mutation == "unknown": rows[0]["gt_occupied"] = None
    with pytest.raises(ValueError):
        freeze_ranked_split(rows)


def test_lossless_json_preserves_exact_types_numbers_null_order_uncertainty_and_metadata():
    original = '{"building":"original_filename","arr":[true,1,1e0,-0.0,null,"不明"],"precise":0.123456789012345678901234567890,"unknown":{"roof":"maybe open"}}'
    canonical, proof = canonicalize_saved_text(original)
    assert proof["typed_tree_equal"] is True
    assert typed_tree(strict_saved_json(original)) == typed_tree(strict_saved_json(canonical))
    assert '1e0' in canonical
    assert "original_filename" in canonical and "maybe open" in canonical
    changed = strict_saved_json(canonical)
    changed["arr"].reverse()
    assert typed_tree(changed) != typed_tree(strict_saved_json(original))
    changed = strict_saved_json(canonical)
    changed["arr"][1] = True
    assert typed_tree(changed) != typed_tree(strict_saved_json(original))


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"nested":{"x":1,"x":1}}', '{"n":NaN}', '{"n":Infinity}', '{"n":1e999999999999999999999}'])
def test_invalid_or_ambiguous_saved_json_is_rejected(text):
    with pytest.raises(ValidationError):
        canonicalize_saved_text(text)


def test_overdeep_saved_tree_is_reported_as_invalid_without_crashing_input_preparation():
    with pytest.raises(ValidationError):
        canonicalize_saved_text("[" * 1500 + "null" + "]" * 1500)


def test_complete_source_audit_and_480_fixed_inputs_per_builder(v4_fixture):
    root, cohort = v4_fixture
    assert cohort["status"] == "CPU_READY"
    assert cohort["selection_uses_model_results"] is False
    for scene in cohort["scenes"]:
        assert len([s for s in scene["source_files"] if s["kind"] == "image_reference"]) == 12
        assert len(scene["description_sources"]) == 3
        assert all(len(s["sha256"]) == 64 for s in scene["source_files"])
        assert scene["control"]["executor_control"] and scene["control"]["exact_roundtrip"]
    records = build_inputs(cohort, root=root, budgets=InputBudgets(context_window_tokens=200000))
    assert input_counts(records)["conditions"] == {"O": 180, "J": 180, "P": 60, "E": 60}
    assert len(records) == 480 and len({r["input_id"] for r in records}) == 480
    assert all(r["description_model"] is None for r in records if r["condition"] in {"P", "E"})
    assert all(r["status"] == "prepared" for r in records)
    smoke = build_inputs(cohort, split="initial_smoke", root=root)
    assert len(smoke) == 8 and len({r["scene_id"] for r in smoke}) == 2
    assert all(r["description_model"] == "openai" for r in smoke if r["condition"] in {"O", "J"})
    evaluation_ids = {r["scene_id"] for r in records}
    assert not evaluation_ids.intersection(cohort["initial_smoke_scene_ids"])
    o = next(r for r in records if r["condition"] == "O")
    assert o["input_text"].encode() == (root / o["source_path"]).read_bytes()
    assert o["source_sha256"] == digest(o["input_text"].encode())


def test_input_limits_retain_all_target_ids_and_complete_untruncated_text(v4_fixture):
    root, cohort = v4_fixture
    good = build_inputs(cohort, root=root)
    limited = build_inputs(cohort, root=root, budgets=InputBudgets(max_input_bytes=16, context_window_tokens=1))
    assert [r["input_id"] for r in limited] == [r["input_id"] for r in good]
    assert len(limited) == 480 and all(r["status"] == "infeasible" for r in limited)
    assert [r["input_text"] for r in limited] == [r["input_text"] for r in good]
    assert all("NOT_FIT_CERTIFIED_BOUND" in r["reasons"] for r in limited)
    assert all(r["feasibility"]["actual_token_count"] is None for r in limited)


def test_changed_description_after_freeze_is_retained_as_infeasible(v4_fixture):
    root, cohort = v4_fixture
    selected = next(r for r in cohort["scenes"] if r["split"] == "evaluation")
    source = root / selected["description_sources"]["openai"]["path"]
    original = source.read_bytes()
    try:
        source.write_bytes(original + b" ")
        records = build_inputs(cohort, root=root)
        broken = [r for r in records if r["scene_id"] == selected["scene_id"] and r["description_model"] == "openai"]
        assert len(records) == 480 and len(broken) == 2
        assert all(r["status"] == "infeasible" and r["source_sha256"] == digest(original) for r in broken)
        assert all("changed after cohort freeze" in r["reasons"][0] for r in broken)
    finally:
        source.write_bytes(original)


def test_missing_image_reference_and_executor_limit_are_reported_without_truncation(v4_fixture):
    root, _ = v4_fixture
    scene = root / "datasets/buildings_100_v4/building_002"
    image = scene / "images/view_11.png"
    raw = image.read_bytes()
    try:
        image.unlink()
        result = audit_v4_scene(scene, root, references={})
        assert result["status"] == "infeasible" and "missing" in result["gt_error"]
    finally:
        image.write_bytes(raw)
    result = audit_v4_scene(scene, root, limits=replace(Limits(), max_operations=0), references={})
    assert result["status"] == "infeasible" and "operation limit" in result["gt_error"]


@pytest.mark.parametrize("provider,model", [("openai", "gpt-5-mini"), ("anthropic", "claude-haiku-4-5-20251001")])
@pytest.mark.parametrize("condition", ["O", "J"])
def test_ordinary_request_serializers_do_not_transmit_manifest_gt_or_image_fields(provider, model, condition):
    original = '{ "building":"already_saved_filename", "summary":"wall", "uncertainties":["hidden interior unknown"] }'
    text = original if condition == "O" else canonicalize_saved_text(original)[0]
    record = {"input_text": text, "condition": condition, "scene_id": "sentinel_scene_metadata_only",
              "GT_path": "sentinel_gt_path", "oracle": "sentinel_oracle", "image": "sentinel_image",
              "bbox": "sentinel_bbox", "origin": "sentinel_origin"}
    payload = build_payload(provider, model, "Public shared action schema", record["input_text"], 16384)
    body = json.dumps(payload)
    assert all(value not in body for key, value in record.items() if key not in {"input_text", "condition"})
    assert "already_saved_filename" in body and "hidden interior unknown" in body
    if provider == "openai":
        assert payload["input"] == text and payload["instructions"] == "Public shared action schema"
        assert payload["truncation"] == "disabled"
    else:
        assert payload["messages"] == [{"role": "user", "content": text}]
        assert payload["system"] == "Public shared action schema"
    assert set(payload).isdisjoint({"GT_path", "oracle", "image", "bbox", "origin", "scene_id"})
