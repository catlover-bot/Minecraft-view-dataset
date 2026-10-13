from copy import deepcopy
from pathlib import Path

import pytest

from offline_evaluation.batch_v1.inputs import (
    _schedule, _source_path, extrapolate, make_manifest, ordered_evaluation,
)
from offline_evaluation.server_v2.generation import GenerationError, digest


def cohort():
    return {"scenes": [{"scene_id": f"s{stratum}-{draw:02d}", "size_rank_stratum": stratum,
                        "stratum_draw_order": draw,
                        "split": "smoke" if draw < 2 else "evaluation" if draw < 14 else "rest"}
                       for stratum in range(5) for draw in range(20)]}


def test_existing_draw_order_survives_file_permutation_and_equal_strata():
    frozen = cohort()
    first = ordered_evaluation(frozen)
    frozen["scenes"].reverse()
    assert ordered_evaluation(frozen) == first
    assert len(first) == 60
    assert [s["scene_id"] for s in first[:10]] == [f"s{s}-{d:02d}" for d in (2, 3) for s in range(5)]
    assert {s["split"] for s in first} == {"evaluation"}


@pytest.mark.parametrize("mutation", ["swap_stratum", "duplicate_scene", "duplicate_draw", "drop_scene"])
def test_fixed_split_invariants_fail_closed(mutation):
    frozen = cohort()
    rows = frozen["scenes"]
    if mutation == "swap_stratum":
        rows[2]["size_rank_stratum"] = 1
    elif mutation == "duplicate_scene":
        rows[2]["scene_id"] = rows[3]["scene_id"]
    elif mutation == "duplicate_draw":
        rows[2]["stratum_draw_order"] = 3
    else:
        rows.pop()
    with pytest.raises(GenerationError):
        ordered_evaluation(frozen)


def test_schedule_keeps_every_condition_source_once_each_provider_and_exact_prefix():
    scenes = ordered_evaluation(cohort())
    slots = [(c, m) for m in ("openai", "anthropic", "gemini") for c in "OJ"] + [("P", None), ("E", None)]
    records = {(s["scene_id"], c, m): {"scene_id": s["scene_id"], "condition": c, "description_model": m,
                                    "input_id": f'{s["scene_id"]}/{c}/{m}', "input_text": f"body-{c}-{m}"}
               for s in scenes for c, m in slots}
    saved = deepcopy(records)
    schedule = _schedule(records, scenes)
    assert len(schedule) == 960
    assert records == saved
    assert len({r["scene_id"] for r in schedule[:160]}) == 10
    for index, row in enumerate(schedule):
        assert row["schedule_index"] == index
        assert row["provider"] == ("openai" if index % 2 == 0 else "anthropic")
    for left, right in zip(schedule[::2], schedule[1::2]):
        assert left["input_id"] == right["input_id"]
        assert left["input_text"] == right["input_text"]
    for provider in ("openai", "anthropic"):
        assert {c: sum(r["provider"] == provider and r["condition"] == c for r in schedule) for c in "OJPE"} == {
            "O": 180, "J": 180, "P": 60, "E": 60}
    assert schedule[0]["condition"] == "O"
    assert schedule[16]["condition"] == "J"  # Counterbalance fixed condition position, independent of outcomes.


def test_source_checks_changed_bytes_and_symlinks(tmp_path):
    path = tmp_path / "source"
    path.write_bytes(b"frozen")
    expected = digest(b"frozen")
    assert _source_path(tmp_path, "source", expected) == path
    path.write_bytes(b"changed")
    with pytest.raises(GenerationError):
        _source_path(tmp_path, "source", expected)
    (tmp_path / "link").symlink_to(path)
    with pytest.raises(GenerationError):
        _source_path(tmp_path, "link", digest(b"changed"))


def test_cohort_file_hash_checked_before_read_or_reselection(tmp_path):
    path = tmp_path / "cohort.json"
    path.write_text("{}")
    with pytest.raises(GenerationError, match="original Goal002 cohort bytes"):
        make_manifest(root=tmp_path, cohort_path=path)


def observed():
    records, entries = [], []
    for scene in ("small", "large"):
        for condition, amount in zip("OJPE", (10, 20, 100, 200)):
            for provider in ("openai", "anthropic"):
                index = len(records)
                records.append({"scene_id": scene, "condition": condition, "provider": provider, "input_id": str(index)})
                entries.append({"schedule_index": index, "input_id": str(index), "provider": provider,
                                "attempt_id": str(index), "usage": {}, "status": "failed" if condition == "O" else "success",
                                "charged_input_tokens": amount, "charged_output_tokens": 2 * amount,
                                "charged_cost_usd": amount / 10000,
                                "started_at_utc": "2026-10-13T20:00:00Z", "finished_at_utc": "2026-10-13T20:00:10Z"})
    pricing = {p: {"context_window_tokens": 400000 if p == "openai" else 200000,
                   "input_usd_per_million": 0.25 if p == "openai" else 1.0,
                   "output_usd_per_million": 2.0 if p == "openai" else 5.0} for p in ("openai", "anthropic")}
    return {"identity": {"schedule": records}}, {"entries": entries}, pricing


def test_cost_projection_uses_3_3_1_1_not_unweighted_smoke_mean_and_keeps_failures():
    report = extrapolate(*observed())
    batch, full = report["projections"]
    assert batch["empirical_extrapolation"]["input_tokens"] == 2 * 10 * (3 * 10 + 3 * 20 + 100 + 200)
    assert batch["empirical_extrapolation"]["elapsed_seconds"] == 1600
    assert full["empirical_extrapolation"]["input_tokens"] == 6 * batch["empirical_extrapolation"]["input_tokens"]
    assert batch["full_context_plus_cap_reservation"]["cost_usd"] == pytest.approx(42.350080)
    assert full["full_context_plus_cap_reservation"]["cost_usd"] == pytest.approx(254.100480)
    assert batch["full_context_plus_cap_reservation"]["tokens"] == 53242880
    assert any(r["outcome"] == "failed" for r in report["observations"])


@pytest.mark.parametrize("mutation", ["missing", "unknown", "wrong_provider", "wrong_input", "negative_elapsed"])
def test_bad_empirical_provenance_is_rejected(mutation):
    plan, ledger, pricing = observed()
    entry = ledger["entries"][0]
    if mutation == "missing":
        ledger["entries"].pop()
    elif mutation == "unknown":
        entry["usage"] = None
    elif mutation == "wrong_provider":
        entry["provider"] = "anthropic"
    elif mutation == "wrong_input":
        entry["input_id"] = "other"
    else:
        entry["finished_at_utc"] = "2026-10-13T19:59:59Z"
    with pytest.raises(GenerationError):
        extrapolate(plan, ledger, pricing)
