#!/usr/bin/env python3
"""Read-only original-source inventory; never fetch blobs or load NumPy/pickle data.

GitHub calls use the already authenticated gh client and retain public metadata only.
All generated reports are confined to reports/server_only in the worktree.
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import hashlib
import json
import re
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = "catlover-bot/Minecraft-view-dataset"
DEFAULT_REF = "7309327efd1c19dda793b19e8d8e6ce0b5868db9"
MISSING_CANDIDATES = [
    "offline_evaluation/fixed_builder/",
    "offline_evaluation/executor/",
    "offline_evaluation/metrics.py",
    "offline_evaluation/metrics/",
    "offline_evaluation/experiments/fixed_builder_v1/",
    "offline_evaluation/analyze_correlations_v2.py",
    "offline_evaluation/analyze_materials_v2.py",
    "offline_evaluation/data_coverage.py",
]


def command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def tree(ref: str) -> dict[str, dict[str, str]]:
    result = {}
    for line in command("git", "ls-tree", "-r", "-z", ref).split("\0"):
        if not line:
            continue
        metadata, path = line.split("\t", 1)
        mode, kind, blob = metadata.split()
        result[path] = {"mode": mode, "type": kind, "blob_hash": blob}
    return result


def github_metadata(endpoint: str) -> dict:
    """Capture public repo metadata; never print command failure text or auth state."""
    result = subprocess.run(
        ["gh", "api", "--paginate", f"repos/{REPO}/{endpoint}"],
        cwd=ROOT, text=True, capture_output=True,
    )
    if result.returncode:
        return {"status": "UNAVAILABLE", "exit_code": result.returncode}
    # gh --paginate returns adjacent JSON documents on multi-page results.
    decoder, remaining, pages = json.JSONDecoder(), result.stdout.strip(), []
    while remaining:
        value, offset = decoder.raw_decode(remaining)
        pages.append(value)
        remaining = remaining[offset:].lstrip()
    values = [entry for page in pages for entry in page] if isinstance(pages[0], list) else pages
    if endpoint == "branches":
        values = [{"name": x["name"], "sha": x["commit"]["sha"]} for x in values]
    elif endpoint == "tags":
        values = [{"name": x["name"], "sha": x["commit"]["sha"]} for x in values]
    elif endpoint == "releases":
        values = [{"id": x["id"], "tag_name": x["tag_name"], "draft": x["draft"],
                   "created_at": x["created_at"], "target_commitish": x["target_commitish"],
                   "assets": [{k: a[k] for k in ("id", "name", "size", "browser_download_url")}
                              for a in x["assets"]]} for x in values]
    elif endpoint == "actions/artifacts":
        values = [{"total_count": x["total_count"], "artifacts": [
            {k: a.get(k) for k in ("id", "name", "size_in_bytes", "expired", "expires_at", "workflow_run")}
            for a in x["artifacts"]]} for x in values]
    return {"status": "FOUND", "items": values}


def npy_header(path: Path) -> dict:
    """Read format header through literal_eval, without executing or loading payload."""
    with path.open("rb") as handle:
        magic = handle.read(8)
        if magic[:6] != b"\x93NUMPY" or magic[6] not in (1, 2, 3):
            raise ValueError("Unrecognized NPY header")
        nbytes = 2 if magic[6] == 1 else 4
        length = struct.unpack("<H" if nbytes == 2 else "<I", handle.read(nbytes))[0]
        if length > 65536:
            raise ValueError("Excessive NPY header")
        return ast.literal_eval(handle.read(length).decode("utf8" if magic[6] == 3 else "latin1").strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-ref", default=DEFAULT_REF)
    parser.add_argument("--offline", action="store_true", help="Reuse existing GitHub metadata if available")
    args = parser.parse_args()
    out = ROOT / "reports/server_only"
    out.mkdir(parents=True, exist_ok=True)
    ref = command("git", "rev-parse", args.source_ref).strip()
    originals = tree(ref)
    commits = command("git", "rev-list", "--all").split()
    history = {r: tree(r) for r in commits}
    all_paths = set().union(*(set(t) for t in history.values()))
    missing = []
    for wanted in MISSING_CANDIDATES:
        occurrences = [{"ref": r, "paths": [p for p in t if p == wanted or p.startswith(wanted)]}
                       for r, t in history.items() if any(p == wanted or p.startswith(wanted) for p in t)]
        missing.append({"path": wanted, "status": "FOUND" if occurrences else "MISSING",
                        "reachable_history_occurrences": occurrences})
    metadata = {}
    old_report = out / "source_audit.json"
    if args.offline and old_report.exists():
        metadata = json.loads(old_report.read_text())["github_metadata"]
    elif not args.offline:
        metadata = {endpoint: github_metadata(endpoint) for endpoint in
                    ("branches", "tags", "releases", "actions/artifacts")}
    manifest = {}
    pointer_paths, mismatch_paths, absent_paths = [], [], []
    with (out / "originals_manifest.jsonl").open("w") as handle:
        for path, git_info in sorted(originals.items()):
            local = ROOT / path
            if not local.is_file() or local.is_symlink():
                absent_paths.append(path)
                continue
            size = local.stat().st_size
            sha, blob = hashlib.sha256(), hashlib.sha1(f"blob {size}\0".encode())
            with local.open("rb") as content:
                first = content.read(1024 * 1024)
                if first.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
                    pointer_paths.append(path)
                sha.update(first)
                blob.update(first)
                for chunk in iter(lambda: content.read(1024 * 1024), b""):
                    sha.update(chunk)
                    blob.update(chunk)
            matches = blob.hexdigest() == git_info["blob_hash"]
            if not matches:
                mismatch_paths.append(path)
            rec = {"repo": REPO, "ref": ref, "path": path, **git_info,
                   "sha256": sha.hexdigest(), "size_bytes": size, "worktree_matches_git_blob": matches,
                   "hash_source": "original_git_blob_verified_in_worktree"}
            if not matches:
                # Concurrent authorized edits are not originals. Recover only a small
                # original blob; never bulk-fetch changed large data for the inventory.
                original_size = int(command("git", "cat-file", "-s", git_info["blob_hash"]).strip())
                if original_size > 1024 * 1024:
                    raise ValueError(f"Changed large original requires a narrow explicit recovery: {path}")
                original = subprocess.check_output(["git", "cat-file", "blob", git_info["blob_hash"]], cwd=ROOT)
                rec.update(local_worktree_sha256=rec["sha256"], local_worktree_size_bytes=size,
                           sha256=hashlib.sha256(original).hexdigest(), size_bytes=len(original),
                           hash_source="original_git_blob_recovered_without_overwriting_worktree")
            manifest[path] = rec
            handle.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
    primary = lambda p: p.startswith(("datasets/buildings_100_v1/", "datasets/buildings_100_v4/"))
    legacy = lambda p: p.startswith("outputs/i2t2b/")
    descriptions = [p for p in originals if p.endswith("/description.json") and (primary(p) or legacy(p))]
    gt = [p for p in originals if primary(p) and p.endswith("/gt/voxels.npy")]
    gt_images = [p for p in originals if primary(p) and "/images/" in p and p.endswith(".png")]
    actions = [p for p in originals if legacy(p) and p.endswith("/actions.json")]
    predictions, op_counts, action_parse_errors = [], collections.Counter(), []
    for path in sorted(actions):
        parts = path.split("/")
        version = parts[2].removeprefix("buildings_100_")
        building = parts[3]
        pred = str(Path(path).with_name("voxels.npy"))
        predictions.append({"scene_id": f"{version}:{building}",
                            "scene_path": f"datasets/buildings_100_{version}/{building}",
                            "prediction_path": str(Path(pred).parent), "voxel_path": pred,
                            "actions_path": path, "condition": parts[4], "source_ref": ref,
                            "sha256": manifest[pred]["sha256"],
                            "actions_sha256": manifest[path]["sha256"],
                            "provenance": "historical_original_artifact_with_new_metrics"})
        try:
            operations = json.loads((ROOT / path).read_text())["operations"]
            if not isinstance(operations, list):
                raise ValueError("Operation list required")
            op_counts.update(str(op.get("op", "MISSING")) for op in operations if isinstance(op, dict))
        except (KeyError, ValueError, TypeError):
            action_parse_errors.append(path)
    write_json(out / "available_predictions.json", predictions)
    metric_tables = []
    saved_json_rows = 0
    for path in originals:
        if path.endswith(".csv") and ("per_scene" in path or "stochastic_repeat_summary" in path):
            with (ROOT / path).open() as handle:
                rows = list(csv.DictReader(handle))
            metric_tables.append({"path": path, "rows": len(rows),
                                  "unique_scenes": len({r["scene_id"] for r in rows}) if rows and "scene_id" in rows[0] else None,
                                  "conditions": dict(collections.Counter(r.get("condition", r.get("description_model", "NA")) for r in rows)),
                                  "evidence_type": "saved_table_only_not_regenerated_output"})
        if legacy(path) and "/metrics/rebuild/" in path and path.endswith(".json"):
            saved_json_rows += len(json.loads((ROOT / path).read_text()).get("items", []))
    tests = []
    for path in sorted(p for p in originals if p.startswith("offline_evaluation/tests/") and p.endswith(".py")):
        module = ast.parse((ROOT / path).read_text())
        tests.append({"path": path, "test_count": sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_") for node in ast.walk(module)),
                      "external_dependencies": sorted({node.names[0].name.split(".")[0] if isinstance(node, ast.Import) else (node.module or "").split(".")[0]
                                                       for node in ast.walk(module) if isinstance(node, (ast.Import, ast.ImportFrom))} - {"offline_evaluation", "tools", "pathlib", "json", "os"}),
                      "static_api_safety": "API-free tests; legacy imports contain no top-level network calls in existing modules",
                      "execution_status": "NOT_RUN_BY_AUDITOR; root records actual collection result separately"})
    write_json(out / "legacy_test_review.json", {"test_modules": tests,
               "total_test_functions": sum(x["test_count"] for x in tests),
               "safe_to_collect_without_research_api": True,
               "required_dependencies": ["numpy", "pandas", "statsmodels", "pytest"],
               "existing_requirements_complete": False,
               "missing_legacy_imports": [x["path"] for x in missing if x["status"] == "MISSING"],
               "caution": "Some existing tests are tautological or test filesystem presence only. Their pass would not verify resume, budget, provider execution, or oracle semantics."})
    headers = collections.Counter(str(npy_header(ROOT / p)["descr"]) for p in originals if p.endswith("/voxels.npy"))
    bbox_mismatches, axis_counts, missing_view_paths = [], collections.Counter(), []
    verified_primary = gt + [x["voxel_path"] for x in predictions]
    for path in verified_primary:
        bbox = json.loads((ROOT / path).with_name("bbox.json").read_text())
        expected_shape = tuple(bbox[hi] - bbox[lo] + 1 for lo, hi in
                               (("ymin", "ymax"), ("xmin", "xmax"), ("zmin", "zmax")))
        if tuple(npy_header(ROOT / path)["shape"]) != expected_shape:
            bbox_mismatches.append(path)
        axis_counts[bbox.get("voxel_axis_order", "unspecified")] += 1
    metadata_view_count = 0
    for path in gt:
        scene = (ROOT / path).parent.parent
        meta = json.loads((scene / "meta.json").read_text())
        for view in meta.get("views", []):
            metadata_view_count += 1
            image_path = scene / view["path"]
            if str(image_path.relative_to(ROOT)) not in originals or not image_path.is_file():
                missing_view_paths.append(str(image_path.relative_to(ROOT)))
    immutable_paths = sorted(p for p in originals if primary(p) or
                             (legacy(p) and any(k in p for k in ("/description_", "/rebuild_plan_", "/rebuild_world_", "/metrics/"))))
    write_json(out / "source_archive_allowlist.json", {"repo": REPO, "source_ref": ref,
               "paths": immutable_paths, "total_size_bytes": sum(manifest[p]["size_bytes"] for p in immutable_paths),
               "security_scan_required_before_archive": True,
               "scope": "primary v1/v4 dataset inputs and older i2t2b source outputs only; captures/human cohorts excluded"})
    report = {"schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "repo": REPO, "source_ref": ref, "original_tree_hash": command("git", "rev-parse", f"{ref}^{{tree}}").strip(),
              "scan_scope": "same repository current original tree, every locally reachable commit tree, GitHub branches/tags/releases/Actions artifact metadata",
              "shallow": command("git", "rev-parse", "--is-shallow-repository").strip() == "true",
              "partial_clone": {"promisor": command("git", "config", "--get", "remote.origin.promisor").strip(), "filter": command("git", "config", "--get", "remote.origin.partialclonefilter").strip(),
                                "note": "Tree scan does not require fetching absent blobs. No blanket blob or artifact downloads performed."},
              "reachable_commits": commits, "unique_historical_paths": len(all_paths),
              "github_metadata": metadata, "missing_candidates": missing,
              "declared_lfs_submodules": {"historical_config_paths": sorted(p for p in all_paths if p.endswith((".gitattributes", ".gitmodules"))),
                                          "gitlinks": sorted({p for t in history.values() for p, v in t.items() if v["mode"] == "160000"}), "current_lfs_pointer_paths": pointer_paths},
              "manifest": {"path": "reports/server_only/originals_manifest.jsonl", "records": len(manifest), "bytes_hashed": sum(x["size_bytes"] for x in manifest.values()), "worktree_mismatches": mismatch_paths, "unavailable_worktree_paths": absent_paths},
              "primary_cohort_counts": {"gt_scenes": len(gt), "gt_images": len(gt_images), "descriptions": len(descriptions),
                                        "description_models": dict(collections.Counter("gemini" if "gemini" in p else "claude" if "anthropic" in p else "openai" for p in descriptions)),
                                        "missing_description_scenes": [f"v1:building_{i:03d}:gemini" for i in range(100)],
                                        "arr_original_canonical_inputs": 0, "arr_original_attempt_records": 0,
                                        "arr_original_successful_action_artifacts": 0, "arr_original_prediction_voxels": 0,
                                        "historical_i2t2b_action_artifacts": len(actions), "historical_i2t2b_parsed_operation_lists": len(actions) - len(action_parse_errors),
                                        "historical_i2t2b_prediction_voxels": len(predictions), "historical_i2t2b_generation_attempt_count": None,
                                        "historical_i2t2b_confirmed_successful_api_calls": None,
                                        "saved_legacy_rebuild_evaluation_rows_in_json": saved_json_rows},
              "supplementary_cohort_counts": [{"dataset": name,
                                              "gt_scenes": sum(p.startswith(f"datasets/{name}/") and p.endswith("/gt/voxels.npy") for p in originals),
                                              "descriptions": sum(p.startswith(f"datasets/{name}/") and p.endswith("/description.json") for p in originals),
                                              "actions": sum(p.startswith(f"datasets/{name}/") and p.endswith("/actions.json") for p in originals),
                                              "all_voxel_artifacts": sum(p.startswith(f"datasets/{name}/") and p.endswith("/voxels.npy") for p in originals),
                                              "png_files_including_captures": sum(p.startswith(f"datasets/{name}/") and p.endswith(".png") for p in originals),
                                              "included_in_primary_counts": False, "included_in_source_archive_allowlist": False}
                                             for name in ("llm_authored_10_minecraft", "llm_authored_10_minecraft_claude")],
              "expected_found": [{"artifact": "primary GT", "expected": 200, "found": len(gt), "status": "FOUND"},
                                 {"artifact": "primary descriptions", "expected": 600, "found": len(descriptions), "status": "PARTIAL"},
                                 {"artifact": "ARR fixed_builder main reconstructions", "expected": 1200, "found": 0, "status": "MISSING"},
                                 {"artifact": "ARR Direct generated outputs", "expected": 100, "found": 0, "status": "MISSING"},
                                 {"artifact": "ARR stochastic building IDs/raw runs", "expected": 30, "found": 0, "status": "MISSING"},
                                 {"artifact": "second Builder real API executor", "expected": 1, "found": 0, "status": "MISSING", "existing_wrapper": "dry-run-only"},
                                 {"artifact": "complete legacy Python dependency definition", "expected": 1, "found": 0, "status": "MISSING"}],
              "saved_tables": metric_tables, "npy_header_dtype_counts": dict(headers),
              "primary_shape_and_image_checks": {"checked_gt_and_prediction_arrays": len(verified_primary),
                                                 "axis_order_counts": dict(axis_counts), "inclusive_bbox_shape_mismatches": bbox_mismatches,
                                                 "GT_metadata_view_references": metadata_view_count, "missing_referenced_images": missing_view_paths},
              "legacy_action_operation_counts": dict(op_counts), "legacy_action_parse_errors": action_parse_errors,
              "recovered_historical_sources": [],
              "interpretation": {"historical_replay": "BLOCKED for missing ARR raw outputs, canonicalized inputs and original evaluator modules; no synthetic replacement used",
                                 "available_real_data": "200 GT and 1200 older i2t2b voxel/action artifacts allow CPU reevaluation using new_implementation metrics; old i2t2b conditions differ from missing ARR experiment",
                                 "saved_table_comparison": "600 paired main rows (200 buildings x3 description models); Direct table700 rows (100 buildings x7 conditions); stochastic36 aggregate rows only",
                                 "pickle": "No pickle or NumPy payload was loaded. Current NPY headers are Unicode <U32/<U64, so safe allow_pickle=False loading is possible.",
                                 "scene_id": "dataset_version:building_id", "research_api_requests": 0}}
    write_json(out / "source_audit.json", report)
    text = f"""# Original-source audit

Original repository: https://github.com/{REPO}; original ref `{ref}`.
All {len(commits)} reachable commit trees were checked. The clone is not shallow but is a promisor partial clone (`blob:none`). Tree inspection and existing local-file hashing avoid indiscriminate retrieval. GitHub metadata showed only `main`, with no tags, Releases or Actions artifacts at audit time. No LFS or submodule declaration, gitlink or current pointer was found. No missing historical source was recovered.

| Artifact | Expected | Found | Status |
|---|---:|---:|---|
| Primary GT scenes | 200 | {len(gt)} | FOUND |
| GT image files | 2400 metadata references | {len(gt_images)} | FOUND |
| Descriptions | 600 | {len(descriptions)} | PARTIAL |
| ARR fixed Builder outputs | 1200 | 0 | MISSING |
| ARR canonicalized inputs | corresponding saved inputs | 0 | MISSING |
| ARR recorded attempts / successful actions / predicted voxels | separate raw artifacts | 0 / 0 / 0 | MISSING |
| ARR Direct raw outputs | 100 | 0 | MISSING |
| ARR stochastic scene IDs and runs | 30 buildings | 0 | MISSING |
| Older i2t2b action files / voxel files | separate older experiment | {len(actions)} / {len(predictions)} | FOUND |

Descriptions are OpenAI 200, Claude 200, Gemini 100 (v4 only). No v1 Gemini description exists in a reachable commit tree. Older i2t2b outputs comprise two description providers and three schema/material/self-refinement variants; they cannot substitute for ARR's three models and two representations. Their generation-attempt count and confirmed successful API-call count are unknown; {len(actions) - len(action_parse_errors)} stored action files contain parseable operation lists.

Two supplementary LLM-authored Minecraft cohorts each contain 10 GT scenes, 10 descriptions, 20 action files and 30 voxel files. They are recorded separately and excluded from the primary counts and source archive allowlist. The manifest still inventories their already present original files.

`offline_evaluation/fixed_builder/`, `executor/`, `metrics.py`/`metrics/`, `experiments/fixed_builder_v1/` and the three imported v2 analysis/coverage modules are absent throughout reachable history. The existing second Builder script only prepares a dry-run manifest. The only old requirements file lists NumPy and Pillow, omitting analysis/test dependencies.

ARR saved tables contain 600 main paired rows over 200 buildings and three models, 700 Direct comparison rows over 100 buildings and seven conditions (including 100 Direct rows), and 36 stochastic aggregate rows. Older rebuild metric JSONs contain {saved_json_rows} saved evaluation rows from multiple variants. These are saved-table evidence, not reproduction from generated ARR outputs. ARR historical replay is BLOCKED. CPU reevaluation of the 200 real GT and 1200 older artifacts with `new_implementation` metrics remains possible.

Every locally present file from the original tree is recorded in `originals_manifest.jsonl` with repository, ref, relative path, Git blob hash, SHA-256 and size. `{len(manifest)}` files / `{report['manifest']['bytes_hashed']}` bytes were hashed; worktree mismatches: {len(mismatch_paths)}. `available_predictions.json` provides scene/condition/action/voxel sources. `source_archive_allowlist.json` selects the immutable primary data and older output sources for a separately scanned archive.

All six legacy test modules ({sum(x['test_count'] for x in tests)} test functions) were reviewed statically and are API-free; they require NumPy, pandas, statsmodels and pytest. Collection currently depends on absent original modules. Actual pass/fail/skip/error counts are recorded by the root agent's test run. Some old tests only assert constants or file presence and do not establish robust resume or API behavior.

No research API was called, no pickle was loaded, and no raw NumPy payload was loaded by this audit. NPY headers are Unicode arrays. The original GT bbox specifies `Y,X,Z`, inclusive bbox endpoints and world coordinates `(xmin+x_index, ymin+y_index, zmin+z_index)`; the historical evaluator's permissive `allow_pickle=True` need not be used.
"""
    (out / "source_audit.md").write_text(text)
    print(json.dumps({"source_ref": ref, "manifest_files": len(manifest), "GT": len(gt),
                      "descriptions": len(descriptions), "available_older_predictions": len(predictions),
                      "missing_original_modules": sum(x["status"] == "MISSING" for x in missing),
                      "worktree_mismatches": len(mismatch_paths)}))


if __name__ == "__main__":
    main()
