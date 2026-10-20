"""Bound-dependency packaging fixture; no production execution or live transport."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import contextlib
import hashlib
import json
import resource
import socket
import time

ROOT = Path("/root/minecraft-runs/goal015d-20261020T114450Z-ebe767ac")
WORK = ROOT / "tests/bound-deps"
started = time.monotonic()
cpu_started = time.process_time()
network_attempts = []


def denied(*args, **kwargs):
    network_attempts.append(True)
    raise AssertionError("External network forbidden in bound-dependency fixture")


socket.socket.connect = denied
socket.socket.connect_ex = denied
socket.create_connection = denied
socket.getaddrinfo = denied

from offline_evaluation.experiments.goal015d import planning, runtime, cli
from offline_evaluation.experiments.goal015d.tests.test_runtime import MockAdapter
from offline_evaluation.experiments.goal015d.tests.test_checkpoint import FakeGithub


def descriptor(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def references(value):
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= set(value):
            yield value
        else:
            for child in value.values():
                yield from references(child)
    elif isinstance(value, list):
        for child in value:
            yield from references(child)


production_path = ROOT / "plan.json"
production_descriptor = descriptor(production_path)
production = planning.verify_plan(production_path)
assert production["fixture_only"] is False
assert len(production["implementation_hashes"]) == 76
assert all(not (ROOT / name).exists() for name in
           ("enabled", "execution.lock", "completion.lock", "completion-progress.json"))

before = {str(production_path): production_descriptor}
for item in references(production):
    actual = descriptor(item["path"])
    assert actual == item
    before[actual["path"]] = actual
for name, digest in production["implementation_hashes"].items():
    actual = descriptor(name)
    assert actual["sha256"] == digest
    before[name] = actual
for source in production["gt_sources"].values():
    for relative, key in (("gt/voxels.npy", "voxels_sha256"),
                          ("gt/bbox.json", "bbox_sha256")):
        actual = descriptor(Path(source["building_directory"]) / relative)
        assert actual["sha256"] == source[key]
        before[actual["path"]] = actual
# Snapshot the existing price review/capture chain without changing any UTC.
pending = [Path(production["pricing_evidence"]["path"])]
seen = set()
while pending:
    path = pending.pop()
    if str(path) in seen or path.suffix != ".json":
        continue
    seen.add(str(path))
    document = json.loads(path.read_bytes())
    for item in references(document):
        actual = descriptor(item["path"])
        assert actual == item
        before[actual["path"]] = actual
        pending.append(Path(actual["path"]))
write_new(WORK / "BOUND_SOURCES_BEFORE.json",
          {"schema": "goal015d_bound_fixture_sources_v1",
           "captured_before_GT_parse_or_fixture_dispatch": True, "files": before})

# Only the new plan is marked fixture. Preserve original bound dependencies.
fixture = deepcopy(production)
fixture_root = WORK / "diagnostic"
fixture_root.mkdir()
(fixture_root / "inputs").mkdir()
(fixture_root / "wires").mkdir()
fixture.update(canonical_root=str(fixture_root), fixture_only=True)
for scene, source in fixture["inputs"].items():
    old = Path(source["input"]["path"])
    new = fixture_root / "inputs" / old.name
    with new.open("xb") as stream:
        stream.write(old.read_bytes())
    source["input"] = descriptor(new)
copied = {}
for row in fixture["schedule"]:
    old = Path(row["wire"]["path"])
    if str(old) not in copied:
        new = fixture_root / "wires" / old.name
        with new.open("xb") as stream:
            stream.write(old.read_bytes())
        copied[str(old)] = descriptor(new)
    row["wire"] = copied[str(old)]
    row["input"] = fixture["inputs"][row["scene_id"]]["input"]
fixture["plan_id"] = planning.plan_identity(fixture)
fixture_path = fixture_root / "plan.json"
write_new(fixture_path, fixture)
planning.verify_plan(fixture_path)

# Normal-empty synthetic output exercises actual GT loading/packaging cheaply;
# these are fixture observations, never real canary outcomes.
adapter = MockAdapter(text='{"operations":[]}')
with (WORK / "mock-progress.log").open("x") as progress, contextlib.redirect_stdout(progress):
    state = runtime.execute(fixture_path, adapter=adapter, confirmation=lambda _: True)
assert state["calls_settled"] == 16 and len(adapter.sent) == 16
assert state["checkpoint"]["status"] == "LOCAL_VERIFIED"
ledger = descriptor(fixture_root / "enabled/ledger.json")
package = runtime.read_json(Path(state["checkpoint"]["path"]) / "checkpoint.json")
github = FakeGithub(package["target_commitish"])
github.fail_download = True
first = cli.finish(fixture_path, allow_create=True, github=github)
assert first["status"] == "SAVE_FAILED"
checkpoint = first["checkpoint"]
github.fail_download = False
second = cli.finish(fixture_path, github=github)
assert second["status"] == "COMPLETE_SAVED"
assert second["checkpoint"] == checkpoint
assert descriptor(fixture_root / "enabled/ledger.json") == ledger
summary = second["evaluation"]
assert summary["row_count"] == 16 and len(summary["by_cell"]) == 8
assert summary["all"]["protocol_valid"] == summary["all"]["normal_empty"] == 16
assert summary["all"]["nonempty"] == 0
saved_only = cli.save_only(
    fixture_path, checkpoint_dir=checkpoint["path"],
    verification_dir=fixture_root / "github-verification/explicit-save-only",
    github=github)
assert saved_only["status"] == "SAVED_ONLY" and len(adapter.sent) == 16
assert descriptor(fixture_root / "enabled/ledger.json") == ledger
assert sum(stage == "release_create" for stage, _, _ in github.calls) == 1
assert sum(stage == "asset_upload" for stage, _, _ in github.calls) == 1
assert not network_attempts

verification = runtime.read_json(second["github_verification"]["path"])
restore = Path(second["github_verification"]["path"]).parent / "restored"
manifest = runtime.read_json(restore / "FILES.json")
by_original = {path: restore / member
               for member, path in manifest["original_paths"].items()}
for name, digest in production["implementation_hashes"].items():
    assert descriptor(by_original[name])["sha256"] == digest
for source in production["gt_sources"].values():
    for relative, key in (("gt/voxels.npy", "voxels_sha256"),
                          ("gt/bbox.json", "bbox_sha256")):
        original = str(Path(source["building_directory"]) / relative)
        assert descriptor(by_original[original])["sha256"] == source[key]
for source in fixture["inputs"].values():
    assert by_original[source["input"]["path"]].read_bytes() == Path(source["input"]["path"]).read_bytes()
for row in fixture["schedule"]:
    assert by_original[row["wire"]["path"]].read_bytes() == Path(row["wire"]["path"]).read_bytes()
assert manifest["external_dependencies"]["python_environment_bundled"] is False
after = {name: descriptor(name) for name in before}
assert before == after
write_new(WORK / "BOUND_SOURCES_AFTER.json",
          {"schema": "goal015d_bound_fixture_sources_v1", "files": after})
assert all(not (ROOT / name).exists() for name in
           ("enabled", "execution.lock", "completion.lock", "completion-progress.json"))

receipt = {
    "schema": "goal015d_final_checkpoint_audit_v1",
    "status": "PASS", "checked_utc": datetime.now(timezone.utc).isoformat(),
    "real_plan": production_descriptor, "fixture_plan": descriptor(fixture_path),
    "scope": "Actual frozen dependencies with explicit fixture-only plan and synthetic normal-empty responses; not real canary outcomes.",
    "source_script": descriptor(__file__),
    "source_before": descriptor(WORK / "BOUND_SOURCES_BEFORE.json"),
    "source_after": descriptor(WORK / "BOUND_SOURCES_AFTER.json"),
    "bound_source_files_unchanged": len(before),
    "mock_calls": 16, "mock_normal_empty_protocol_valid": 16,
    "real_research_API_calls": 0, "real_approvals": 0,
    "real_Draft_operations": 0, "production_launcher_executed": False,
    "host_credentials_or_environment_inspected": False,
    "network_attempts": len(network_attempts),
    "actual_bound_implementation_files_restored": 76,
    "actual_GT_files_restored": 4, "actual_FULL_inputs_restored": 2,
    "actual_unique_wires_restored": 8,
    "actual_pricing_documents": 11,
    "restored_file_count": len(manifest["files"]),
    "evaluation": summary["evaluation_artifact"],
    "evaluation_summary": summary["summary_artifact"],
    "checkpoint": checkpoint,
    "verification": second["github_verification"],
    "restoration": verification["restoration"],
    "fake_drafts": 1, "fake_assets": 1,
    "failed_download_then_same_checkpoint_retry": True,
    "save_only_after_finish_without_regeneration": True,
    "ledger_exact_bytes_unchanged_by_save": True,
    "original_source_bytes_unchanged": True,
    "real_plan_controls_remain_absent": True,
    "protected_C_release_and_asset_contacted": False,
    "external_dependencies": manifest["external_dependencies"],
    "limitations": [
        "Fake GitHub transport proves local orchestration, hashing and restoration, not live remote access.",
        "Model responses are synthetic empty outputs; no provider behavior or research outcome is measured.",
        "Python environment is external; deeper historical provenance references are not recursively bundled."
    ],
    "resources": {"cpu_seconds": time.process_time()-cpu_started,
                  "elapsed_seconds": time.monotonic()-started,
                  "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
}
write_new(ROOT / "FINAL_CHECKPOINT_AUDIT.json", receipt)
print(json.dumps(descriptor(ROOT / "FINAL_CHECKPOINT_AUDIT.json")))
