"""Real local checkpoint/evaluation with fixture-only GitHub and model transport."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from offline_evaluation.experiments.goal015d import checkpoint as c, evaluate as e, runtime as r
from offline_evaluation.experiments.goal015d.tests.test_runtime import fixture_plan, MockAdapter


class FakeGithub:
    def __init__(self, commit):
        self.commit = commit
        self.releases, self.assets, self.blobs, self.calls = [], {}, {}, []
        self.fail_download = False

    def call(self, stage, args, output=None):
        self.calls.append((stage, args, output))
        if stage == "commit_ancestry":
            return {"status": "identical", "behind_by": 0, "base_commit": {"sha": self.commit}}
        if stage == "release_create":
            release = dict(json.loads(Path(args[-1]).read_text()), id=123, published_at=None)
            self.releases.append(release)
            return deepcopy(release)
        if stage == "release_lookup":
            return deepcopy(next(row for row in self.releases if row["id"] == int(args[-1].split("/")[-1])))
        if stage == "asset_upload":
            data = Path(args[-1]).read_bytes()
            row = {"id": 456, "name": "goal015d-state.tar.gz", "size": len(data),
                   "digest": "sha256:" + hashlib.sha256(data).hexdigest(), "state": "uploaded"}
            self.assets[123] = [row]
            self.blobs[456] = data
            return deepcopy(row)
        if stage == "asset_download":
            assert "/releases/assets/456" in args[1]
            if self.fail_download:
                raise RuntimeError("fixture interrupted download")
            Path(output).write_bytes(self.blobs[456])
            return None
        raise AssertionError(stage)

    def pages(self, endpoint, stage):
        self.calls.append((stage, endpoint, None))
        if stage == "release_list":
            return deepcopy(self.releases)
        if stage == "asset_list":
            return deepcopy(self.assets.get(int(endpoint.split("/")[-2]), []))
        raise AssertionError(stage)


def prepared(tmp_path):
    path = fixture_plan(tmp_path)
    adapter = MockAdapter()
    status = r.execute(path, adapter=adapter, confirmation=lambda _: True)
    assert status["calls_settled"] == 16 and status["checkpoint"]["status"] == "LOCAL_VERIFIED"
    summary = e.evaluate_plan(path)
    saved = c.create_save_only(path)
    directory = Path(saved["path"])
    package = r.read_json(directory / "checkpoint.json")
    return path, adapter, summary, saved, directory, package


def test_mock16_evaluate_archive_upload_failure_then_save_only_retry(tmp_path):
    path, adapter, summary, saved, directory, package = prepared(tmp_path)
    ledger_before = (path.parent / "enabled/ledger.json").read_bytes()
    restored = directory / "verified-restoration"
    evaluation = r.bound(summary["evaluation_artifact"])
    assert len(evaluation["rows"]) == 16
    assert len(summary["by_cell"]) == len(summary["paired_differences"]) == 8
    assert all(cell["planned"] == 2 for cell in summary["by_cell"])
    assert summary["all"]["protocol_valid"] == summary["all"]["nonempty"] == 16
    assert all("task_pass" not in row for row in evaluation["rows"])
    assert package["restoration"]["evaluation_included"]
    assert package["restoration"]["wire_payloads_and_bound_file_dependencies_included"]
    assert package["restoration"]["gt_source_files_included"] == 4
    plan = r.read_json(path)
    assert len({row["wire"]["path"] for row in plan["schedule"]}) == 8
    assert len({row["pair_id"] for row in plan["schedule"]}) == 8
    for row in plan["schedule"]:
        assert (restored / Path(row["wire"]["path"]).relative_to(path.parent)).read_bytes() == Path(row["wire"]["path"]).read_bytes()
    manifest = r.read_json(restored / "FILES.json")
    restored_paths = {source: restored / member
                      for member, source in manifest["original_paths"].items()}
    for source in plan["inputs"].values():
        original = source["input"]
        assert restored_paths[original["path"]].read_bytes() == Path(original["path"]).read_bytes()
    for source in plan["gt_sources"].values():
        for name, key in (("voxels.npy", "voxels_sha256"), ("bbox.json", "bbox_sha256")):
            original = str(Path(source["building_directory"]) / "gt" / name)
            assert hashlib.sha256(restored_paths[original].read_bytes()).hexdigest() == source[key]
    assert (restored / Path(summary["evaluation_artifact"]["path"]).relative_to(path.parent)).read_bytes() == Path(summary["evaluation_artifact"]["path"]).read_bytes()
    github = FakeGithub(package["target_commitish"])
    github.fail_download = True
    with pytest.raises(RuntimeError, match="fixture interrupted"):
        c.upload_checkpoint(directory, tmp_path / "failed-save", allow_create=True, github=github)
    assert r.read_json(tmp_path / "failed-save/SAVE_FAILURE.json")["status"] == "SAVE_FAILED"
    github.fail_download = False
    receipt = c.upload_checkpoint(directory, tmp_path / "retried-save", github=github)
    assert receipt["status"] == "GITHUB_VERIFIED" and receipt["draft"] and not receipt["uploaded"]
    assert receipt["restoration"]["evaluation_included"] and receipt["restoration"]["planned_calls"] == 16
    assert sum(stage == "release_create" for stage, _, _ in github.calls) == 1
    assert sum(stage == "asset_upload" for stage, _, _ in github.calls) == 1
    assert len(github.releases) == 1 and github.releases[0]["body"].startswith("Goal015D immutable")
    assert (path.parent / "enabled/ledger.json").read_bytes() == ledger_before
    assert len(adapter.sent) == 16
    r._save(tmp_path / "MOCK16_SAVE_END_TO_END_RECEIPT.json",
            {"status": "PASS", "fixture_only": True, "actual_research_api_requests": 0,
             "planned_mock_calls": 16, "settled_mock_calls": 16,
             "unique_wire_payloads": 8, "full_inputs": 2, "gt_source_files": 4,
             "evaluation": summary["evaluation_artifact"], "checkpoint": saved["checkpoint"],
             "archive_sha256": receipt["archive_sha256"], "fake_drafts": 1, "fake_assets": 1,
             "save_retry_preserves_ledger_exact_bytes": True,
             "redownload_and_safe_restore": receipt["restoration"]}, exclusive=True)


def test_missing_draft_requires_explicit_creation(tmp_path):
    _, _, _, _, directory, package = prepared(tmp_path)
    github = FakeGithub(package["target_commitish"])
    with pytest.raises(r.RuntimeErrorSafe, match="explicit creation"):
        c.upload_checkpoint(directory, tmp_path / "refused", github=github)
    assert not github.releases


def test_repeated_save_only_does_not_expand_checkpoint_history(tmp_path):
    path, adapter, _, _, directory, _ = prepared(tmp_path)
    before = (path.parent / "enabled/ledger.json").read_bytes()
    second = c.create_save_only(path)
    archive = Path(second["path"]) / "goal015d-state.tar.gz"
    assert archive.read_bytes() == (directory / "goal015d-state.tar.gz").read_bytes()
    with tarfile.open(archive) as saved:
        names = saved.getnames()
    assert sum(name.startswith("checkpoints/") for name in names) == 1
    assert not any(name.endswith(".tar.gz") for name in names)
    assert (path.parent / "enabled/ledger.json").read_bytes() == before and len(adapter.sent) == 16


def test_unknown_transport_and_lower_fixture_budget_remain_saveable(tmp_path):
    path = fixture_plan(tmp_path, cap="0.40")
    adapter = MockAdapter(transport_at=1)
    status = r.execute(path, adapter=adapter, confirmation=lambda _: True)
    assert status["calls_uncertain"] == 1 and status["calls_not_sent"] == 15
    assert status["checkpoint"]["status"] == "LOCAL_VERIFIED"
    summary = e.evaluate_plan(path)
    rows = r.bound(summary["evaluation_artifact"])["rows"]
    assert len(rows) == 16 and all(row["nonempty"] is None for row in rows)
    assert summary["all"]["planned"] == 16 and summary["all"]["metric_rows"] == 0
    assert all(value["B_minus_A"] is None
               for pair in summary["paired_differences"] for value in pair["metrics"].values())
    saved = c.create_save_only(path)
    assert r.bound(saved["checkpoint"])["restoration"]["evaluation_included"]
    assert len(adapter.sent) == 1


def test_completed_goal015c_release_is_refused_before_any_github_access(tmp_path):
    github = FakeGithub("0" * 40)
    with pytest.raises(r.RuntimeErrorSafe, match="Goal015C Draft is protected"):
        c.upload_checkpoint(tmp_path / "absent-package", tmp_path / "not-created",
                            release_id=418290999, allow_create=True, github=github)
    assert github.calls == [] and not (tmp_path / "not-created").exists()


@pytest.mark.parametrize("change", [{"draft": False}, {"published_at": "2026-01-01"}, {"tag_name": "old-draft"}])
def test_existing_release_identity_never_overwritten(tmp_path, change):
    _, _, _, _, directory, package = prepared(tmp_path)
    github = FakeGithub(package["target_commitish"])
    c.upload_checkpoint(directory, tmp_path / "first", allow_create=True, github=github)
    github.releases[0].update(change)
    with pytest.raises(r.RuntimeErrorSafe, match="mismatch"):
        c.upload_checkpoint(directory, tmp_path / "refused", release_id=123,
                            allow_create=True, github=github)
    assert sum(stage == "release_create" for stage, _, _ in github.calls) == 1
    assert sum(stage == "asset_upload" for stage, _, _ in github.calls) == 1


def repack(directory, destination, change):
    with tarfile.open(directory / "goal015d-state.tar.gz") as archive:
        contents = {member.name: archive.extractfile(member).read() for member in archive.getmembers()}
    manifest = json.loads(contents["FILES.json"])
    change(contents, manifest)
    for name in manifest["files"]:
        manifest["files"][name] = {"bytes": len(contents[name]), "sha256": hashlib.sha256(contents[name]).hexdigest()}
    contents["FILES.json"] = json.dumps(manifest).encode()
    with tarfile.open(destination, "w:gz") as archive:
        for name, data in contents.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


@pytest.mark.parametrize("control,key,value", [
    ("enabled/approval.json", "approved_budget_usd", "7.00"),
    ("enabled/authority.json", "metadata_binding_sha256", "0" * 64),
    ("enabled/ledger.json", "plan_id", "0" * 64)])
def test_rehashed_archive_controls_rejected(tmp_path, control, key, value):
    _, _, _, _, directory, _ = prepared(tmp_path)
    def change(contents, manifest):
        data = json.loads(contents[control])
        data[key] = value
        contents[control] = json.dumps(data).encode()
    corrupted = tmp_path / "rehashed.tar.gz"
    repack(directory, corrupted, change)
    with pytest.raises(r.RuntimeErrorSafe):
        c.verify_archive(corrupted, r.descriptor(corrupted)["sha256"], destination=tmp_path / "refused")
    assert not (tmp_path / "refused").exists()


def test_rehashed_gt_source_rejected_before_restore(tmp_path):
    _, _, _, _, directory, _ = prepared(tmp_path)
    def change(contents, manifest):
        target = next(name for name, original in manifest["original_paths"].items()
                      if original.endswith("/gt/bbox.json"))
        contents[target] = b'{"changed":true}\n'
    corrupted = tmp_path / "rehashed-gt.tar.gz"
    repack(directory, corrupted, change)
    with pytest.raises(r.RuntimeErrorSafe, match="GT source absent or changed"):
        c.verify_archive(corrupted, r.descriptor(corrupted)["sha256"],
                         destination=tmp_path / "refused")
    assert not (tmp_path / "refused").exists()


@pytest.mark.parametrize("name,link", [("../escape", False), ("/absolute", False), ("link", True)])
def test_traversal_and_links_refused_before_restore(tmp_path, name, link):
    path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo(name)
        if link:
            member.type, member.linkname = tarfile.SYMTYPE, "/etc/passwd"
            archive.addfile(member)
        else:
            member.size = 1
            archive.addfile(member, io.BytesIO(b"x"))
        raw = b'{"files":{}}'
        member = tarfile.TarInfo("FILES.json")
        member.size = len(raw)
        archive.addfile(member, io.BytesIO(raw))
    with pytest.raises((ValueError, r.RuntimeErrorSafe)):
        c.verify_archive(path, r.descriptor(path)["sha256"], destination=tmp_path / "refused")
    assert not (tmp_path / "refused").exists()


def test_checkpoint_import_has_no_provider_or_generation_import():
    script = """import sys
from offline_evaluation.experiments.goal015d import checkpoint
assert not any(name.endswith('.providers') or name.endswith('.generation')
               or name.endswith('goal015.adapter') or name.endswith('goal015d.adapter')
               for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
