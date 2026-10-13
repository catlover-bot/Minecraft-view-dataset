import fcntl
import json
from pathlib import Path
import socket
import tarfile

import pytest

from offline_evaluation.server_v2 import checkpoint, goal003_forensic as forensic
from offline_evaluation.server_v2.io_utils import read_json, sha256


@pytest.fixture
def evidence(tmp_path):
    paid, terminal, support = (tmp_path / name for name in ("paid", "terminal", "support"))
    for directory in (paid, terminal, support):
        directory.mkdir()
    included = {}
    for name in ("generation.py", "analysis_v4.py", "goal003_forensic.py", "tests/test_goal003_forensic.py"):
        path = support / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# immutable fixture code\n")
        included["code/offline_evaluation/server_v2/" + name] = path
    for prefix in ("diagnosis", "reports"):
        path = support / (prefix + ".json")
        path.write_text('{"status":"PARTIAL"}\n')
        included[prefix + "/status.json"] = path
    bound = {"generation.py": sha256(support / "generation.py")}
    identity = {"campaign_id": "fixture-campaign", "ledger_directory": str(paid),
                "implementation_hashes": bound}
    (paid / "campaign.json").write_text(json.dumps({"identity": identity, "identity_hash": "a" * 64}))
    (paid / "ledger.json").write_text('{"calls":14,"retries":4}\n')
    for provider in ("openai", "anthropic"):
        run = paid / "runs" / provider
        run.mkdir(parents=True)
        (run / "manifest.json").write_text(json.dumps({"identity": identity}))
    raw = paid / "runs/openai/attempts" / ("c" * 32) / "raw.json"
    raw.parent.mkdir(parents=True)
    raw.write_text('{"response_model":"gpt-5-mini-2025-08-07"}\n')
    (terminal / "generation.log").write_text("GENERATION_EXIT_CODE=0\n")
    for name in forensic.CONTROLS:
        (paid / name.removeprefix("paid_smoke/")).touch()
    files, controls = [], []
    roots = {"paid_smoke": paid, "terminal_logs": terminal}
    for root, directory in roots.items():
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                name = root + "/" + path.relative_to(directory).as_posix()
                row = {"path": name, "size": path.stat().st_size, "sha256": sha256(path)}
                (controls if name in forensic.CONTROLS else files).append(row)
    baseline = {"roots": {name: str(path) for name, path in roots.items()},
                "files": files, "control_files": controls, "evidence_count": len(files)}
    return {"destination": tmp_path / "snapshot", "baseline": baseline, "included_files": included,
            "commit": "d" * 40, "snapshot_id": "goal003-fixture", "source_reference": dict(forensic.SOURCE_ID)}


def test_forensic_roundtrip_preserves_originals_and_canonical_binding(evidence, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("Forensic packaging must not open a network connection")
    monkeypatch.setattr(socket, "create_connection", no_network)
    original_bytes = {}
    for row in evidence["baseline"]["files"] + evidence["baseline"]["control_files"]:
        root, relative = row["path"].split("/", 1)
        path = Path(evidence["baseline"]["roots"][root]) / relative
        original_bytes[path] = path.read_bytes()
    result = forensic.create_forensic_snapshot(**evidence)
    assert result["experiment_completion"] == "PARTIAL"
    assert result["archive_completeness"] == "COMPLETE"
    assert result["kind"] == "read_only_forensic" and result["resumable"] is False
    assert all(path.read_bytes() == data for path, data in original_bytes.items())
    assert not (Path(evidence["baseline"]["roots"]["paid_smoke"]) / "completed.json").exists()
    archive = evidence["destination"] / "forensic.tar.gz"
    destination = evidence["destination"].parent / "restored"
    restored = checkpoint.restore_archive(archive, destination, result["assets"][0]["sha256"])
    assert restored["files"] == len(evidence["baseline"]["files"]) + len(evidence["included_files"])
    campaign = read_json(destination / "originals/paid_smoke/campaign.json")
    assert campaign["identity"]["ledger_directory"] == evidence["baseline"]["roots"]["paid_smoke"]
    with tarfile.open(archive, "r:gz") as source:
        assert not any(name.endswith(".lock") or name.endswith("completed.json") for name in source.getnames())
        manifest = json.load(source.extractfile("FILES.json"))
    assert len(manifest["metadata"]["controls_excluded_from_archive"]) == 4
    for line in (evidence["destination"] / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ")
        assert sha256(evidence["destination"] / name) == digest
    assert not (evidence["destination"] / "sources.tar.gz").exists()


def test_changed_original_rejected_before_any_output(evidence):
    (Path(evidence["baseline"]["roots"]["paid_smoke"]) / "ledger.json").write_text('{"calls":0}')
    with pytest.raises(ValueError, match="changed"):
        forensic.create_forensic_snapshot(**evidence)
    assert not evidence["destination"].exists()


@pytest.mark.parametrize("name", ["approval.json", "unknown.txt"])
def test_unlisted_original_cannot_be_silently_dropped(evidence, name):
    (Path(evidence["baseline"]["roots"]["paid_smoke"]) / name).write_text("unlisted")
    with pytest.raises(ValueError, match="allowlist"):
        forensic.create_forensic_snapshot(**evidence)
    assert not evidence["destination"].exists()


def test_even_explicit_unknown_original_path_is_rejected(evidence):
    paid = Path(evidence["baseline"]["roots"]["paid_smoke"])
    path = paid / "arbitrary.txt"
    path.write_text("not research evidence")
    evidence["baseline"]["files"].append({"path": "paid_smoke/arbitrary.txt", "sha256": sha256(path), "size": path.stat().st_size})
    evidence["baseline"]["evidence_count"] += 1
    with pytest.raises(ValueError, match="Unknown original"):
        forensic.create_forensic_snapshot(**evidence)


def test_unknown_original_directory_is_not_silently_ignored(evidence):
    (Path(evidence["baseline"]["roots"]["paid_smoke"]) / "new-output").mkdir()
    with pytest.raises(ValueError, match="allowlist"):
        forensic.create_forensic_snapshot(**evidence)


def test_symlink_in_original_tree_is_rejected(evidence):
    paid = Path(evidence["baseline"]["roots"]["paid_smoke"])
    (paid / "linked-directory").symlink_to(paid / "runs", target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        forensic.create_forensic_snapshot(**evidence)


def test_symlink_in_additional_source_parent_is_rejected(evidence):
    source = evidence["included_files"]["diagnosis/status.json"]
    link = evidence["destination"].parent / "linked-support"
    link.symlink_to(source.parent, target_is_directory=True)
    evidence["included_files"]["diagnosis/status.json"] = link / source.name
    with pytest.raises(ValueError, match="Symlink"):
        forensic.create_forensic_snapshot(**evidence)


def test_output_cannot_be_inside_original_tree(evidence):
    evidence["destination"] = Path(evidence["baseline"]["roots"]["paid_smoke"]) / "snapshot"
    with pytest.raises(ValueError, match="overlap"):
        forensic.create_forensic_snapshot(**evidence)
    assert not evidence["destination"].exists()


def test_existing_output_is_never_overwritten(evidence):
    evidence["destination"].mkdir()
    marker = evidence["destination"] / "existing"
    marker.write_bytes(b"preserve")
    with pytest.raises(FileExistsError, match="no overwrites"):
        forensic.create_forensic_snapshot(**evidence)
    assert marker.read_bytes() == b"preserve"


def test_missing_control_lock_is_not_created(evidence):
    lock = Path(evidence["baseline"]["roots"]["paid_smoke"]) / "budget.lock"
    lock.unlink()
    with pytest.raises(FileNotFoundError):
        forensic.create_forensic_snapshot(**evidence)
    assert not lock.exists() and not evidence["destination"].exists()


def test_active_generation_lock_refuses_snapshot(evidence):
    lock = Path(evidence["baseline"]["roots"]["paid_smoke"]) / "run.lock"
    with lock.open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Active original run lock"):
            forensic.create_forensic_snapshot(**evidence)
    assert not evidence["destination"].exists()


def test_changed_hashbound_code_cannot_be_packaged_as_original_implementation(evidence):
    evidence["included_files"]["code/offline_evaluation/server_v2/generation.py"].write_text("# changed generation\n")
    with pytest.raises(ValueError, match="saved implementation hash"):
        forensic.create_forensic_snapshot(**evidence)
    assert not evidence["destination"].exists()


def test_secret_like_content_refused_without_printing_value(evidence):
    secret = "sk-" + "x" * 40
    evidence["included_files"]["diagnosis/status.json"].write_text(secret)
    with pytest.raises(ValueError, match="value suppressed") as error:
        forensic.create_forensic_snapshot(**evidence)
    assert secret not in str(error.value)
    assert not evidence["destination"].exists()


def test_source_reference_cannot_redirect_to_mutable_or_other_asset(evidence):
    evidence["source_reference"]["asset_id"] += 1
    with pytest.raises(ValueError, match="Immutable source"):
        forensic.create_forensic_snapshot(**evidence)


def test_copied_enabled_campaign_is_not_accepted_as_canonical_original(evidence):
    campaign_path = Path(evidence["baseline"]["roots"]["paid_smoke"]) / "campaign.json"
    campaign = read_json(campaign_path)
    campaign["identity"]["ledger_directory"] = "/root/minecraft-runs/different-canonical-original"
    campaign_path.write_text(json.dumps(campaign))
    for row in evidence["baseline"]["files"]:
        if row["path"] == "paid_smoke/campaign.json":
            row.update(size=campaign_path.stat().st_size, sha256=sha256(campaign_path))
    with pytest.raises(ValueError, match="canonical original"):
        forensic.create_forensic_snapshot(**evidence)
    assert not evidence["destination"].exists()


def test_original_change_during_packaging_refuses_success(evidence, monkeypatch):
    archive = checkpoint._archive
    def change_after_read(*args, **kwargs):
        result = archive(*args, **kwargs)
        (Path(evidence["baseline"]["roots"]["paid_smoke"]) / "ledger.json").write_text('{"calls":0}')
        return result
    monkeypatch.setattr(checkpoint, "_archive", change_after_read)
    with pytest.raises(ValueError, match="changed"):
        forensic.create_forensic_snapshot(**evidence)
    assert not (evidence["destination"] / "forensic.json").exists()
