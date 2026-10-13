"""Allowlisted read-only evidence snapshot for the partial Goal002 paid smoke.

This module never activates or resumes a campaign. Original campaign and ledger
bytes retain their canonical-directory binding, including in restored evidence.
"""
from __future__ import annotations

from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import re
import stat

from . import checkpoint
from .io_utils import atomic_json, read_json, sha256, writable


SOURCE_ID = {
    "repo": "catlover-bot/Minecraft-view-dataset",
    "name": "sources.tar.gz",
    "release_id": 411682582,
    "asset_id": 637059866,
    "sha256": "97b5d9f0862449c881cd5804485f1a27cc5164baf3f51df09c66c2658ac3d6da",
}
CONTROLS = {
    "paid_smoke/budget.lock", "paid_smoke/run.lock",
    "paid_smoke/runs/openai/run.lock", "paid_smoke/runs/anthropic/run.lock",
}
TERMINAL_NAMES = {
    "approval.json", "approval_template.json", "activation_result.json",
    "generation.log", "evaluate_openai.log", "evaluate_anthropic.log",
    "summary.log", "final_status.log",
}
PAID_PATH = re.compile(
    r"(?:campaign|approval|ledger)\.json|runs/(?:openai|anthropic)/(?:"
    r"(?:manifest|config|state|inputs|evaluation)\.json|prompt\.txt|"
    r"attempts/[0-9a-f]{32}/(?:attempt|raw|parsed|voxels)\.json|"
    r"evaluations/[0-9a-f]{64}/metrics\.json)"
)


def _plain_path(value):
    """Reject links in every existing component before resolving a source."""
    path = Path(value).absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError("Symlink source or destination component rejected")
    return path.resolve()


def _record(name, path):
    path = _plain_path(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Only regular source files are allowed")
    return {"path": name, "sha256": sha256(path), "size": path.stat().st_size}


def _row_map(rows):
    result = {}
    for row in rows:
        name = row["path"]
        relative = checkpoint.safe_relative(name)
        if relative.as_posix() != name or name in result:
            raise ValueError("Duplicate or noncanonical baseline path")
        if (not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
                or type(row["size"]) is not int or row["size"] < 0):
            raise ValueError("Invalid baseline hash or size")
        result[name] = {key: row[key] for key in ("path", "sha256", "size")}
    return result


def _baseline_inputs(baseline):
    roots = {name: _plain_path(value) for name, value in baseline["roots"].items()}
    if set(roots) != {"paid_smoke", "terminal_logs"}:
        raise ValueError("Exactly the paid smoke and terminal log roots are required")
    if any(not path.is_dir() for path in roots.values()):
        raise ValueError("Original evidence roots must already exist")
    paid, terminal = roots["paid_smoke"], roots["terminal_logs"]
    if paid.is_relative_to(terminal) or terminal.is_relative_to(paid):
        raise ValueError("Original evidence roots must not overlap")
    evidence = _row_map(baseline["files"])
    controls = _row_map(baseline["control_files"])
    if set(controls) != CONTROLS or not evidence or set(evidence) & set(controls):
        raise ValueError("Explicit evidence and four existing control locks are required")
    for name in evidence:
        root, relative = name.split("/", 1)
        if not ((root == "paid_smoke" and PAID_PATH.fullmatch(relative))
                or (root == "terminal_logs" and relative in TERMINAL_NAMES)):
            raise ValueError("Unknown original evidence path rejected")
    if baseline.get("evidence_count", len(evidence)) != len(evidence):
        raise ValueError("Baseline evidence count mismatch")
    return roots, evidence, controls


def _original_entries(roots, evidence, controls):
    expected = evidence | controls
    observed = {}
    paths = {}
    for root_name, directory in roots.items():
        expected_directories = {
            parent.as_posix()
            for name in expected if name.startswith(root_name + "/")
            for parent in Path(name.removeprefix(root_name + "/")).parents
        }
        for path in sorted(directory.rglob("*")):
            # Test links before is_dir(), which follows directory links.
            if path.is_symlink():
                raise ValueError("Symlink in original evidence tree rejected")
            if path.is_dir():
                if path.relative_to(directory).as_posix() not in expected_directories:
                    raise ValueError("Original directory differs from explicit baseline allowlist")
                continue
            name = root_name + "/" + path.relative_to(directory).as_posix()
            if name not in expected:
                raise ValueError("Original tree differs from explicit baseline allowlist")
            observed[name] = _record(name, path)
            paths[name] = path
    if observed != expected:
        raise ValueError("Original evidence changed or is missing since baseline")
    return [("originals/" + name, paths[name]) for name in sorted(evidence)]


def _code_binding(paid_root, included):
    identities = [read_json(paid_root / "campaign.json")["identity"]]
    identities += [read_json(paid_root / "runs" / provider / "manifest.json")["identity"]
                   for provider in ("openai", "anthropic")]
    for identity in identities:
        bound = identity.get("implementation_hashes")
        if not isinstance(bound, dict) or not bound:
            raise ValueError("Saved implementation hashes are required")
        for name, digest in bound.items():
            if not re.fullmatch(r"[a-z0-9_]+\.py", name):
                raise ValueError("Invalid saved implementation filename")
            key = "code/offline_evaluation/server_v2/" + name
            if key not in included or sha256(included[key]) != digest:
                raise ValueError("Snapshot code does not match saved implementation hash")
    for name in ("analysis_v4.py", "goal003_forensic.py", "tests/test_goal003_forensic.py"):
        if "code/offline_evaluation/server_v2/" + name not in included:
            raise ValueError("Corrected analysis and forensic code/tests must be included")


def create_forensic_snapshot(destination, *, baseline, included_files, commit,
                             snapshot_id, source_reference):
    """Write a new snapshot without writing even control bytes to the originals.

    ``baseline`` is a mapping or JSON path with roots, files and control_files;
    each file row has path, sha256 and size. ``included_files`` explicitly maps
    archive paths under diagnosis/, reports/ or code/ to regular source files.
    The caller must finish and verify all diagnosis/report inputs beforehand.
    This does not publish, call a research transport, or download old sources.
    """
    baseline = read_json(baseline) if isinstance(baseline, (str, Path)) else baseline
    source_reference = (read_json(source_reference)
                        if isinstance(source_reference, (str, Path)) else source_reference)
    roots, evidence, controls = _baseline_inputs(baseline)
    destination = writable(_plain_path(destination))
    if destination.exists():
        raise FileExistsError("Forensic destination must be new; no overwrites")
    if any(destination.is_relative_to(root) or root.is_relative_to(destination)
           for root in roots.values()):
        raise ValueError("Forensic output must not overlap original evidence")
    if any(source_reference.get(key) != value for key, value in SOURCE_ID.items()):
        raise ValueError("Immutable source asset identity mismatch")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("An exact code commit SHA is required")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", snapshot_id):
        raise ValueError("Invalid snapshot ID")
    included = {}
    for name, source in included_files.items():
        relative = checkpoint.safe_relative(name)
        if relative.as_posix() != name or relative.parts[0] not in {"diagnosis", "reports", "code"}:
            raise ValueError("Only explicit diagnosis, report and code additions are allowed")
        path = _plain_path(source)
        if path.is_relative_to(destination) or any(path.is_relative_to(root) for root in roots.values()):
            raise ValueError("Additional source overlaps snapshot output or original evidence")
        _record(name, path)
        included[name] = path
    if not all(any(name.startswith(prefix + "/") for name in included)
               for prefix in ("diagnosis", "reports", "code")):
        raise ValueError("Diagnosis, report and code closure are all required")

    with ExitStack() as stack:
        for name in sorted(controls):
            root, relative = name.split("/", 1)
            lock = _plain_path(roots[root] / relative)
            if _record(name, lock) != controls[name]:
                raise ValueError("Control file changed since baseline")
            stream = stack.enter_context(lock.open("rb"))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Active original run lock; forensic snapshot refused") from None
        entries = _original_entries(roots, evidence, controls)
        _code_binding(roots["paid_smoke"], included)
        entries += sorted(included.items())
        expected = {name: _record(name, path) for name, path in entries}
        for _, path in entries:
            checkpoint.scan_file(path)
        campaign = read_json(roots["paid_smoke"] / "campaign.json")
        if campaign["identity"]["ledger_directory"] != str(roots["paid_smoke"]):
            raise ValueError("Only the canonical original campaign directory may be snapshotted")
        metadata = {
            "schema": "goal003_forensic_snapshot_v1", "snapshot_id": snapshot_id,
            "commit": commit, "kind": "read_only_forensic", "resumable": False,
            "archive_completeness": "COMPLETE", "experiment_completion": "PARTIAL",
            "original_roots": {name: str(path) for name, path in roots.items()},
            "campaign_id": campaign["identity"]["campaign_id"],
            "campaign_identity_hash": campaign["identity_hash"],
            "canonical_ledger_directory": campaign["identity"]["ledger_directory"],
            "original_evidence_count": len(evidence),
            "controls_excluded_from_archive": list(controls.values()),
            "sources": source_reference,
            "research_transport_invoked_by_snapshot": False,
        }
        destination.mkdir(parents=True, exist_ok=False)
        asset = checkpoint._archive(destination / "forensic.tar.gz", entries, metadata)
        if {row["path"]: row for row in asset["manifest"]} != expected:
            raise ValueError("Archived source changed since initial verification")
        _original_entries(roots, evidence, controls)
        if any(_record(name, path) != expected[name] for name, path in entries):
            raise ValueError("Snapshot source changed during archive creation")
        atomic_json(destination / "forensic.tar.gz.files.json", asset.pop("manifest"))
        atomic_json(destination / "originals.sha256.json", {
            "status": "UNCHANGED", "files": list(evidence.values()),
            "control_files": list(controls.values()),
        })
        atomic_json(destination / "source_reference.json", source_reference)
        result = dict(metadata, assets=[asset], status="LOCAL_VERIFIED_PENDING_UPLOAD")
        atomic_json(destination / "forensic.json", result)
        (destination / "RESTORE.md").write_text(
            "# Read-only forensic evidence\n\n"
            "This archive is complete for its explicit allowlist. The paid smoke experiment is PARTIAL.\n"
            "Verify all downloaded assets against SHA256SUMS. Restore forensic.tar.gz with "
            "offline_evaluation.server_v2.checkpoint.restore_archive(archive, NEW_DIRECTORY, expected_sha256). "
            "The extractor checks the archive and every file, and refuses existing destinations.\n\n"
            "Restored originals are evidence only, NOT a generation-resume clone. "
            "Campaign, approval, ledger, state and raw bytes are unchanged and preserve the original "
            "canonical ledger-directory binding. Do not activate, generate, copy the campaign into a runnable "
            "directory, replace identities or reset its consumed budget/retries. Original lock bytes are "
            "recorded separately and excluded from the archive.\n\n"
            "The immutable sources.tar.gz is reused by release/asset/SHA reference in source_reference.json; "
            "it is not recompressed or uploaded again. Existing canonical originals are never overwritten.\n",
            encoding="utf-8",
        )
        sidecars = [path for path in sorted(destination.iterdir()) if path.name != "SHA256SUMS"]
        for path in sidecars:
            # The archive's uncompressed entries were already scanned above.
            if path.name != "forensic.tar.gz":
                checkpoint.scan_file(path)
        (destination / "SHA256SUMS").write_text(
            "".join(f"{sha256(path)}  {path.name}\n" for path in sidecars), encoding="utf-8")
        return result
