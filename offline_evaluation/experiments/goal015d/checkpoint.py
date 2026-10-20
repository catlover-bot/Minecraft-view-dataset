"""Goal015D immutable local packages and explicit Draft-only save retries.

Generation and credentials are not accessed by these paths. Frozen GitHub
transport/upload code is reused without modifying its original module.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
from decimal import Decimal
from functools import lru_cache
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
from urllib.parse import quote
import uuid

from ..goal015 import checkpoint as frozen
from ..goal015.runtime import descriptor, path_safe, read_json, require, _save
from ...server_v2.checkpoint import safe_relative, scan_file

Github = frozen.Github
REPOSITORY, BRANCH, MAX_BYTES = frozen.REPOSITORY, frozen.BRANCH, frozen.MAX_BYTES
IDENTITY = ("call_id", "builder", "condition", "repeat_index", "scene_id", "pair_id")
PROTECTED_RELEASE_IDS = frozenset((418290999,))


def _descriptors(value):
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"} <= set(value):
            yield value
        else:
            for child in value.values():
                yield from _descriptors(child)
    elif isinstance(value, list):
        for child in value:
            yield from _descriptors(child)


def _bound(desc):
    actual = descriptor(desc["path"])
    require(all(actual[key] == desc[key] for key in ("bytes", "sha256")),
            "checkpoint dependency bytes/hash differ")
    return path_safe(desc["path"])


def _gt_sources(plan):
    """The frozen GT schema binds files by digest rather than descriptors."""
    for scene, source in plan["gt_sources"].items():
        directory = path_safe(source["building_directory"])
        for name, key in (("voxels.npy", "voxels_sha256"),
                          ("bbox.json", "bbox_sha256")):
            yield scene, directory / "gt" / name, source[key]


def verify_archive(path, expected_sha256, *, destination=None):
    """Validate every member and cross-binding before optional safe restoration."""
    from . import runtime, planning
    require(descriptor(path)["sha256"] == expected_sha256, "archive SHA differs")
    target = path_safe(destination) if destination is not None else None
    if target is not None:
        require(not target.exists(), "new verification destination required")
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        require(len(names) == len(set(names)) and "FILES.json" in names,
                "duplicate archive member or missing manifest")
        total = 0
        for member in members:
            safe_relative(member.name)
            require(member.isfile() and PurePosixPath(member.name).as_posix() == member.name,
                    "archive regular canonical files required")
            total += member.size
            require(total <= MAX_BYTES, "archive expanded limit")
        def raw(name):
            return archive.extractfile(name).read()
        def document(name):
            return json.loads(raw(name))
        manifest = document("FILES.json")
        require(manifest["schema"] == "goal015d_checkpoint_files_v1",
                "Goal015D manifest required")
        require(set(names) == set(manifest["files"]) | {"FILES.json"},
                "archive membership differs")
        required = {"plan.json", "enabled/ledger.json", "enabled/approval.json",
                    "enabled/authority.json"}
        require(required <= set(names), "execution controls missing")
        for member in members:
            data = raw(member.name)
            if member.name != "FILES.json":
                require(manifest["files"][member.name] ==
                        {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()},
                        "archive member hash differs")
        mappings = manifest["original_paths"]
        require(set(mappings) == set(manifest["files"])
                and len(set(mappings.values())) == len(mappings),
                "archive original path mapping differs")
        reverse = {value: name for name, value in mappings.items()}
        def bound(desc):
            require(desc["path"] in reverse, "bound artifact absent from archive")
            name = reverse[desc["path"]]
            require(manifest["files"][name] ==
                    {key: desc[key] for key in ("bytes", "sha256")},
                    "bound artifact hash differs")
            return name
        plan = document("plan.json")
        ledger = document("enabled/ledger.json")
        approval = document("enabled/approval.json")
        authority = document("enabled/authority.json")
        require(plan["plan_id"] == planning.plan_identity(plan) == manifest["plan_id"],
                "archive plan identity differs")
        cap = Decimal(str(plan["proposed_budget_usd"]))
        require(plan["schema"] == "goal015d_plan_v1" and plan["scope_calls"] == 16
                and plan["api_allowed"] is False and plan["budget_approved"] is False
                and plan["retries"] == 0 and plan["concurrency"] == 1
                and (Decimal(0) < cap <= Decimal("5.00") if plan.get("fixture_only") is True
                     else str(plan["proposed_budget_usd"]) == "5.00"),
                "archive diagnostic scope differs")
        require(approval["status"] == "APPROVED"
                and approval["plan_id"] == plan["plan_id"]
                and approval["canonical_root"] == plan["canonical_root"]
                and approval["scope_calls"] == 16
                and approval["approved_budget_usd"] == plan["proposed_budget_usd"]
                and approval["approval_text"] == runtime.approval_text(plan),
                "archive approval differs")
        require(authority["status"] == "APPROVED"
                and authority["plan_id"] == plan["plan_id"]
                and authority["canonical_root"] == plan["canonical_root"]
                and authority["scope_calls"] == 16 and authority["max_retries"] == 0
                and bound(authority["plan"]) == "plan.json"
                and bound(authority["approval"]) == "enabled/approval.json",
                "archive authority binding differs")
        require(authority["metadata_binding_sha256"] == runtime.metadata_binding(plan),
                "archive authority metadata differs")
        runtime._validate_ledger(plan, ledger, approval, verify_files=False)
        require(ledger["plan_id"] == plan["plan_id"]
                and len(ledger["calls"]) == len(plan["schedule"]) == 16,
                "archive ledger scope differs")
        for expected, call in zip(plan["schedule"], ledger["calls"]):
            require(all(call.get(key) == value for key, value in expected.items()),
                    "archive selected schedule changed")
            require(call["state"] in ("NOT_SENT", "SENT_PENDING", "UNKNOWN_USAGE", "SETTLED"),
                    "archive call state differs")
            for desc in _descriptors(expected):
                bound(desc)
            if call["state"] == "NOT_SENT":
                require(not call.get("raw") and not call.get("response_text_path")
                        and call["send_status"] == "NOT_SENT" and not call["response_received"],
                        "NOT_SENT has response evidence")
            if call["state"] == "SETTLED":
                require(call["usage_known"] is True and call["known_cost_usd"] is not None
                        and call.get("raw"), "settled evidence missing")
            else:
                require(call["usage_known"] is False and call["known_cost_usd"] is None,
                        "unresolved usage differs")
            for key in ("raw", "response_text_path"):
                if call.get(key):
                    require(bound(call[key]).startswith("enabled/responses/"),
                            "response escaped canonical subtree")
        for desc in _descriptors(plan):
            bound(desc)
        for desc in _descriptors([ledger, approval, authority]):
            bound(desc)
        for name, sha in plan.get("implementation_hashes", {}).items():
            require(name in reverse and manifest["files"][reverse[name]]["sha256"] == sha,
                    "implementation dependency absent or changed")
        gt_count = 0
        for _, path, sha in _gt_sources(plan):
            require(str(path) in reverse
                    and manifest["files"][reverse[str(path)]]["sha256"] == sha,
                    "bound GT source absent or changed")
            gt_count += 1
        require(gt_count == 4 and len(plan["gt_sources"]) == 2,
                "two buildings with both GT sources required")
        evaluation = manifest.get("evaluation")
        if evaluation is not None:
            data = document(bound(evaluation))
            binding = data["plan_binding"]
            require(binding["plan_id"] == plan["plan_id"]
                    and bound(binding["plan"]) == "plan.json"
                    and bound(binding["ledger"]) == "enabled/ledger.json",
                    "evaluation plan/ledger binding differs")
            require(len(data["rows"]) == 16, "evaluation must retain planned sixteen rows")
            for call, row in zip(ledger["calls"], data["rows"]):
                require(all(row[key] == call[key] for key in IDENTITY)
                        and row["state"] == call["state"], "evaluation row identity differs")
                for desc in _descriptors(row):
                    bound(desc)
        if target is not None:
            target.mkdir(parents=True)
            for member in members:
                out = target / member.name
                out.parent.mkdir(parents=True, exist_ok=True)
                with out.open("xb") as stream:
                    stream.write(raw(member.name))
                scan_file(out)
    return {"status": "VERIFIED", "archive_sha256": expected_sha256,
            "files": len(manifest["files"]), "plan_id": plan["plan_id"],
            "planned_calls": 16, "evaluation_included": evaluation is not None,
            "fixture_only": plan.get("fixture_only") is True,
            "wire_payloads_and_bound_file_dependencies_included": True,
            "gt_source_files_included": gt_count,
            "python_environment_included": False,
            "geometry_or_generation_reexecuted": False}


def create_checkpoint(plan_path, ledger_path=None, *, reason="SAVE_ONLY", target_commit=None):
    """Caller holds runtime locks; the public save-only wrapper acquires them."""
    from . import runtime, evaluate
    plan = runtime.verify_plan(plan_path)
    root = path_safe(plan["canonical_root"])
    ledger_path = path_safe(ledger_path or root / "enabled/ledger.json")
    require(ledger_path == root / "enabled/ledger.json", "canonical ledger required")
    enabled_state = runtime.verify_enabled(plan)
    summary = evaluate._find_existing_evaluation_locked(plan_path, plan)
    entries = {"plan.json": path_safe(plan_path)}
    by_path = {str(path_safe(plan_path)): "plan.json"}
    def add(path):
        path = path_safe(path)
        require(path.is_file(), "regular dependency file required")
        if str(path) in by_path:
            return
        if path.is_relative_to(root):
            member = path.relative_to(root).as_posix()
        else:
            member = "dependencies/" + hashlib.sha256(str(path).encode()).hexdigest()[:20] + "/" + path.name
        safe_relative(member)
        require(member not in entries, "archive dependency path collision")
        entries[member] = path
        by_path[str(path)] = member
    def tree(source):
        source = path_safe(source)
        for path in sorted(source.rglob("*")):
            require(not path.is_symlink(), "archive source symlink forbidden")
            if path.is_dir() or path.name.startswith(".pending-"):
                continue
            add(path)
    tree(root / "enabled")
    for desc in _descriptors(plan):
        add(_bound(desc))
    for desc in _descriptors(enabled_state):
        add(_bound(desc))
    for name, sha in plan.get("implementation_hashes", {}).items():
        require(descriptor(name)["sha256"] == sha, "implementation hash changed")
        add(name)
    for _, path, sha in _gt_sources(plan):
        require(descriptor(path)["sha256"] == sha, "GT source hash changed")
        add(path)
    # Follow official-pricing review/capture descriptors only. Do not recursively
    # expand historical provenance or attempt to parse arbitrary raw HTTP bytes.
    pending = [path_safe(plan["pricing_evidence"]["path"])]
    for control in enabled_state.values():
        for key in ("pricing_evidence", "accepted_pricing_evidence"):
            if isinstance(control, dict) and isinstance(control.get(key), dict):
                pending.append(_bound(control[key]))
    seen = set()
    while pending:
        path = pending.pop()
        if str(path) in seen:
            continue
        seen.add(str(path))
        if path.suffix != ".json":
            continue
        value = read_json(path)
        for desc in _descriptors(value):
            dependency = _bound(desc)
            if str(dependency) not in by_path:
                add(dependency)
                pending.append(dependency)
    if summary is not None:
        tree(Path(summary["evaluation_artifact"]["path"]).parent)
        for desc in _descriptors(read_json(summary["evaluation_artifact"]["path"])):
            add(_bound(desc))
    files = {}
    total = 0
    for name, path in entries.items():
        scan_file(path)
        desc = descriptor(path)
        files[name] = {key: desc[key] for key in ("bytes", "sha256")}
        total += desc["bytes"]
        require(total <= MAX_BYTES, "checkpoint expanded limit")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = "goal015d-" + plan["plan_id"][:12] + "-" + stamp + "-" + uuid.uuid4().hex[:6]
    directory = root / "checkpoints" / name
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "goal015d_checkpoint_files_v1", "plan_id": plan["plan_id"],
                "files": files, "original_paths": {key: str(value) for key, value in entries.items()},
                "evaluation": summary["evaluation_artifact"] if summary else None,
                "external_dependencies": {
                    "python_environment_bundled": False,
                    "environment": plan.get("existing_execution_environment"),
                    "note": "Direct plan/control file dependencies, both buildings' original voxels.npy and bbox.json, and pricing capture chains are included. References inside historical provenance documents are not recursively expanded. A compatible Python and installed dependencies remain external; no credentials are included."}}
    archive_path = directory / "goal015d-state.tar.gz"
    with archive_path.open("xb") as stream, gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as archive:
        for member, path in entries.items():
            raw = path.read_bytes()
            require(files[member] == {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
                    "source changed during checkpoint")
            item = tarfile.TarInfo(member)
            item.size, item.mode = len(raw), 0o600
            archive.addfile(item, io.BytesIO(raw))
        raw = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        item = tarfile.TarInfo("FILES.json")
        item.size, item.mode = len(raw), 0o600
        archive.addfile(item, io.BytesIO(raw))
    arc = descriptor(archive_path)
    require(arc["bytes"] <= MAX_BYTES, "checkpoint compressed limit")
    restored = verify_archive(archive_path, arc["sha256"], destination=directory / "verified-restoration")
    if target_commit is None:
        target_commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                cwd=Path(__file__).resolve().parents[3]).decode().strip()
    require(len(target_commit) == 40 and all(c in "0123456789abcdef" for c in target_commit),
            "full target commit required")
    package = {"schema": "goal015d_checkpoint_v1", "status": "LOCAL_VERIFIED",
               "github_status": "NOT_UPLOADED", "checkpoint_id": name, "directory": str(directory),
               "canonical_root": str(root), "plan_id": plan["plan_id"], "tag": "checkpoint-" + name,
               "target_commitish": target_commit, "reason": reason,
               "assets": [{"name": archive_path.name, "bytes": arc["bytes"], "sha256": arc["sha256"]}],
               "archive_sha256": arc["sha256"], "restoration": restored,
               "fixture_only": plan.get("fixture_only") is True,
               "evaluation": manifest["evaluation"], "research_requests_sent_by_save": 0}
    _save(directory / "checkpoint.json", package, exclusive=True)
    return {"status": "LOCAL_VERIFIED", "path": str(directory),
            "checkpoint": descriptor(directory / "checkpoint.json"),
            "archive_sha256": arc["sha256"], "research_api_requests_sent": 0}


def create_save_only(plan_path, *, reason="SAVE_ONLY_RETRY"):
    from . import runtime
    plan = runtime.verify_plan(plan_path)
    with runtime.locks(plan, create=False):
        return create_checkpoint(plan_path, reason=reason)


@lru_cache(maxsize=1)
def _uploader():
    tree = ast.parse(Path(frozen.__file__).read_text())
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "upload_checkpoint"]
    require(len(functions) == 1, "frozen uploader missing")
    class Namespace(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                node.value = {"goal015_checkpoint_v1": "goal015d_checkpoint_v1",
                              "goal015-": "goal015d-",
                              "goal015-state.tar.gz": "goal015d-state.tar.gz"}.get(node.value, node.value)
                node.value = node.value.replace("Goal015 immutable", "Goal015D immutable")
            return node
    module = ast.fix_missing_locations(Namespace().visit(ast.Module(body=functions, type_ignores=[])))
    namespace = {name: globals()[name] for name in
                 ("path_safe", "read_json", "require", "descriptor", "verify_archive",
                  "_save", "Github", "REPOSITORY", "BRANCH", "quote")}
    exec(compile(module, str(frozen.__file__), "exec"), namespace)
    return namespace["upload_checkpoint"]


def upload_checkpoint(directory, verification_directory, *, release_id=None,
                      allow_create=False, github=None):
    """No generation; failures preserve the immutable package for explicit retry."""
    require(release_id not in PROTECTED_RELEASE_IDS,
            "completed Goal015C Draft is protected; only the new Goal015D checkpoint may be saved")
    directory, target = path_safe(directory), path_safe(verification_directory)
    require(not target.exists(), "new verification directory required")
    package = read_json(directory / "checkpoint.json")
    proof = verify_archive(directory / "goal015d-state.tar.gz", package["archive_sha256"])
    require(package.get("fixture_only") is proof["fixture_only"],
            "package fixture status differs from archived plan")
    require(github is None or proof["fixture_only"], "fake GitHub forbidden for production")
    if not proof["fixture_only"]:
        from . import runtime
        plan = runtime.verify_plan(Path(package["canonical_root"]) / "plan.json")
        require(plan["plan_id"] == proof["plan_id"], "current canonical plan differs")
    try:
        return _uploader()(directory, target, release_id=release_id,
                           allow_create=allow_create, github=github)
    except Exception as exc:
        target.mkdir(parents=True, exist_ok=True)
        _save(target / "SAVE_FAILURE.json",
              {"status": "SAVE_FAILED", "checkpoint": descriptor(directory / "checkpoint.json"),
               "exception_type": type(exc).__name__, "generation_reexecuted": False,
               "research_api_requests_sent": 0,
               "recovery": "Retry this same immutable checkpoint with a new verification directory; never regenerate."},
              exclusive=True)
        raise
