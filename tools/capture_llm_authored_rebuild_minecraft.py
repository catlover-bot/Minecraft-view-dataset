#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture rebuild branches via Minecraft/Malmo for llm_authored cases.")
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--rebuild_subdir", required=True, help="e.g. rebuild_world_direct or rebuild_world_structured")
    p.add_argument("--out_subdir", required=True, help="e.g. direct_rebuild_images_minecraft_capture")
    p.add_argument("--provenance_prefix", required=True, choices=("direct_rebuild", "structured_rebuild"))
    p.add_argument("--building_pattern", default="llm_case_*")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--port", type=int, default=10000)
    p.add_argument("--views", type=int, default=8)
    p.add_argument("--image_size", nargs=2, type=int, default=[960, 540], metavar=("W", "H"))
    p.add_argument("--fov", type=float, default=70.0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _run(cmd: List[str], env: Dict[str, str]) -> None:
    print("[capture_llm_authored_rebuild_minecraft] $ " + " ".join(cmd))
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _ensure_env(root: Path) -> Dict[str, str]:
    env = dict(os.environ)
    malmo_dir = env.get("MALMO_DIR", "").strip()
    if not malmo_dir:
        local = root / "MalmoPlatform"
        if local.is_dir():
            malmo_dir = str(local.resolve())
            env["MALMO_DIR"] = malmo_dir
    if not malmo_dir:
        raise SystemExit("MALMO_DIR is not set and local MalmoPlatform was not found.")

    if not env.get("JAVA_HOME", "").strip():
        try:
            jh = subprocess.check_output(["/usr/libexec/java_home", "-v", "1.8"], text=True).strip()
            if jh:
                env["JAVA_HOME"] = jh
        except Exception:
            pass
    if not env.get("JAVA_HOME", "").strip():
        raise SystemExit("JAVA_HOME (Java 8) is required.")

    xsd = Path(malmo_dir) / "Schemas"
    if xsd.is_dir() and (xsd / "Mission.xsd").is_file():
        env["MALMO_XSD_PATH"] = str(xsd.resolve())
    else:
        raise SystemExit(f"Valid Malmo schema directory not found: {xsd}")

    py_candidates = [
        Path(malmo_dir) / "build" / "install" / "Python_Examples",
        Path(malmo_dir) / "build" / "Malmo" / "src" / "PythonWrapper",
        Path(malmo_dir) / "scripts" / "python-wheel" / "backwards-compatible-imports",
    ]
    py_add = [str(p) for p in py_candidates if p.is_dir()]
    if py_add:
        old = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = ":".join(py_add + ([old] if old else []))
    return env


def _list_cases(root: Path, pattern: str, limit: int) -> List[Path]:
    xs = sorted([p for p in root.glob(pattern) if p.is_dir()])
    if limit > 0:
        xs = xs[:limit]
    return xs


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_case_provenance(
    case_dir: Path,
    prefix: str,
    branch_meta: Dict[str, Any],
    env: Dict[str, str],
    out_subdir: str,
) -> None:
    src_meta_path = case_dir / "meta.json"
    src_meta = _load_json(src_meta_path) if src_meta_path.is_file() else {}
    src_cap = src_meta.get("source_capture", {}) if isinstance(src_meta.get("source_capture"), dict) else {}

    prov_path = case_dir / "provenance.json"
    prov = _load_json(prov_path) if prov_path.is_file() else {}

    prov["source_spec_provider"] = src_cap.get("provenance", {}).get("source_spec_provider")
    prov["source_spec_model"] = src_cap.get("provenance", {}).get("source_spec_model")

    prov["source_build_origin"] = src_cap.get("source_build_origin", "minecraft_instantiated")
    prov["source_image_origin"] = src_cap.get("source_image_origin", "minecraft_capture")
    prov["source_capture_script"] = src_cap.get("source_capture_script", "tools/capture_one_building.py")
    prov["source_capture_timestamp"] = src_cap.get("source_capture_timestamp")

    prov[f"{prefix}_build_origin"] = "minecraft_instantiated"
    prov[f"{prefix}_image_origin"] = "minecraft_capture"
    prov[f"{prefix}_capture_script"] = "tools/capture_rebuild_world.py"
    prov[f"{prefix}_capture_timestamp"] = datetime.now(timezone.utc).isoformat()
    prov[f"{prefix}_images_subdir"] = out_subdir

    prov["active_python_interpreter"] = sys.executable
    prov["active_python_version"] = sys.version.split()[0]
    malmo_dir = Path(env.get("MALMO_DIR", "")).resolve() if env.get("MALMO_DIR") else None
    if malmo_dir:
        mp = malmo_dir / "build" / "install" / "Python_Examples" / "MalmoPython.so"
        prov["malmo_python_path"] = str(mp)
    prov["any_fallback_used"] = bool(src_cap.get("provenance", {}).get("fallback_used", False))

    prov_path.write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")

    # Also persist branch-level provenance in the branch capture meta.
    if isinstance(branch_meta, dict):
        branch_meta[f"{prefix}_build_origin"] = "minecraft_instantiated"
        branch_meta[f"{prefix}_image_origin"] = "minecraft_capture"
        branch_meta[f"{prefix}_capture_script"] = "tools/capture_rebuild_world.py"
        branch_meta[f"{prefix}_capture_timestamp"] = prov[f"{prefix}_capture_timestamp"]
        branch_meta["active_python_interpreter"] = prov["active_python_interpreter"]
        branch_meta["active_python_version"] = prov["active_python_version"]
        branch_meta["malmo_python_path"] = prov.get("malmo_python_path")
        branch_meta["fallback_used"] = False
        out_meta_path = case_dir / out_subdir / "meta.json"
        out_meta_path.write_text(json.dumps(branch_meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset_root not found: {dataset_root}")

    env = _ensure_env(root)

    _run(["bash", str(root / "scripts" / "start_malmo_client_mac.sh"), "--port", str(args.port)], env=env)
    _run(
        [
            "bash",
            str(root / "scripts" / "wait_for_malmo_port.sh"),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--timeout",
            "420",
        ],
        env=env,
    )

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for case_dir in _list_cases(dataset_root, args.building_pattern, args.limit):
        case_id = case_dir.name
        rebuild_world_dir = case_dir / args.rebuild_subdir
        out_dir = case_dir / args.out_subdir
        out_meta = out_dir / "meta.json"
        if out_meta.is_file() and not args.overwrite:
            rows.append({"case_id": case_id, "status": "skipped_exists", "out_dir": str(out_dir)})
            print(f"[capture_llm_authored_rebuild_minecraft] skip {case_id} (exists)")
            continue

        if not rebuild_world_dir.is_dir():
            failures.append({"case_id": case_id, "reason": f"missing rebuild_world_dir: {rebuild_world_dir}"})
            print(f"[capture_llm_authored_rebuild_minecraft] FAIL {case_id}: missing {rebuild_world_dir}")
            continue

        cmd = [
            sys.executable,
            str(root / "tools" / "capture_rebuild_world.py"),
            "--rebuild_world_dir",
            str(rebuild_world_dir),
            "--out",
            str(out_dir),
            "--port",
            str(args.port),
            "--views",
            str(args.views),
            "--image_size",
            str(args.image_size[0]),
            str(args.image_size[1]),
            "--fov",
            str(args.fov),
        ]
        rc = subprocess.run(cmd, env=env).returncode
        if rc != 0:
            failures.append({"case_id": case_id, "reason": f"capture_rebuild_world rc={rc}", "out_dir": str(out_dir)})
            print(f"[capture_llm_authored_rebuild_minecraft] FAIL {case_id} rc={rc}")
            continue

        branch_meta = _load_json(out_meta) if out_meta.is_file() else {}
        _write_case_provenance(case_dir=case_dir, prefix=args.provenance_prefix, branch_meta=branch_meta, env=env, out_subdir=args.out_subdir)
        rows.append({"case_id": case_id, "status": "captured", "out_dir": str(out_dir)})
        print(f"[capture_llm_authored_rebuild_minecraft] captured {case_id}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "rebuild_subdir": args.rebuild_subdir,
        "out_subdir": args.out_subdir,
        "provenance_prefix": args.provenance_prefix,
        "port": int(args.port),
        "views": int(args.views),
        "image_size": [int(args.image_size[0]), int(args.image_size[1])],
        "fov": float(args.fov),
        "rows": rows,
        "failed_cases": failures,
        "completed_cases": [r["case_id"] for r in rows if r.get("status") == "captured"],
    }
    manifest_path = dataset_root / f"minecraft_rebuild_capture_manifest_{args.provenance_prefix}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[capture_llm_authored_rebuild_minecraft] wrote {manifest_path}")

    if failures:
        raise SystemExit(f"rebuild capture failed for {len(failures)} case(s)")


if __name__ == "__main__":
    main()
