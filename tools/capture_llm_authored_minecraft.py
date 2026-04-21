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
    p = argparse.ArgumentParser(description="Capture llm_authored_10 source buildings via Minecraft/Malmo.")
    p.add_argument("--spec_json", default="datasets/llm_authored_10/source_specs/source_specs.json")
    p.add_argument("--out_root", default="datasets/llm_authored_10")
    p.add_argument("--port", type=int, default=10000)
    p.add_argument("--views", type=int, default=10)
    p.add_argument("--image_size", nargs=2, type=int, default=[960, 540], metavar=("W", "H"))
    p.add_argument("--fov", type=float, default=70.0)
    p.add_argument("--seed_base", type=int, default=12000)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--case_prefix", default="llm_case_")
    return p.parse_args()


def _run(cmd: List[str], env: Dict[str, str]) -> None:
    print("[capture_llm_authored_minecraft] $ " + " ".join(cmd))
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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

    # Force a valid schema path for this run. Existing user shell values can be stale.
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


def _is_case_complete(case_dir: Path, expected_views: int, image_subdir: str = "source_images_minecraft_capture") -> bool:
    meta = case_dir / "meta.json"
    gt_bbox = case_dir / "gt" / "bbox.json"
    gt_vox = case_dir / "gt" / "voxels.npy"
    images_dir = case_dir / image_subdir
    if not (meta.is_file() and gt_bbox.is_file() and gt_vox.is_file() and images_dir.is_dir()):
        return False
    try:
        m = _load_json(meta)
    except Exception:
        return False
    views = m.get("views", []) if isinstance(m.get("views"), list) else []
    return len(views) >= expected_views


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    spec_json = Path(args.spec_json).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    env = _ensure_env(root)

    payload = _load_json(spec_json)
    cases = payload.get("cases", []) if isinstance(payload.get("cases"), list) else []
    if not cases:
        raise SystemExit(f"No cases in spec_json: {spec_json}")

    selected: List[Dict[str, Any]] = []
    for c in cases:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("case_id", "")).strip()
        if not cid or not cid.startswith(args.case_prefix):
            continue
        selected.append(c)
    if args.limit > 0:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit("No selected cases to capture.")

    # Start and wait Malmo client.
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
    for i, case in enumerate(selected):
        case_id = str(case.get("case_id", f"llm_case_{i+1:03d}")).strip() or f"llm_case_{i+1:03d}"
        case_dir = out_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        if _is_case_complete(case_dir, expected_views=int(args.views)) and not args.overwrite:
            print(f"[capture_llm_authored_minecraft] skip {case_id} (complete)")
            rows.append({"case_id": case_id, "status": "skipped_complete", "path": str(case_dir)})
            continue

        cmd = [
            sys.executable,
            str(root / "tools" / "capture_one_building.py"),
            "--out",
            str(case_dir),
            "--port",
            str(args.port),
            "--views",
            str(args.views),
            "--image_size",
            str(args.image_size[0]),
            str(args.image_size[1]),
            "--fov",
            str(args.fov),
            "--seed",
            str(int(args.seed_base) + i),
            "--style_id",
            str(i),
            "--images_subdir",
            "source_images_minecraft_capture",
            "--source_spec_json",
            str(spec_json),
            "--source_case_id",
            case_id,
            "--source_case_index",
            str(i),
        ]
        rc = subprocess.run(cmd, env=env).returncode
        if rc != 0:
            failures.append({"case_id": case_id, "returncode": rc, "path": str(case_dir)})
            print(f"[capture_llm_authored_minecraft] FAIL {case_id} rc={rc}")
            continue

        rows.append({"case_id": case_id, "status": "captured", "path": str(case_dir)})
        print(f"[capture_llm_authored_minecraft] captured {case_id}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "study": "llm_authored_10_minecraft_capture",
        "source_spec_json": str(spec_json),
        "out_root": str(out_root),
        "port": int(args.port),
        "views": int(args.views),
        "image_size": [int(args.image_size[0]), int(args.image_size[1])],
        "fov": float(args.fov),
        "selected_cases": [str(c.get("case_id", "")) for c in selected],
        "completed_cases": [r["case_id"] for r in rows if r.get("status") == "captured"],
        "failed_cases": failures,
        "rows": rows,
    }
    (out_root / "minecraft_capture_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[capture_llm_authored_minecraft] wrote {out_root / 'minecraft_capture_manifest.json'}")
    if failures:
        raise SystemExit(f"capture failed for {len(failures)} case(s)")


if __name__ == "__main__":
    main()
