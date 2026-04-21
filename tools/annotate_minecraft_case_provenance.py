#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate llm_authored Minecraft case meta provenance fields.")
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--building_pattern", default="llm_case_*")
    p.add_argument("--python_executable", default=sys.executable)
    p.add_argument("--python_version", default=sys.version.split()[0])
    p.add_argument("--malmo_python_path", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    if not root.is_dir():
        raise SystemExit(f"dataset_root not found: {root}")

    count = 0
    for bdir in sorted(p for p in root.glob(args.building_pattern) if p.is_dir()):
        meta_path = bdir / "meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sc = meta.get("source_capture")
        if not isinstance(sc, dict):
            sc = {}
            meta["source_capture"] = sc

        # Keep existing values when already set by capture.
        sc.setdefault("source_build_origin", "minecraft_instantiated")
        sc.setdefault("source_image_origin", "minecraft_capture")
        sc.setdefault("source_capture_script", "tools/capture_one_building.py")
        sc["active_python_interpreter"] = str(args.python_executable)
        sc["active_python_version"] = str(args.python_version)
        if args.malmo_python_path:
            sc["malmo_python_path"] = str(args.malmo_python_path)
        sc.setdefault("fallback_used", False)

        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        count += 1

    print(f"[annotate_minecraft_case_provenance] updated meta.json for {count} case(s) under {root}")


if __name__ == "__main__":
    main()
