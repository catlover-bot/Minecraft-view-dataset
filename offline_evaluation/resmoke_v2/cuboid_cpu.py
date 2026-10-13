"""Focused CPU-only 200-GT roundtrip and representation-size report.

Run with --output-dir pointing to a new directory. This neither imports an API
transport nor installs/downloads a tokenizer. All reference actions are CPU
controls, never new model outputs.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import time
from unittest.mock import patch

from offline_evaluation.server_v2.core import execute
from offline_evaluation.server_v2.data import load_scene, sha256_file
from offline_evaluation.server_v2.oracle import encode_exact_spec, gt_to_actions

from .cuboids import compact_json, decode_exact, encode_exact, exact_builder_text, reference_actions


def _tokenizer_status(use_local_openai: bool):
    libraries = {}
    for name in ("tiktoken", "tokenizers", "transformers"):
        installed = importlib.util.find_spec(name) is not None
        libraries[name] = {"installed": installed,
                           "version": importlib.metadata.version(name) if installed else None}
    encoding = None
    if use_local_openai:
        import tiktoken
        # A cache miss MUST fail instead of fetching a vocabulary. The initial
        # dependency/cache acquisition is a separate scoped approval operation.
        with patch("tiktoken.load.read_file", side_effect=RuntimeError("tokenizer cache miss; downloads forbidden")):
            encoding = tiktoken.encoding_for_model("gpt-5-mini")
        if encoding.name != "o200k_base":
            raise ValueError("unexpected fixed-model tokenizer mapping")
    return {"libraries": libraries, "model_compatible_local_tokenizer_used": encoding is not None,
            "openai_local_encoding": encoding.name if encoding else None,
            "openai_model": "gpt-5-mini", "tokenizer_vocab_size": encoding.n_vocab if encoding else None,
            "openai_exact_provider_billed_tokens": None, "anthropic_exact_tokens": None,
            "provider_token_count_api_calls": 0,
            "reason": "Local tokenizer measures text only; provider framing/schema overhead and actual generated output are unknown. Tokenizer cache downloads are disabled during this report.",
            "estimate": "UTF-8 bytes / 4 is a labeled size heuristic, not a certified token count."}, encoding


def run(root: Path, output_dir: Path, *, use_local_openai: bool = False, proposed_output_cap: int = 16384) -> dict:
    started = time.perf_counter()
    root = root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / "cuboid_cpu.json", output_dir / "cuboid_cpu.csv"]
    if any(path.exists() for path in targets):
        raise FileExistsError("CPU report already exists; choose a new output directory")
    tokenizer_status, encoding = _tokenizer_status(use_local_openai)
    rows = []
    source_hashes = []
    for dataset in ("buildings_100_v1", "buildings_100_v4"):
        scene_paths = sorted((root / "datasets" / dataset).glob("building_*/gt/voxels.npy"))
        if len(scene_paths) != 100:
            raise ValueError(f"expected 100 existing GT scenes in {dataset}")
        for voxel_path in scene_paths:
            tick = time.perf_counter()
            scene = load_scene(voxel_path.parent.parent)
            grid = scene.grid
            old = encode_exact_spec(grid)
            new = encode_exact(grid)
            reference = reference_actions(new)
            decoded = decode_exact(new)
            reconstructed = execute(reference["operations"], origin=grid.origin)
            roundtrip = decoded.to_dict() == grid.to_dict()
            action_roundtrip = reconstructed.to_dict() == grid.as_world().to_dict()
            if not roundtrip or not action_roundtrip:
                raise AssertionError(f"lossless roundtrip failed: {scene.scene_id}")
            reference_bytes = len(compact_json(reference).encode("utf-8"))
            row = {"scene_id": scene.scene_id, "dataset": dataset,
                   "gt_voxels": len(grid.voxels), "old_z_runs": len(old["runs"]),
                   "new_cuboids": len(new["cuboids"]),
                   "old_input_bytes": len(compact_json(old).encode("utf-8")),
                   "new_spec_bytes": len(compact_json(new).encode("utf-8")),
                   "new_input_bytes": len(exact_builder_text(new).encode("utf-8")),
                   "old_reference_action_bytes": len(compact_json({"operations": gt_to_actions(grid)}).encode("utf-8")),
                   "new_reference_action_bytes": reference_bytes,
                   "new_reference_tokens_bytes_div4_heuristic": reference_bytes / 4,
                   "roundtrip_exact": roundtrip, "reference_executor_exact": action_roundtrip,
                   "cpu_wall_seconds": round(time.perf_counter() - tick, 6)}
            row.update({"old_input_tokens_local_o200k_base": len(encoding.encode(compact_json(old))) if encoding else None,
                        "new_input_tokens_local_o200k_base": len(encoding.encode(exact_builder_text(new))) if encoding else None,
                        "old_reference_tokens_local_o200k_base": len(encoding.encode(compact_json({"operations": gt_to_actions(grid)}))) if encoding else None,
                        "new_reference_tokens_local_o200k_base": len(encoding.encode(compact_json(reference))) if encoding else None})
            row["cpu_wall_seconds"] = round(time.perf_counter() - tick, 6)
            rows.append(row)
            for path in (voxel_path, voxel_path.parent / "bbox.json"):
                source_hashes.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
    fixed = [row for row in rows if row["scene_id"] in
             {"buildings_100_v4_building_060", "buildings_100_v4_building_075"}]
    maximum_reference_tokens = max(row["new_reference_tokens_local_o200k_base"] for row in fixed) if encoding else None
    headroom = proposed_output_cap - maximum_reference_tokens if encoding else None
    report = {
        "schema": "goal004_cuboid_cpu_v1", "research_api_calls": 0,
        "controls_are_model_successes": False,
        "algorithm": "first remaining Y,X,Z cell; extend +Z, then +X, then +Y through remaining same-material cells",
        "minimal_partition_claim": False, "model_output_length_guarantee": False,
        "rows": rows, "source_hashes": source_hashes,
        "source_code_sha256": {name: sha256_file(root / name) for name in (
            "offline_evaluation/resmoke_v2/cuboids.py", "offline_evaluation/resmoke_v2/cuboid_cpu.py",
            "offline_evaluation/server_v2/core.py", "offline_evaluation/server_v2/data.py",
            "offline_evaluation/server_v2/oracle.py")},
        "summary": {"gt_scenes": len(rows), "gt_roundtrip_pass": sum(row["roundtrip_exact"] for row in rows),
                    "reference_executor_pass": sum(row["reference_executor_exact"] for row in rows),
                    "old_z_runs_total": sum(row["old_z_runs"] for row in rows),
                    "new_cuboids_total": sum(row["new_cuboids"] for row in rows),
                    "wall_seconds": round(time.perf_counter() - started, 6)},
        "tokenizer": tokenizer_status, "fixed_smoke": fixed,
        "output_cap_assessment": {
            "proposed_tokens": proposed_output_cap, "providers": ["openai", "anthropic"], "conditions": ["O", "J", "P", "E"],
            "approval": "UNAPPROVED_PROPOSAL",
            "status": ("LOCAL_OPENAI_REFERENCE_FITS_ACCOUNT_NOT_RUN" if headroom >= 4096
                       else "INSUFFICIENT_LOCAL_REFERENCE_HEADROOM_ACCOUNT_NOT_RUN") if encoding
                      else "TOKEN_FEASIBILITY_UNCONFIRMED_ACCOUNT_NOT_RUN",
            "original_output_cap": 16384,
            "original_cap_fits_both_local_reference_bodies": maximum_reference_tokens <= 16384 if encoding else None,
            "maximum_fixed_reference_tokens_local_o200k_base": maximum_reference_tokens,
            "minimum_reference_headroom_under_proposed_cap_tokens": headroom,
            "anthropic_output_tokens": "UNKNOWN; no compatible local tokenizer available",
            "reference": "One feasible fill sequence per partition; not a theoretical minimum or model output guarantee.",
            "caveat": "OpenAI reasoning and formatting also consume output tokens; reference headroom is not an output length guarantee. Actual provider counts and response lengths remain unknown. Anthropic fit is an empirical prospect, not certified by OpenAI tokenization.",
            "basis": "A new common output cap must cover the measured local reference body with headroom; the old16384 cap fails for075. Byte/4 alone materially underestimates these JSON references. No target replacement, partial E, or reasoning-only fix is used.",
            "fixed_reference_bytes_below_16384": all(row["new_reference_action_bytes"] < 16384 for row in fixed),
        },
    }
    with targets[1].open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with targets[0].open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-openai-tokenizer", action="store_true", help="Use installed tiktoken and an already cached fixed-model vocabulary; never download")
    parser.add_argument("--proposed-output-cap", type=int, default=16384)
    args = parser.parse_args()
    report = run(args.root, args.output_dir, use_local_openai=args.local_openai_tokenizer,
                 proposed_output_cap=args.proposed_output_cap)
    print(json.dumps({"summary": report["summary"], "fixed_smoke": report["fixed_smoke"],
                      "output_cap_assessment": report["output_cap_assessment"]}, indent=2))


if __name__ == "__main__":
    main()
