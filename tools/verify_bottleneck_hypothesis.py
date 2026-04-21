#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MainCase:
    model_key: str
    dataset_key: str
    dataset_name: str
    metrics_path: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Verify bottleneck hypotheses for current I2T2B results using existing outputs only "
            "(no new LLM runs)."
        )
    )
    p.add_argument(
        "--main_json",
        default="reports/final/main_shared_hparams_results_2026-03-10.json",
        help="Main fair-comparison summary JSON path.",
    )
    p.add_argument(
        "--outputs_root",
        default="outputs/i2t2b",
        help="Root of experiment outputs.",
    )
    p.add_argument(
        "--datasets_root",
        default="datasets",
        help="Root of GT datasets.",
    )
    p.add_argument(
        "--op_budget",
        type=int,
        default=260,
        help="Operation budget used in representation-ceiling diagnostic.",
    )
    p.add_argument(
        "--extra_budgets",
        default="64,128,512",
        help="Extra operation budgets for diagnostics (comma separated).",
    )
    p.add_argument(
        "--building_pattern",
        default="building_*",
        help="Building glob pattern under each dataset root.",
    )
    p.add_argument(
        "--out_json",
        default="reports/final/bottleneck_verification_2026-04-08.json",
        help="Output JSON report path.",
    )
    p.add_argument(
        "--out_md",
        default="reports/final/bottleneck_verification_2026-04-08.md",
        help="Output Markdown summary path.",
    )
    return p.parse_args()


def _safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else float(a) / float(b)


def _mean(xs: Iterable[float]) -> float:
    seq = list(xs)
    if not seq:
        return 0.0
    return float(sum(seq)) / float(len(seq))


def _stdev(xs: Iterable[float]) -> float:
    seq = list(xs)
    if len(seq) <= 1:
        return 0.0
    mu = _mean(seq)
    return float(math.sqrt(sum((x - mu) ** 2 for x in seq) / float(len(seq) - 1)))


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx = _mean(x)
    my = _mean(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(len(x)))
    return float(cov / math.sqrt(vx * vy))


def _rankdata(vals: Sequence[float]) -> List[float]:
    if not vals:
        return []
    indexed = sorted([(float(v), i) for i, v in enumerate(vals)], key=lambda t: t[0])
    ranks = [0.0] * len(vals)
    i = 0
    n = len(indexed)
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][0] == indexed[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[indexed[k][1]] = avg_rank
        i = j + 1
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _pearson(rx, ry)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_main_cases(main_json: Path) -> List[MainCase]:
    obj = _load_json(main_json)
    out: List[MainCase] = []
    models = obj.get("models", {})
    if not isinstance(models, dict):
        return out
    for model_key, per_model in models.items():
        if not isinstance(per_model, dict):
            continue
        for dataset_key in ("v1", "v4"):
            node = per_model.get(dataset_key, {})
            if not isinstance(node, dict):
                continue
            mp = str(node.get("metrics_path", "")).strip()
            if not mp:
                continue
            dataset_name = f"buildings_100_{dataset_key}"
            out.append(
                MainCase(
                    model_key=str(model_key),
                    dataset_key=dataset_key,
                    dataset_name=dataset_name,
                    metrics_path=Path(mp),
                )
            )
    return out


def _model_tag_for_desc(model_key: str) -> str:
    m = model_key.strip().lower()
    if m == "openai":
        return "openai_gpt_5_mini"
    if m == "claude":
        return "claude_haiku_4_5"
    return m


def _parse_extra_budgets(text: str) -> List[int]:
    out: List[int] = []
    for tok in str(text).split(","):
        s = tok.strip()
        if not s:
            continue
        try:
            v = int(s)
        except Exception:
            continue
        if v > 0:
            out.append(v)
    return sorted(set(out))


def _normalize_token(raw: Any) -> str:
    token = str(raw).strip().lower()
    token = token.split("[", 1)[0]
    token = token.split("{", 1)[0]
    if ":" in token:
        token = token.split(":", 1)[1]
    token = token.replace(" ", "_").replace("-", "_")
    token = re.sub(r"_+", "_", token).strip("_")
    token = re.sub(r"(stone_slab)\d+$", r"\1", token)
    token = re.sub(r"(double_stone_slab)\d+$", r"\1", token)
    if token in {"air", "cave_air", "void_air", ""}:
        return "air"
    # Lightweight normalization sufficient for this diagnostic.
    mapping = {
        "stone_bricks": "stonebrick",
        "stone_brick": "stonebrick",
        "minecraft_stone_bricks": "stonebrick",
        "minecraft_stone_brick": "stonebrick",
        "bricks": "brick",
        "brick_block": "brick",
        "oak_planks": "wood",
        "spruce_planks": "wood",
        "birch_planks": "wood",
        "jungle_planks": "wood",
        "acacia_planks": "wood",
        "dark_oak_planks": "wood",
        "planks": "wood",
        "wooden_planks": "wood",
        "stone_slab": "slab_stone",
        "stone_slab2": "slab_stone",
        "double_stone_slab": "slab_stone",
        "double_stone_slab2": "slab_stone",
        "wooden_slab": "slab_wood",
        "double_wooden_slab": "slab_wood",
        "glass_pane": "glass",
        "stained_glass": "glass",
        "stained_glass_pane": "glass",
    }
    if token in mapping:
        return mapping[token]
    if token.endswith("_fence"):
        return "fence"
    return token


def _load_gt_voxels(path: Path) -> np.ndarray:
    vox = np.load(path, allow_pickle=True)
    if vox.ndim != 3:
        raise ValueError(f"Expected 3D voxels: {path} shape={vox.shape}")
    out = np.empty(vox.shape, dtype="<U32")
    it = np.nditer(vox, flags=["multi_index", "refs_ok"])
    while not it.finished:
        out[it.multi_index] = _normalize_token(it[0].item())
        it.iternext()
    return out


def _make_line_cuboids_x(vox: np.ndarray) -> List[Tuple[int, int, int, int, int, int, str]]:
    sy, sx, sz = vox.shape
    cuboids: List[Tuple[int, int, int, int, int, int, str]] = []
    for y in range(sy):
        for z in range(sz):
            x = 0
            while x < sx:
                b = vox[y, x, z]
                if b == "air":
                    x += 1
                    continue
                x1 = x
                while x + 1 < sx and vox[y, x + 1, z] == b:
                    x += 1
                x2 = x
                cuboids.append((x1, x2, y, y, z, z, str(b)))
                x += 1
    return cuboids


def _merge_along_z(
    cuboids: List[Tuple[int, int, int, int, int, int, str]]
) -> List[Tuple[int, int, int, int, int, int, str]]:
    if not cuboids:
        return []
    # key: y, x-range, block. merge consecutive z.
    xs = sorted(cuboids, key=lambda c: (c[2], c[0], c[1], c[6], c[4], c[5]))
    out: List[Tuple[int, int, int, int, int, int, str]] = []
    cur = list(xs[0])
    for c in xs[1:]:
        if (
            c[2] == cur[2]
            and c[3] == cur[3]
            and c[0] == cur[0]
            and c[1] == cur[1]
            and c[6] == cur[6]
            and c[4] == cur[5] + 1
        ):
            cur[5] = c[5]
            continue
        out.append((cur[0], cur[1], cur[2], cur[3], cur[4], cur[5], cur[6]))
        cur = list(c)
    out.append((cur[0], cur[1], cur[2], cur[3], cur[4], cur[5], cur[6]))
    return out


def _merge_along_y(
    cuboids: List[Tuple[int, int, int, int, int, int, str]]
) -> List[Tuple[int, int, int, int, int, int, str]]:
    if not cuboids:
        return []
    # key: x-range,z-range,block. merge consecutive y.
    xs = sorted(cuboids, key=lambda c: (c[0], c[1], c[4], c[5], c[6], c[2], c[3]))
    out: List[Tuple[int, int, int, int, int, int, str]] = []
    cur = list(xs[0])
    for c in xs[1:]:
        if (
            c[0] == cur[0]
            and c[1] == cur[1]
            and c[4] == cur[4]
            and c[5] == cur[5]
            and c[6] == cur[6]
            and c[2] == cur[3] + 1
        ):
            cur[3] = c[3]
            continue
        out.append((cur[0], cur[1], cur[2], cur[3], cur[4], cur[5], cur[6]))
        cur = list(c)
    out.append((cur[0], cur[1], cur[2], cur[3], cur[4], cur[5], cur[6]))
    return out


def _cuboid_volume(c: Tuple[int, int, int, int, int, int, str]) -> int:
    return (c[1] - c[0] + 1) * (c[3] - c[2] + 1) * (c[5] - c[4] + 1)


def _representation_diagnostic_for_vox(
    vox: np.ndarray,
    budgets: Sequence[int],
) -> Dict[str, Any]:
    runs = _make_line_cuboids_x(vox)
    merged = _merge_along_y(_merge_along_z(runs))
    vols = [_cuboid_volume(c) for c in merged]
    non_air = int(np.sum(vox != "air"))
    if non_air <= 0:
        return {
            "non_air": 0,
            "exact_ops_count": 0,
            "exact_recall": 1.0,
            "budget_metrics": {int(b): {"recall": 1.0, "iou": 1.0, "f1": 1.0, "precision": 1.0} for b in budgets},
        }
    sorted_vols = sorted(vols, reverse=True)
    budget_metrics: Dict[int, Dict[str, float]] = {}
    for b in budgets:
        k = max(0, int(b))
        covered = int(sum(sorted_vols[:k]))
        recall = _safe_div(float(covered), float(non_air))
        precision = 1.0 if covered > 0 else 0.0
        iou = recall  # subset of GT only => union==GT non-air
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        budget_metrics[int(b)] = {
            "recall": recall,
            "precision": precision,
            "iou": iou,
            "f1": f1,
            "covered_voxels": float(covered),
            "non_air_voxels": float(non_air),
        }
    return {
        "non_air": int(non_air),
        "exact_ops_count": int(len(merged)),
        "exact_recall": 1.0,
        "budget_metrics": budget_metrics,
    }


def _extract_plan_features(plan_req: Dict[str, Any], plan_obj: Dict[str, Any]) -> Dict[str, float]:
    coerce = plan_req.get("coerce_report", {}) if isinstance(plan_req.get("coerce_report"), dict) else {}
    val = plan_req.get("validation_report", {}) if isinstance(plan_req.get("validation_report"), dict) else {}
    ops = plan_obj.get("operations", []) if isinstance(plan_obj.get("operations"), list) else []
    strict_blocking = val.get("strict_blocking_issues", [])
    budget_viol = val.get("budget_violations", [])
    schema_viol = val.get("schema_violations", [])

    def _len(v: Any) -> int:
        return len(v) if isinstance(v, list) else 0

    return {
        "fallback_triggered": 1.0 if bool(plan_req.get("fallback_triggered")) else 0.0,
        "llm_failed": 1.0 if bool(plan_req.get("llm_failed")) else 0.0,
        "coerce_repaired_count": float(coerce.get("repaired_operations_count", 0) or 0),
        "coerce_expanded_count": float(coerce.get("expanded_operations_count", 0) or 0),
        "coerce_dropped_count": float(coerce.get("dropped_operations_count", 0) or 0),
        "strict_blocking_count": float(_len(strict_blocking)),
        "has_strict_blocking": 1.0 if _len(strict_blocking) > 0 else 0.0,
        "budget_violation_count": float(_len(budget_viol)),
        "schema_violation_count": float(_len(schema_viol)),
        "operations_trimmed": 1.0 if bool(val.get("operations_trimmed")) else 0.0,
        "role_fixed_block_count": float(val.get("role_fixed_block_count", 0) or 0),
        "operations_assigned_role_count": float(val.get("operations_assigned_role_count", 0) or 0),
        "operations_unknown_role_count": float(val.get("operations_unknown_role_count", 0) or 0),
        "bbox_outside_operation_count": float(val.get("bbox_outside_operation_count", 0) or 0),
        "plan_operation_count": float(len(ops)),
        "empty_operations": 1.0 if len(ops) == 0 else 0.0,
        "valid_strict": 1.0 if bool(val.get("valid_strict")) else 0.0,
    }


def _corr_table(rows: List[Dict[str, float]], x_keys: Sequence[str], y_keys: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for xk in x_keys:
        xv = [float(r.get(xk, 0.0)) for r in rows]
        per_y: Dict[str, Any] = {}
        for yk in y_keys:
            yv = [float(r.get(yk, 0.0)) for r in rows]
            per_y[yk] = {
                "pearson": _pearson(xv, yv),
                "spearman": _spearman(xv, yv),
            }
        out[xk] = per_y
    return out


def _corr_table_by_case(
    rows: List[Dict[str, Any]],
    x_keys: Sequence[str],
    y_keys: Sequence[str],
    *,
    case_key: str = "case",
) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for row in rows:
        ck = str(row.get(case_key, "")).strip()
        if not ck:
            continue
        grouped.setdefault(ck, []).append(row)

    per_case: Dict[str, Any] = {}
    for ck, grows in grouped.items():
        per_case[ck] = _corr_table(grows, x_keys, y_keys)

    weighted: Dict[str, Any] = {}
    for xk in x_keys:
        weighted[xk] = {}
        for yk in y_keys:
            num_p = 0.0
            den_p = 0.0
            num_s = 0.0
            den_s = 0.0
            for ck, grows in grouped.items():
                n = len(grows)
                if n < 3:
                    continue
                corr = per_case.get(ck, {}).get(xk, {}).get(yk, {})
                p = float(corr.get("pearson", 0.0))
                s = float(corr.get("spearman", 0.0))
                num_p += n * p
                den_p += n
                num_s += n * s
                den_s += n
            weighted[xk][yk] = {
                "pearson_weighted_by_case_n": _safe_div(num_p, den_p),
                "spearman_weighted_by_case_n": _safe_div(num_s, den_s),
            }

    return {
        "case_counts": {k: len(v) for k, v in grouped.items()},
        "per_case": per_case,
        "weighted_by_case_n": weighted,
    }


def _binary_effects(rows: List[Dict[str, float]], bin_keys: Sequence[str], y_keys: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for bk in bin_keys:
        per_y: Dict[str, Any] = {}
        t_rows = [r for r in rows if float(r.get(bk, 0.0)) >= 0.5]
        f_rows = [r for r in rows if float(r.get(bk, 0.0)) < 0.5]
        for yk in y_keys:
            mt = _mean(float(r.get(yk, 0.0)) for r in t_rows) if t_rows else 0.0
            mf = _mean(float(r.get(yk, 0.0)) for r in f_rows) if f_rows else 0.0
            per_y[yk] = {
                "true_mean": mt,
                "false_mean": mf,
                "delta_true_minus_false": mt - mf,
                "true_count": len(t_rows),
                "false_count": len(f_rows),
            }
        out[bk] = per_y
    return out


def _dimension_score_from_bbox(pred_bbox: Dict[str, Any], gt_bbox: Dict[str, Any]) -> float:
    try:
        w_gt = int(gt_bbox["xmax"]) - int(gt_bbox["xmin"]) + 1
        d_gt = int(gt_bbox["zmax"]) - int(gt_bbox["zmin"]) + 1
        h_gt = int(gt_bbox["ymax"]) - int(gt_bbox["ymin"]) + 1
        w_pr = int(pred_bbox["xmax"]) - int(pred_bbox["xmin"]) + 1
        d_pr = int(pred_bbox["zmax"]) - int(pred_bbox["zmin"]) + 1
        h_pr = int(pred_bbox["ymax"]) - int(pred_bbox["ymin"]) + 1
        if min(w_gt, d_gt, h_gt, w_pr, d_pr, h_pr) <= 0:
            return 0.0
    except Exception:
        return 0.0
    rel = (abs(w_pr - w_gt) / float(w_gt) + abs(d_pr - d_gt) / float(d_gt) + abs(h_pr - h_gt) / float(h_gt)) / 3.0
    return max(0.0, 1.0 - min(1.0, rel))


def _extract_plan_bbox(plan_obj: Dict[str, Any]) -> Dict[str, int]:
    box = plan_obj.get("bbox", {}) if isinstance(plan_obj.get("bbox"), dict) else {}
    if not box and all(k in plan_obj for k in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")):
        box = plan_obj
    return {
        "xmin": int(box.get("xmin", 0) or 0),
        "xmax": int(box.get("xmax", 0) or 0),
        "ymin": int(box.get("ymin", 0) or 0),
        "ymax": int(box.get("ymax", 0) or 0),
        "zmin": int(box.get("zmin", 0) or 0),
        "zmax": int(box.get("zmax", 0) or 0),
    }


def _load_desc_item_map(outputs_root: Path, dataset_name: str, model_key: str) -> Dict[str, Dict[str, float]]:
    desc_file = outputs_root / dataset_name / "metrics" / "description" / f"{_model_tag_for_desc(model_key)}.json"
    if not desc_file.is_file():
        return {}
    dobj = _load_json(desc_file)
    ditems = dobj.get("items", [])
    if not isinstance(ditems, list):
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for dit in ditems:
        if not isinstance(dit, dict):
            continue
        bname = str(dit.get("building", "")).strip()
        if not bname:
            continue
        sm = dit.get("strict_material_metrics", {})
        cm = dit.get("coarse_material_metrics", {})
        dm = dit.get("dimension_metrics", {})
        if not isinstance(sm, dict) or not isinstance(cm, dict) or not isinstance(dm, dict):
            continue
        out[bname] = {
            "auto_score": float(dit.get("auto_score", 0.0) or 0.0),
            "strict_material_f1": float(sm.get("f1", 0.0) or 0.0),
            "coarse_material_f1": float(cm.get("f1", 0.0) or 0.0),
            "dimension_score": float(dm.get("dim_score", 0.0) or 0.0),
        }
    return out


def _markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Bottleneck Verification (No New LLM Runs)")
    lines.append("")
    lines.append(f"- created_at: `{payload['created_at']}`")
    lines.append(f"- op_budget: `{payload['settings']['op_budget']}`")
    lines.append("")

    a = payload["A_stage_isolation"]
    lines.append("## A. Stage-Isolation Upper-Bound (Oracle-like GT Cuboid Injection)")
    lines.append("")
    lines.append("- `oracle_copy` (GT voxel copy): theoretical IoU/F1=1.0 (sanity upper bound).")
    lines.append("- `oracle_budgeted_cuboids`: GT-derived cuboid decomposition under operation budget.")
    lines.append("")
    lines.append("| dataset | exact_ops_mean | budget_iou_mean | budget_f1_mean | budget_recall_mean |")
    lines.append("|---|---:|---:|---:|---:|")
    for ds, row in a["per_dataset"].items():
        bm = row["budget_metrics"][str(payload["settings"]["op_budget"])]
        lines.append(
            f"| {ds} | {row['exact_ops_mean']:.1f} | {bm['iou_mean']:.4f} | {bm['f1_mean']:.4f} | {bm['recall_mean']:.4f} |"
        )
    lines.append("")
    lines.append(
        f"- all-200 budgeted upper-bound (IoU): `{a['all_200_budget_iou_mean']:.4f}` vs Main OpenAI `{a['main_openai_iou_all_200']:.4f}` / Main Claude `{a['main_claude_iou_all_200']:.4f}`"
    )
    lines.append(
        f"- description->plan dimension retention: mean desc `{a['description_to_plan_dimension_preservation']['desc_dim_score_mean']:.4f}` "
        f"-> plan `{a['description_to_plan_dimension_preservation']['plan_dim_score_mean']:.4f}` "
        f"(delta `{a['description_to_plan_dimension_preservation']['plan_minus_desc_mean']:+.4f}`)"
    )
    lines.append("")

    b = payload["B_plan_fidelity_audit"]
    lines.append("## B. Plan Fidelity Audit")
    lines.append("")
    lines.append(f"- analyzed_rows: `{b['analyzed_rows']}`")
    lines.append("- strongest |spearman| predictors for IoU:")
    for row in b["top_predictors_iou"][:8]:
        lines.append(f"  - `{row['feature']}`: spearman={row['spearman']:+.4f}, pearson={row['pearson']:+.4f}")
    lines.append("- case-weighted IoU predictors (controls v1/v4 + model mix):")
    for row in b["top_predictors_iou_case_weighted"][:6]:
        lines.append(
            f"  - `{row['feature']}`: spearman_w={row['spearman_weighted_by_case_n']:+.4f}, "
            f"pearson_w={row['pearson_weighted_by_case_n']:+.4f}"
        )
    lines.append("")

    c = payload["C_representation_ceiling"]
    lines.append("## C. Representation Ceiling Diagnostic")
    lines.append("")
    lines.append(f"- op_budget={payload['settings']['op_budget']}, all-200 budgeted IoU mean: `{c['all_200_budget_iou_mean']:.4f}`")
    lines.append(f"- exact_ops_count mean: `{c['all_200_exact_ops_mean']:.1f}` (perfect under this decomposition).")
    lines.append("")

    d = payload["D_description_metric_validity"]
    lines.append("## D. Description Metric Validity")
    lines.append("")
    lines.append(f"- analyzed_rows: `{d['analyzed_rows']}`")
    lines.append("- Spearman(description -> rebuild) pooled over Main cases:")
    for k, row in d["pooled_spearman_summary"].items():
        lines.append(
            f"  - `{k}`: IoU={row['iou']:+.4f}, F1={row['f1']:+.4f}, material={row['material_match']:+.4f}, coarse={row['coarse_material_match']:+.4f}, correct_placement={row['correct_placement_rate']:+.4f}"
        )
    lines.append("- Spearman(description -> rebuild) weighted within-case:")
    for k, row in d["within_case_weighted_spearman_summary"].items():
        lines.append(
            f"  - `{k}`: IoU={row['iou']:+.4f}, F1={row['f1']:+.4f}, material={row['material_match']:+.4f}, coarse={row['coarse_material_match']:+.4f}, correct_placement={row['correct_placement_rate']:+.4f}"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    main_json = (ROOT / args.main_json).resolve() if not Path(args.main_json).is_absolute() else Path(args.main_json)
    outputs_root = (ROOT / args.outputs_root).resolve() if not Path(args.outputs_root).is_absolute() else Path(args.outputs_root)
    datasets_root = (ROOT / args.datasets_root).resolve() if not Path(args.datasets_root).is_absolute() else Path(args.datasets_root)
    out_json = (ROOT / args.out_json).resolve() if not Path(args.out_json).is_absolute() else Path(args.out_json)
    out_md = (ROOT / args.out_md).resolve() if not Path(args.out_md).is_absolute() else Path(args.out_md)

    cases = _parse_main_cases(main_json)
    if not cases:
        raise SystemExit(f"No main cases found: {main_json}")

    budgets = sorted(set([int(args.op_budget)] + _parse_extra_budgets(args.extra_budgets)))

    # ---------- A/C: GT-derived oracle cuboid diagnostics ----------
    a_per_dataset: Dict[str, Any] = {}
    c_per_dataset: Dict[str, Any] = {}
    all_budget_iou: List[float] = []
    all_exact_ops: List[float] = []
    for ds in sorted(set(c.dataset_name for c in cases)):
        ds_root = datasets_root / ds
        bdirs = sorted([p for p in ds_root.glob(args.building_pattern) if p.is_dir()])
        exact_ops: List[float] = []
        per_budget_vals: Dict[int, Dict[str, List[float]]] = {
            b: {"iou": [], "f1": [], "recall": []} for b in budgets
        }
        for bdir in bdirs:
            vox_path = bdir / "gt" / "voxels.npy"
            if not vox_path.is_file():
                continue
            vox = _load_gt_voxels(vox_path)
            diag = _representation_diagnostic_for_vox(vox=vox, budgets=budgets)
            exact_ops.append(float(diag["exact_ops_count"]))
            all_exact_ops.append(float(diag["exact_ops_count"]))
            for b in budgets:
                bm = diag["budget_metrics"][int(b)]
                per_budget_vals[b]["iou"].append(float(bm["iou"]))
                per_budget_vals[b]["f1"].append(float(bm["f1"]))
                per_budget_vals[b]["recall"].append(float(bm["recall"]))
                if int(b) == int(args.op_budget):
                    all_budget_iou.append(float(bm["iou"]))

        budget_agg = {
            str(b): {
                "iou_mean": _mean(v["iou"]),
                "iou_std": _stdev(v["iou"]),
                "f1_mean": _mean(v["f1"]),
                "recall_mean": _mean(v["recall"]),
            }
            for b, v in per_budget_vals.items()
        }
        row = {
            "buildings": len(exact_ops),
            "exact_ops_mean": _mean(exact_ops),
            "exact_ops_std": _stdev(exact_ops),
            "budget_metrics": budget_agg,
        }
        a_per_dataset[ds] = row
        c_per_dataset[ds] = row

    # Main fair reference numbers.
    main_obj = _load_json(main_json)
    main_openai_iou = float(main_obj["models"]["openai"]["all_200"]["iou"])
    main_claude_iou = float(main_obj["models"]["claude"]["all_200"]["iou"])

    # ---------- B: plan-fidelity audit ----------
    y_keys = ["iou", "f1", "material_match", "coarse_material_match", "correct_placement_rate"]
    plan_rows: List[Dict[str, float]] = []
    stage_rows: List[Dict[str, float]] = []
    op_kind_counts: Dict[str, int] = {}
    strict_issue_freq: Dict[str, int] = {}
    gt_bbox_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    desc_maps: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}
    per_case_counts: Dict[str, int] = {}
    for c in cases:
        metrics_obj = _load_json(c.metrics_path)
        pred_subdir = str(metrics_obj.get("settings", {}).get("pred_subdir", "")).strip()
        if not pred_subdir:
            continue
        plan_subdir = pred_subdir.replace("rebuild_world", "rebuild_plan", 1)
        items = metrics_obj.get("items", [])
        if not isinstance(items, list):
            continue
        count = 0
        ds_out_root = outputs_root / c.dataset_name
        desc_map = desc_maps.setdefault((c.dataset_name, c.model_key), _load_desc_item_map(outputs_root, c.dataset_name, c.model_key))
        for it in items:
            if not isinstance(it, dict):
                continue
            bname = str(it.get("building", "")).strip()
            if not bname:
                continue
            metrics = it.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            req_path = ds_out_root / bname / plan_subdir / "plan.request.json"
            plan_path = ds_out_root / bname / plan_subdir / "plan.json"
            if not req_path.is_file() or not plan_path.is_file():
                continue
            req = _load_json(req_path)
            plan = _load_json(plan_path)
            vr_raw = req.get("validation_report", {}) if isinstance(req.get("validation_report"), dict) else {}
            strict_raw = vr_raw.get("strict_blocking_issues", [])
            if isinstance(strict_raw, list):
                for issue in strict_raw:
                    key = str(issue).strip() or "unknown"
                    strict_issue_freq[key] = int(strict_issue_freq.get(key, 0)) + 1
            for op in plan.get("operations", []) if isinstance(plan.get("operations"), list) else []:
                if not isinstance(op, dict):
                    continue
                kind = str(op.get("op", "")).strip().lower()
                if not kind:
                    continue
                op_kind_counts[kind] = int(op_kind_counts.get(kind, 0)) + 1
            features = _extract_plan_features(req, plan)
            row = {
                **features,
                "case": f"{c.dataset_key}/{c.model_key}",
                "iou": float(metrics.get("iou", 0.0) or 0.0),
                "f1": float(metrics.get("f1", 0.0) or 0.0),
                "material_match": float(metrics.get("material_match", 0.0) or 0.0),
                "coarse_material_match": float(metrics.get("coarse_material_match", 0.0) or 0.0),
                "correct_placement_rate": float(metrics.get("correct_placement_rate", 0.0) or 0.0),
            }
            plan_rows.append(row)

            key = (c.dataset_name, bname)
            gt_bbox = gt_bbox_cache.get(key)
            if gt_bbox is None:
                gt_bbox_path = datasets_root / c.dataset_name / bname / "gt" / "bbox.json"
                if gt_bbox_path.is_file():
                    gt_bbox = _load_json(gt_bbox_path)
                else:
                    gt_bbox = {}
                gt_bbox_cache[key] = gt_bbox
            if gt_bbox:
                plan_bbox = _extract_plan_bbox(plan)
                plan_dim_score = _dimension_score_from_bbox(plan_bbox, gt_bbox)
                drow = desc_map.get(bname, {})
                desc_dim = float(drow.get("dimension_score", 0.0) or 0.0)
                stage_rows.append(
                    {
                        "case": f"{c.dataset_key}/{c.model_key}",
                        "desc_dimension_score": desc_dim,
                        "plan_dimension_score": plan_dim_score,
                        "plan_minus_desc_dimension_score": plan_dim_score - desc_dim,
                        "iou": row["iou"],
                        "f1": row["f1"],
                        "material_match": row["material_match"],
                        "coarse_material_match": row["coarse_material_match"],
                        "correct_placement_rate": row["correct_placement_rate"],
                    }
                )
            count += 1
        per_case_counts[f"{c.dataset_key}/{c.model_key}"] = count

    feature_keys = [
        "fallback_triggered",
        "llm_failed",
        "coerce_repaired_count",
        "coerce_expanded_count",
        "coerce_dropped_count",
        "strict_blocking_count",
        "has_strict_blocking",
        "budget_violation_count",
        "schema_violation_count",
        "operations_trimmed",
        "role_fixed_block_count",
        "operations_assigned_role_count",
        "operations_unknown_role_count",
        "bbox_outside_operation_count",
        "plan_operation_count",
        "empty_operations",
        "valid_strict",
    ]
    plan_corr = _corr_table(plan_rows, feature_keys, y_keys)
    plan_corr_by_case = _corr_table_by_case(plan_rows, feature_keys, y_keys, case_key="case")
    plan_bin = _binary_effects(
        plan_rows,
        ["fallback_triggered", "has_strict_blocking", "operations_trimmed", "empty_operations", "llm_failed"],
        y_keys,
    )
    binary_prevalence = {
        key: _mean(float(r.get(key, 0.0)) for r in plan_rows)
        for key in ["fallback_triggered", "llm_failed", "has_strict_blocking", "operations_trimmed", "empty_operations"]
    }
    top_iou = []
    for f in feature_keys:
        sc = float(plan_corr.get(f, {}).get("iou", {}).get("spearman", 0.0))
        pc = float(plan_corr.get(f, {}).get("iou", {}).get("pearson", 0.0))
        top_iou.append({"feature": f, "spearman": sc, "pearson": pc, "abs_spearman": abs(sc)})
    top_iou.sort(key=lambda r: r["abs_spearman"], reverse=True)
    top_iou_case_weighted = []
    weighted_tbl = plan_corr_by_case["weighted_by_case_n"]
    for f in feature_keys:
        row = weighted_tbl.get(f, {}).get("iou", {})
        s = float(row.get("spearman_weighted_by_case_n", 0.0))
        p = float(row.get("pearson_weighted_by_case_n", 0.0))
        top_iou_case_weighted.append(
            {
                "feature": f,
                "spearman_weighted_by_case_n": s,
                "pearson_weighted_by_case_n": p,
                "abs_spearman_weighted_by_case_n": abs(s),
            }
        )
    top_iou_case_weighted.sort(key=lambda r: r["abs_spearman_weighted_by_case_n"], reverse=True)

    stage_desc_vs_rebuild = _corr_table(
        stage_rows,
        ["desc_dimension_score", "plan_dimension_score", "plan_minus_desc_dimension_score"],
        y_keys,
    )
    stage_desc_vs_rebuild_by_case = _corr_table_by_case(
        stage_rows,
        ["desc_dimension_score", "plan_dimension_score", "plan_minus_desc_dimension_score"],
        y_keys,
        case_key="case",
    )

    # ---------- D: description metric validity ----------
    desc_x_keys = ["auto_score", "strict_material_f1", "coarse_material_f1", "dimension_score"]
    desc_y_keys = y_keys
    desc_rows: List[Dict[str, float]] = []
    per_case_desc_n: Dict[str, int] = {}
    for c in cases:
        metrics_obj = _load_json(c.metrics_path)
        items = metrics_obj.get("items", [])
        if not isinstance(items, list):
            continue
        rebuild_by_building = {}
        for it in items:
            if not isinstance(it, dict):
                continue
            bname = str(it.get("building", "")).strip()
            mm = it.get("metrics", {})
            if not bname or not isinstance(mm, dict):
                continue
            rebuild_by_building[bname] = {
                "case": f"{c.dataset_key}/{c.model_key}",
                "iou": float(mm.get("iou", 0.0) or 0.0),
                "f1": float(mm.get("f1", 0.0) or 0.0),
                "material_match": float(mm.get("material_match", 0.0) or 0.0),
                "coarse_material_match": float(mm.get("coarse_material_match", 0.0) or 0.0),
                "correct_placement_rate": float(mm.get("correct_placement_rate", 0.0) or 0.0),
            }

        desc_map = desc_maps.setdefault((c.dataset_name, c.model_key), _load_desc_item_map(outputs_root, c.dataset_name, c.model_key))
        if not desc_map:
            continue
        n = 0
        for bname, dvals in desc_map.items():
            if not bname or bname not in rebuild_by_building:
                continue
            row = {
                "case": str(rebuild_by_building[bname].get("case", "")),
                "auto_score": float(dvals.get("auto_score", 0.0) or 0.0),
                "strict_material_f1": float(dvals.get("strict_material_f1", 0.0) or 0.0),
                "coarse_material_f1": float(dvals.get("coarse_material_f1", 0.0) or 0.0),
                "dimension_score": float(dvals.get("dimension_score", 0.0) or 0.0),
                **rebuild_by_building[bname],
            }
            desc_rows.append(row)
            n += 1
        per_case_desc_n[f"{c.dataset_key}/{c.model_key}"] = n

    desc_corr = _corr_table(desc_rows, desc_x_keys, desc_y_keys)
    desc_corr_by_case = _corr_table_by_case(desc_rows, desc_x_keys, desc_y_keys, case_key="case")
    pooled_spearman_summary = {
        xk: {yk: float(desc_corr.get(xk, {}).get(yk, {}).get("spearman", 0.0)) for yk in desc_y_keys}
        for xk in desc_x_keys
    }
    within_case_weighted_spearman_summary = {
        xk: {
            yk: float(
                desc_corr_by_case.get("weighted_by_case_n", {})
                .get(xk, {})
                .get(yk, {})
                .get("spearman_weighted_by_case_n", 0.0)
            )
            for yk in desc_y_keys
        }
        for xk in desc_x_keys
    }

    stage_preservation = {
        "analyzed_rows": len(stage_rows),
        "desc_dim_score_mean": _mean(float(r.get("desc_dimension_score", 0.0)) for r in stage_rows),
        "plan_dim_score_mean": _mean(float(r.get("plan_dimension_score", 0.0)) for r in stage_rows),
        "plan_minus_desc_mean": _mean(float(r.get("plan_minus_desc_dimension_score", 0.0)) for r in stage_rows),
        "plan_below_desc_rate": _mean(
            1.0 if float(r.get("plan_dimension_score", 0.0)) < float(r.get("desc_dimension_score", 0.0)) else 0.0
            for r in stage_rows
        ),
        "stage_metric_correlations": stage_desc_vs_rebuild,
        "stage_metric_correlations_by_case": stage_desc_vs_rebuild_by_case,
    }

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "main_json": str(main_json),
            "outputs_root": str(outputs_root),
            "datasets_root": str(datasets_root),
            "op_budget": int(args.op_budget),
            "extra_budgets": budgets,
            "building_pattern": str(args.building_pattern),
        },
        "A_stage_isolation": {
            "method": (
                "Inject GT-derived oracle-like cuboid representation directly into plan/render space; "
                "measure budgeted upper-bound under current operation family."
            ),
            "oracle_copy_theoretical": {"iou": 1.0, "f1": 1.0},
            "per_dataset": a_per_dataset,
            "all_200_budget_iou_mean": _mean(all_budget_iou),
            "main_openai_iou_all_200": main_openai_iou,
            "main_claude_iou_all_200": main_claude_iou,
            "budget_gap_vs_openai_main_iou": _mean(all_budget_iou) - main_openai_iou,
            "budget_gap_vs_claude_main_iou": _mean(all_budget_iou) - main_claude_iou,
            "description_to_plan_dimension_preservation": stage_preservation,
        },
        "B_plan_fidelity_audit": {
            "analyzed_rows": len(plan_rows),
            "rows_per_case": per_case_counts,
            "feature_metric_correlations": plan_corr,
            "feature_metric_correlations_by_case": plan_corr_by_case,
            "binary_feature_effects": plan_bin,
            "binary_feature_prevalence": binary_prevalence,
            "strict_blocking_issue_frequency": dict(sorted(strict_issue_freq.items())),
            "top_predictors_iou": top_iou,
            "top_predictors_iou_case_weighted": top_iou_case_weighted,
        },
        "C_representation_ceiling": {
            "method": (
                "Approximate diagnostic: GT voxel -> disjoint cuboid partition (x-runs merged over z/y), "
                "then top-K cuboids by volume under op budget."
            ),
            "not_a_global_optimum": True,
            "renderer_supported_ops": ["fill", "carve", "set"],
            "observed_plan_op_kinds": dict(sorted(op_kind_counts.items())),
            "per_dataset": c_per_dataset,
            "all_200_budget_iou_mean": _mean(all_budget_iou),
            "all_200_exact_ops_mean": _mean(all_exact_ops),
            "all_200_exact_ops_std": _stdev(all_exact_ops),
            "per_dataset_exceed_budget_rate": {
                ds: _mean(
                    1.0 if float(x) > float(args.op_budget) else 0.0
                    for x in [
                        float(_representation_diagnostic_for_vox(_load_gt_voxels(b / "gt" / "voxels.npy"), budgets)["exact_ops_count"])
                        for b in sorted((datasets_root / ds).glob(args.building_pattern))
                        if (b / "gt" / "voxels.npy").is_file()
                    ]
                )
                for ds in sorted(set(c.dataset_name for c in cases))
            },
        },
        "D_description_metric_validity": {
            "analyzed_rows": len(desc_rows),
            "rows_per_case": per_case_desc_n,
            "metric_correlations": desc_corr,
            "metric_correlations_by_case": desc_corr_by_case,
            "pooled_spearman_summary": pooled_spearman_summary,
            "within_case_weighted_spearman_summary": within_case_weighted_spearman_summary,
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_markdown(payload), encoding="utf-8")

    print(f"[verify_bottleneck_hypothesis] wrote json: {out_json}")
    print(f"[verify_bottleneck_hypothesis] wrote md:   {out_md}")


if __name__ == "__main__":
    main()
