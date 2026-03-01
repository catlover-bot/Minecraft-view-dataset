#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize external validity across dataset distributions.")
    parser.add_argument("--v1_summary", required=True, help="ablation_summary_*.json for v1 distribution")
    parser.add_argument("--v4_summary", required=True, help="ablation_summary_*.json for v4 distribution")
    parser.add_argument("--out_json", required=True, help="Output JSON")
    parser.add_argument("--out_md", required=True, help="Output Markdown report")
    parser.add_argument("--title", default="External Validity Summary")
    return parser.parse_args()


def _load(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def _best_condition(summary: Dict[str, Dict[str, float]], metric: str) -> str:
    conds = list(summary.keys())
    if not conds:
        return ""
    return max(conds, key=lambda c: float(summary[c].get(metric, -1e18)))


def main() -> None:
    args = parse_args()
    v1_path = Path(args.v1_summary).resolve()
    v4_path = Path(args.v4_summary).resolve()
    out_json = Path(args.out_json).resolve()
    out_md = Path(args.out_md).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    v1 = _load(v1_path)
    v4 = _load(v4_path)
    s1 = v1.get("summary", {})
    s4 = v4.get("summary", {})
    if not isinstance(s1, dict) or not isinstance(s4, dict):
        raise SystemExit("invalid ablation summary format")

    conditions = sorted(set(s1.keys()) & set(s4.keys()))
    metrics = ["iou", "f1", "material_match"]
    rows: List[Dict] = []
    for cond in conditions:
        row = {"condition": cond}
        for m in metrics:
            v1v = float(s1.get(cond, {}).get(m, 0.0))
            v4v = float(s4.get(cond, {}).get(m, 0.0))
            row[f"v1_{m}"] = v1v
            row[f"v4_{m}"] = v4v
            row[f"gap_v4_minus_v1_{m}"] = v4v - v1v
        rows.append(row)

    best = {
        "v1": {m: _best_condition(s1, m) for m in metrics},
        "v4": {m: _best_condition(s4, m) for m in metrics},
    }

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "title": args.title,
        "v1_summary_path": str(v1_path),
        "v4_summary_path": str(v4_path),
        "conditions": conditions,
        "rows": rows,
        "best_condition": best,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append(f"# {args.title}")
    lines.append("")
    lines.append("## Condition Table (v1 vs v4)")
    lines.append("")
    lines.append("| condition | v1 IoU | v4 IoU | gap(v4-v1) | v1 F1 | v4 F1 | gap(v4-v1) | v1 material | v4 material | gap(v4-v1) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            "| {c} | {v1i} | {v4i} | {gi} | {v1f} | {v4f} | {gf} | {v1m} | {v4m} | {gm} |".format(
                c=r["condition"],
                v1i=_fmt(r["v1_iou"]),
                v4i=_fmt(r["v4_iou"]),
                gi=_fmt(r["gap_v4_minus_v1_iou"]),
                v1f=_fmt(r["v1_f1"]),
                v4f=_fmt(r["v4_f1"]),
                gf=_fmt(r["gap_v4_minus_v1_f1"]),
                v1m=_fmt(r["v1_material_match"]),
                v4m=_fmt(r["v4_material_match"]),
                gm=_fmt(r["gap_v4_minus_v1_material_match"]),
            )
        )
    lines.append("")
    lines.append("## Best Condition by Metric")
    lines.append("")
    lines.append(f"- v1: IoU={best['v1']['iou']}, F1={best['v1']['f1']}, material={best['v1']['material_match']}")
    lines.append(f"- v4: IoU={best['v4']['iou']}, F1={best['v4']['f1']}, material={best['v4']['material_match']}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- v1 と v4 で最良条件が一致すれば、分布を跨いだ再現性が高い。")
    lines.append("- 差が大きい場合は分布依存性が強く、追加の正則化/条件分岐が必要。")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"[summarize_external_validity] wrote: {out_json}")
    print(f"[summarize_external_validity] wrote: {out_md}")


if __name__ == "__main__":
    main()

