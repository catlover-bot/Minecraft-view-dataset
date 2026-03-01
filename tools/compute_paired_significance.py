#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_METRICS = ("iou", "f1", "material_match", "coarse_material_match", "component_f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired significance tests between two metrics JSON files.")
    parser.add_argument("--baseline", required=True, help="Baseline metrics_levels*.json")
    parser.add_argument("--variant", required=True, help="Variant metrics_levels*.json")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--label", default="", help="Optional label for this comparison")
    parser.add_argument("--metrics", nargs="*", default=list(DEFAULT_METRICS), help="Metric keys to test")
    parser.add_argument("--bootstrap_samples", type=int, default=10000, help="Bootstrap resamples")
    parser.add_argument("--permutation_samples", type=int, default=20000, help="Sign-flip permutation samples")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    return parser.parse_args()


def _load(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _item_metric_map(obj: Dict, metric: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in obj.get("items", []) if isinstance(obj.get("items"), list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("building", "")).strip()
        if not name:
            continue
        metrics = item.get("metrics", {}) if isinstance(item.get("metrics"), dict) else {}
        if metric not in metrics:
            continue
        try:
            out[name] = float(metrics[metric])
        except Exception:
            continue
    return out


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _stdev(values: List[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mu = _mean(values)
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    return math.sqrt(max(0.0, var))


def _bootstrap_ci_mean_diff(diffs: List[float], rng: random.Random, samples: int) -> Tuple[float, float]:
    n = len(diffs)
    if n == 0:
        return (0.0, 0.0)
    means: List[float] = []
    for _ in range(max(1, samples)):
        sm = 0.0
        for _j in range(n):
            sm += diffs[rng.randrange(0, n)]
        means.append(sm / n)
    means.sort()
    lo_idx = int(0.025 * (len(means) - 1))
    hi_idx = int(0.975 * (len(means) - 1))
    return float(means[lo_idx]), float(means[hi_idx])


def _paired_sign_flip_pvalue(diffs: List[float], rng: random.Random, samples: int) -> float:
    n = len(diffs)
    if n == 0:
        return 1.0
    obs = abs(_mean(diffs))
    extreme = 0
    total = max(1, samples)
    for _ in range(total):
        sm = 0.0
        for d in diffs:
            sm += d if rng.random() < 0.5 else -d
        if abs(sm / n) >= obs:
            extreme += 1
    # add-one smoothing
    return float((extreme + 1) / (total + 1))


def _aggregate_metric(obj: Dict, metric: str) -> float:
    try:
        return float(obj["aggregate"]["metrics"][metric])
    except Exception:
        return 0.0


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline).resolve()
    variant_path = Path(args.variant).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    b = _load(baseline_path)
    v = _load(variant_path)
    rng = random.Random(int(args.seed))

    metrics_out: Dict[str, Dict] = {}
    for metric in args.metrics:
        bmap = _item_metric_map(b, metric)
        vmap = _item_metric_map(v, metric)
        common = sorted(set(bmap.keys()) & set(vmap.keys()))
        diffs = [float(vmap[k] - bmap[k]) for k in common]

        mu_diff = _mean(diffs)
        sd_diff = _stdev(diffs)
        dz = (mu_diff / sd_diff) if sd_diff > 0 else 0.0
        ci_lo, ci_hi = _bootstrap_ci_mean_diff(diffs, rng, int(args.bootstrap_samples))
        p_val = _paired_sign_flip_pvalue(diffs, rng, int(args.permutation_samples))
        improved = sum(1 for d in diffs if d > 0.0)
        worsened = sum(1 for d in diffs if d < 0.0)
        tied = len(diffs) - improved - worsened

        metrics_out[metric] = {
            "n_pairs": len(common),
            "baseline_aggregate": _aggregate_metric(b, metric),
            "variant_aggregate": _aggregate_metric(v, metric),
            "paired_mean_diff": mu_diff,
            "paired_std_diff": sd_diff,
            "effect_size_dz": dz,
            "bootstrap_ci95": [ci_lo, ci_hi],
            "sign_flip_pvalue_two_sided": p_val,
            "improved_count": improved,
            "worsened_count": worsened,
            "tied_count": tied,
        }

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label or f"{baseline_path.name}__vs__{variant_path.name}",
        "baseline": str(baseline_path),
        "variant": str(variant_path),
        "settings": {
            "bootstrap_samples": int(args.bootstrap_samples),
            "permutation_samples": int(args.permutation_samples),
            "seed": int(args.seed),
            "metrics": list(args.metrics),
        },
        "results": metrics_out,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[compute_paired_significance] wrote: {out_path}")
    for metric in args.metrics:
        r = metrics_out[metric]
        print(
            f"  {metric}: diff={r['paired_mean_diff']:+.4f} "
            f"CI95=[{r['bootstrap_ci95'][0]:+.4f},{r['bootstrap_ci95'][1]:+.4f}] "
            f"p={r['sign_flip_pvalue_two_sided']:.6f} "
            f"improved={r['improved_count']}/{r['n_pairs']}"
        )


if __name__ == "__main__":
    main()

