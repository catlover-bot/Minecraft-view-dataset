"""Paired inference with buildings as clusters; never impute failure as zero."""
from __future__ import annotations
import math
import numpy as np

def paired_cluster_ci(rows, metric, seed=20261013, replicates=2000):
    clusters = {}
    excluded = 0
    for row in rows:
        a, b = row.get('a'), row.get('b')
        if (row.get('status_a', 'success') != 'success' or
                row.get('status_b', 'success') != 'success' or
                not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or
                not math.isfinite(a) or not math.isfinite(b)):
            excluded += 1
            continue
        clusters.setdefault(row['scene_id'], []).append(b - a)
    vals = np.array([np.mean(clusters[k]) for k in sorted(clusters)])
    result = {'metric': metric, 'estimand': 'mean within-building paired difference B-A, equal building weight',
              'buildings': len(vals), 'paired_rows': sum(map(len, clusters.values())),
              'excluded_pairs': excluded, 'seed': seed, 'replicates': replicates}
    if len(vals) < 2:
        return dict(result, status='INSUFFICIENT_CLUSTERS', mean_delta=None, ci95=None)
    rng = np.random.default_rng(seed)
    draws = [float(np.mean(rng.choice(vals, len(vals), replace=True))) for _ in range(replicates)]
    return dict(result, status='success', mean_delta=float(vals.mean()),
                ci95=np.quantile(draws, [.025, .975]).tolist(), method='percentile building-cluster bootstrap')

def holm_adjust(pvalues):
    if any(not isinstance(p, (int, float)) or not 0 <= p <= 1 for p in pvalues):
        raise ValueError('p values must be finite and inside [0,1]')
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted = [None] * len(order)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(order) - rank) * pvalues[index]))
        adjusted[index] = previous
    return adjusted
