# External Validity (Claude Haiku 4.5)

## Condition Table (v1 vs v4)

| condition | v1 IoU | v4 IoU | gap(v4-v1) | v1 F1 | v4 F1 | gap(v4-v1) | v1 material | v4 material | gap(v4-v1) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_stack | 0.2797 | 0.1947 | -0.0849 | 0.4295 | 0.3220 | -0.1075 | 0.2141 | 0.2437 | 0.0296 |
| material_only | 0.2362 | 0.1718 | -0.0644 | 0.3769 | 0.2888 | -0.0880 | 0.1726 | 0.2281 | 0.0555 |
| schema_only | 0.2362 | 0.1720 | -0.0642 | 0.3768 | 0.2890 | -0.0878 | 0.1467 | 0.1844 | 0.0378 |
| self_refine_only | 0.2753 | 0.1952 | -0.0801 | 0.4240 | 0.3221 | -0.1019 | 0.1282 | 0.1627 | 0.0345 |

## Best Condition by Metric

- v1: IoU=full_stack, F1=full_stack, material=full_stack
- v4: IoU=self_refine_only, F1=self_refine_only, material=full_stack

Interpretation:
- v1 と v4 で最良条件が一致すれば、分布を跨いだ再現性が高い。
- 差が大きい場合は分布依存性が強く、追加の正則化/条件分岐が必要。