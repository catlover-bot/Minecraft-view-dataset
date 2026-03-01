# External Validity (OpenAI GPT-5 mini)

## Condition Table (v1 vs v4)

| condition | v1 IoU | v4 IoU | gap(v4-v1) | v1 F1 | v4 F1 | gap(v4-v1) | v1 material | v4 material | gap(v4-v1) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_stack | 0.3031 | 0.2040 | -0.0992 | 0.4591 | 0.3341 | -0.1250 | 0.2271 | 0.3017 | 0.0746 |
| material_only | 0.2872 | 0.1899 | -0.0974 | 0.4398 | 0.3142 | -0.1255 | 0.2193 | 0.2934 | 0.0741 |
| schema_only | 0.2790 | 0.1840 | -0.0950 | 0.4302 | 0.3056 | -0.1246 | 0.1614 | 0.1939 | 0.0325 |
| self_refine_only | 0.3038 | 0.2048 | -0.0989 | 0.4593 | 0.3349 | -0.1244 | 0.1176 | 0.1786 | 0.0611 |

## Best Condition by Metric

- v1: IoU=self_refine_only, F1=self_refine_only, material=full_stack
- v4: IoU=self_refine_only, F1=self_refine_only, material=full_stack

Interpretation:
- v1 と v4 で最良条件が一致すれば、分布を跨いだ再現性が高い。
- 差が大きい場合は分布依存性が強く、追加の正則化/条件分岐が必要。