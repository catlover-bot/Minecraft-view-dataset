# Real Agent 実験: 失敗タイプ介入 + 2段生成A/B（limit=10）

- 作成時刻: `2026-03-15T13:22:49.575363+00:00`
- placement_mode: `hand_place` (Hand Place (use))
- building_pattern: `building_*`
- limit: `10`
- low_iou_threshold: `0.20`

## v1/OpenAI

| variant | IoU | F1 | material | placement | overbuild率 | underbuild率 | mission失敗率 | low-IoU率 | ΔIoU | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline tuned (existing) | 0.3464 | 0.5007 | 0.2302 | 0.1129 | 70.00% | 20.00% | 0.00% | 20.00% | +0.0000 | +0.0000 |
| Two-stage OFF | 0.3323 | 0.4859 | 0.2676 | 0.1368 | 50.00% | 30.00% | 0.00% | 20.00% | -0.0141 | -0.0148 |
| Overbuild intervention | 0.3305 | 0.4841 | 0.2522 | 0.1332 | 50.00% | 40.00% | 0.00% | 20.00% | -0.0159 | -0.0166 |
| Underbuild intervention | 0.3293 | 0.4825 | 0.2552 | 0.1318 | 50.00% | 30.00% | 0.00% | 20.00% | -0.0171 | -0.0182 |
| Material intervention | 0.3249 | 0.4780 | 0.2413 | 0.1225 | 40.00% | 30.00% | 0.00% | 20.00% | -0.0214 | -0.0227 |
| Mission stability intervention | 0.3348 | 0.4895 | 0.1986 | 0.0955 | 70.00% | 20.00% | 0.00% | 20.00% | -0.0115 | -0.0112 |

| variant | overbuild | underbuild | material_mismatch | mission_failure |
|---|---:|---:|---:|---:|
| Baseline tuned (existing) | 0 | 2 | 0 | 0 |
| Two-stage OFF | 0 | 2 | 0 | 0 |
| Overbuild intervention | 0 | 2 | 0 | 0 |
| Underbuild intervention | 0 | 2 | 0 | 0 |
| Material intervention | 0 | 2 | 0 | 0 |
| Mission stability intervention | 0 | 2 | 0 | 0 |

## v1/Claude

| variant | IoU | F1 | material | placement | overbuild率 | underbuild率 | mission失敗率 | low-IoU率 | ΔIoU | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline tuned (existing) | 0.2603 | 0.4100 | 0.1647 | 0.0727 | 60.00% | 20.00% | 0.00% | 20.00% | +0.0000 | +0.0000 |
| Two-stage OFF | 0.2546 | 0.4042 | 0.1808 | 0.0815 | 40.00% | 40.00% | 0.00% | 10.00% | -0.0058 | -0.0058 |
| Overbuild intervention | 0.2552 | 0.4054 | 0.1789 | 0.0846 | 20.00% | 50.00% | 0.00% | 0.00% | -0.0051 | -0.0046 |
| Underbuild intervention | 0.2556 | 0.4055 | 0.1783 | 0.0820 | 30.00% | 40.00% | 0.00% | 10.00% | -0.0048 | -0.0045 |
| Material intervention | 0.2568 | 0.4074 | 0.1809 | 0.0823 | 40.00% | 40.00% | 0.00% | 10.00% | -0.0035 | -0.0026 |
| Mission stability intervention | 0.2627 | 0.4133 | 0.1606 | 0.0719 | 50.00% | 20.00% | 0.00% | 10.00% | +0.0023 | +0.0033 |

| variant | overbuild | underbuild | material_mismatch | mission_failure |
|---|---:|---:|---:|---:|
| Baseline tuned (existing) | 2 | 0 | 0 | 0 |
| Two-stage OFF | 1 | 0 | 0 | 0 |
| Overbuild intervention | 0 | 0 | 0 | 0 |
| Underbuild intervention | 0 | 1 | 0 | 0 |
| Material intervention | 0 | 1 | 0 | 0 |
| Mission stability intervention | 1 | 0 | 0 | 0 |

## v4/OpenAI

| variant | IoU | F1 | material | placement | overbuild率 | underbuild率 | mission失敗率 | low-IoU率 | ΔIoU | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline tuned (existing) | 0.2050 | 0.3349 | 0.1722 | 0.0603 | 10.00% | 50.00% | 0.00% | 50.00% | +0.0000 | +0.0000 |
| Two-stage OFF | 0.1481 | 0.2557 | 0.1674 | 0.0568 | 0.00% | 80.00% | 0.00% | 80.00% | -0.0570 | -0.0792 |
| Overbuild intervention | 0.1495 | 0.2573 | 0.1690 | 0.0585 | 10.00% | 80.00% | 0.00% | 80.00% | -0.0555 | -0.0775 |
| Underbuild intervention | 0.1514 | 0.2604 | 0.1721 | 0.0599 | 10.00% | 80.00% | 0.00% | 80.00% | -0.0536 | -0.0745 |
| Material intervention | - | - | - | - | - | - | - | - | - | - |
| Mission stability intervention | - | - | - | - | - | - | - | - | - | - |

| variant | overbuild | underbuild | material_mismatch | mission_failure |
|---|---:|---:|---:|---:|
| Baseline tuned (existing) | 0 | 4 | 1 | 0 |
| Two-stage OFF | 0 | 8 | 0 | 0 |
| Overbuild intervention | 0 | 8 | 0 | 0 |
| Underbuild intervention | 0 | 8 | 0 | 0 |
| Material intervention | 0 | 0 | 0 | 0 |
| Mission stability intervention | 0 | 0 | 0 | 0 |

## v4/Claude

| variant | IoU | F1 | material | placement | overbuild率 | underbuild率 | mission失敗率 | low-IoU率 | ΔIoU | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline tuned (existing) | - | - | - | - | - | - | - | - | - | - |
| Two-stage OFF | - | - | - | - | - | - | - | - | - | - |
| Overbuild intervention | - | - | - | - | - | - | - | - | - | - |
| Underbuild intervention | - | - | - | - | - | - | - | - | - | - |
| Material intervention | - | - | - | - | - | - | - | - | - | - |
| Mission stability intervention | - | - | - | - | - | - | - | - | - | - |

| variant | overbuild | underbuild | material_mismatch | mission_failure |
|---|---:|---:|---:|---:|
| Baseline tuned (existing) | 0 | 0 | 0 | 0 |
| Two-stage OFF | 0 | 0 | 0 | 0 |
| Overbuild intervention | 0 | 0 | 0 | 0 |
| Underbuild intervention | 0 | 0 | 0 | 0 |
| Material intervention | 0 | 0 | 0 | 0 |
| Mission stability intervention | 0 | 0 | 0 | 0 |

## メモ
- 失敗分類は low-IoU建物だけを対象。
- 優先順位は mission_failure -> overbuild/underbuild -> material_mismatch。
- `mission_stable_exec` は plan/renderを変えず、実行条件だけ変更。
