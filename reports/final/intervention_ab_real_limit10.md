# Real Agent 実験: 失敗タイプ介入 + 2段生成A/B（limit=10）

- 作成時刻: `2026-03-12T12:02:57.340225+00:00`
- building_pattern: `building_*`
- limit: `10`
- low_iou_threshold: `0.20`

## v1/OpenAI

| variant | IoU | F1 | material | placement | overbuild率 | underbuild率 | mission失敗率 | low-IoU率 | ΔIoU | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline tuned (existing) | 0.3283 | 0.4853 | 0.4067 | 0.1980 | 70.00% | 20.00% | 0.00% | 20.00% | +0.0000 | +0.0000 |
| Two-stage OFF | 0.3369 | 0.4920 | 0.4157 | 0.2148 | 60.00% | 20.00% | 0.00% | 20.00% | +0.0086 | +0.0068 |
| Overbuild intervention | 0.3396 | 0.4944 | 0.3806 | 0.2132 | 40.00% | 40.00% | 0.00% | 20.00% | +0.0112 | +0.0092 |
| Underbuild intervention | 0.3320 | 0.4831 | 0.3335 | 0.1922 | 40.00% | 40.00% | 0.00% | 20.00% | +0.0037 | -0.0021 |
| Material intervention | 0.3373 | 0.4932 | 0.4085 | 0.2274 | 40.00% | 50.00% | 0.00% | 20.00% | +0.0090 | +0.0080 |
| Mission stability intervention | 0.3290 | 0.4856 | 0.4166 | 0.2032 | 70.00% | 20.00% | 0.00% | 20.00% | +0.0007 | +0.0003 |

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
| Baseline tuned (existing) | 0.2553 | 0.4033 | 0.2638 | 0.1130 | 50.00% | 20.00% | 0.00% | 30.00% | +0.0000 | +0.0000 |
| Two-stage OFF | 0.2483 | 0.3960 | 0.2083 | 0.0954 | 20.00% | 40.00% | 0.00% | 20.00% | -0.0070 | -0.0073 |
| Overbuild intervention | 0.2211 | 0.3533 | 0.1964 | 0.1064 | 30.00% | 40.00% | 0.00% | 30.00% | -0.0342 | -0.0500 |
| Underbuild intervention | 0.2312 | 0.3662 | 0.2011 | 0.0932 | 40.00% | 50.00% | 0.00% | 20.00% | -0.0241 | -0.0370 |
| Material intervention | 0.2633 | 0.4153 | 0.2058 | 0.0982 | 40.00% | 40.00% | 0.00% | 10.00% | +0.0080 | +0.0121 |
| Mission stability intervention | 0.2539 | 0.4014 | 0.2520 | 0.1089 | 60.00% | 30.00% | 0.00% | 20.00% | -0.0014 | -0.0018 |

| variant | overbuild | underbuild | material_mismatch | mission_failure |
|---|---:|---:|---:|---:|
| Baseline tuned (existing) | 2 | 1 | 0 | 0 |
| Two-stage OFF | 0 | 1 | 1 | 0 |
| Overbuild intervention | 0 | 2 | 1 | 0 |
| Underbuild intervention | 0 | 1 | 1 | 0 |
| Material intervention | 0 | 0 | 1 | 0 |
| Mission stability intervention | 1 | 1 | 0 | 0 |

## v4/OpenAI

| variant | IoU | F1 | material | placement | overbuild率 | underbuild率 | mission失敗率 | low-IoU率 | ΔIoU | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline tuned (existing) | 0.2318 | 0.3719 | 0.2359 | 0.0857 | 50.00% | 20.00% | 0.00% | 30.00% | +0.0000 | +0.0000 |
| Two-stage OFF | 0.1744 | 0.2942 | 0.3710 | 0.1497 | 10.00% | 60.00% | 0.00% | 70.00% | -0.0574 | -0.0776 |
| Overbuild intervention | 0.1751 | 0.2954 | 0.3766 | 0.1535 | 20.00% | 70.00% | 0.00% | 80.00% | -0.0567 | -0.0765 |
| Underbuild intervention | 0.1772 | 0.2985 | 0.3728 | 0.1509 | 10.00% | 60.00% | 0.00% | 70.00% | -0.0546 | -0.0734 |
| Material intervention | 0.1664 | 0.2787 | 0.3710 | 0.1526 | 10.00% | 70.00% | 0.00% | 70.00% | -0.0654 | -0.0931 |
| Mission stability intervention | 0.2168 | 0.3493 | 0.2436 | 0.0923 | 50.00% | 40.00% | 0.00% | 40.00% | -0.0150 | -0.0226 |

| variant | overbuild | underbuild | material_mismatch | mission_failure |
|---|---:|---:|---:|---:|
| Baseline tuned (existing) | 1 | 2 | 0 | 0 |
| Two-stage OFF | 0 | 7 | 0 | 0 |
| Overbuild intervention | 0 | 8 | 0 | 0 |
| Underbuild intervention | 0 | 7 | 0 | 0 |
| Material intervention | 0 | 7 | 0 | 0 |
| Mission stability intervention | 1 | 3 | 0 | 0 |

## v4/Claude

| variant | IoU | F1 | material | placement | overbuild率 | underbuild率 | mission失敗率 | low-IoU率 | ΔIoU | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline tuned (existing) | 0.1880 | 0.3101 | 0.2309 | 0.1057 | 50.00% | 40.00% | 0.00% | 60.00% | +0.0000 | +0.0000 |
| Two-stage OFF | 0.1750 | 0.2936 | 0.2398 | 0.1066 | 30.00% | 50.00% | 0.00% | 70.00% | -0.0130 | -0.0166 |
| Overbuild intervention | 0.1651 | 0.2798 | 0.2775 | 0.1234 | 30.00% | 50.00% | 0.00% | 70.00% | -0.0229 | -0.0304 |
| Underbuild intervention | 0.1653 | 0.2801 | 0.2650 | 0.1185 | 20.00% | 50.00% | 0.00% | 70.00% | -0.0227 | -0.0301 |
| Material intervention | 0.1648 | 0.2796 | 0.2643 | 0.1199 | 20.00% | 60.00% | 0.00% | 70.00% | -0.0232 | -0.0305 |
| Mission stability intervention | 0.1873 | 0.3091 | 0.2205 | 0.1025 | 50.00% | 40.00% | 0.00% | 60.00% | -0.0007 | -0.0011 |

| variant | overbuild | underbuild | material_mismatch | mission_failure |
|---|---:|---:|---:|---:|
| Baseline tuned (existing) | 3 | 3 | 0 | 0 |
| Two-stage OFF | 2 | 4 | 1 | 0 |
| Overbuild intervention | 2 | 4 | 1 | 0 |
| Underbuild intervention | 2 | 4 | 1 | 0 |
| Material intervention | 2 | 5 | 0 | 0 |
| Mission stability intervention | 3 | 3 | 0 | 0 |

## メモ
- 失敗分類は low-IoU建物だけを対象。
- 優先順位は mission_failure -> overbuild/underbuild -> material_mismatch。
- `mission_stable_exec` は plan/renderを変えず、実行条件だけ変更。
