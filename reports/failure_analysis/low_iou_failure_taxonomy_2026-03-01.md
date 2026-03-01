# 低IoU失敗ケースの系統分析

## 判定設定
- low_iou_threshold: `0.200`
- material_match_threshold: `0.200`
- plan_collapse_ratio: `< 0.55` または `> 1.80`
- min_operations_for_non_collapse: `25`

## 失敗タイプ別件数（全体）

| cause | count | share of low-IoU | mean IoU | mean F1 | mean material |
|---|---:|---:|---:|---:|---:|
| fallback | 105 | 76.1% | 0.1553 | 0.2674 | 0.2409 |
| material_mismatch | 26 | 18.8% | 0.1613 | 0.2768 | 0.2344 |
| plan_collapse | 7 | 5.1% | 0.1685 | 0.2874 | 0.3003 |

- total low-IoU cases: `138`

## 失敗タイプ別件数（設定別）

| config | low-IoU count | fallback | plan_collapse | material_mismatch |
|---|---:|---:|---:|---:|
| v1/openai | 9 | 4 | 0 | 5 |
| v1/claude | 20 | 13 | 1 | 6 |
| v4/openai | 49 | 30 | 6 | 13 |
| v4/claude | 60 | 58 | 0 | 2 |

## 次の修正優先順位

| priority | cause | score | rationale | next fix |
|---:|---|---:|---|---|
| 1 | fallback | 4.6894 | count=105, mean_iou=0.1553 | 出力スキーマ拘束強化と parser自動修復ルール拡張（ops配列復旧依存を減らす） |
| 2 | material_mismatch | 1.0067 | count=26, mean_iou=0.1613 | 材質budget再投影と role固定の閾値再調整（budget違反削減を最優先） |
| 3 | plan_collapse | 0.2205 | count=7, mean_iou=0.1685 | 粗形状の下限保証（最小wall/roof/floor密度）と operation最小本数ガード |

## 参考: 最悪ケース例（各原因Top）

### fallback

| config | building | iou | f1 | material | parse_method | budget_viol | op_count | pred/gt |
|---|---|---:|---:|---:|---|---:|---:|---:|
| v4/openai | building_076 | 0.0320 | 0.0620 | 0.4311 | operation_object_scan | 3 | 41 | 0.040 |
| v4/claude | building_065 | 0.0624 | 0.1175 | 0.3176 | ops_array:operations | 1 | 31 | 0.117 |
| v4/openai | building_011 | 0.0732 | 0.1364 | 0.0319 | operation_object_scan | 1 | 55 | 0.241 |
| v4/openai | building_064 | 0.0785 | 0.1456 | 0.1242 | operation_object_scan | 3 | 44 | 1.493 |
| v4/claude | building_024 | 0.0830 | 0.1533 | 0.0156 | operation_object_scan | 3 | 81 | 0.188 |
| v4/claude | building_021 | 0.0886 | 0.1628 | 0.1183 | ops_array:operations | 2 | 32 | 1.015 |
| v4/claude | building_005 | 0.0895 | 0.1643 | 0.1518 | operation_object_scan | 1 | 38 | 1.431 |
| v4/claude | building_003 | 0.0953 | 0.1740 | 0.1565 | operation_object_scan | 0 | 29 | 0.508 |

### plan_collapse

| config | building | iou | f1 | material | parse_method | budget_viol | op_count | pred/gt |
|---|---|---:|---:|---:|---|---:|---:|---:|
| v4/openai | building_024 | 0.1090 | 0.1966 | 0.2824 | extract_json_object | 0 | 40 | 0.205 |
| v4/openai | building_000 | 0.1597 | 0.2754 | 0.3053 |  | 0 | 21 | 0.902 |
| v1/claude | building_037 | 0.1618 | 0.2785 | 0.3756 | extract_json_object | 0 | 21 | 0.960 |
| v4/openai | building_073 | 0.1744 | 0.2970 | 0.2241 | extract_json_object | 0 | 45 | 0.700 |
| v4/openai | building_014 | 0.1759 | 0.2991 | 0.3554 | extract_json_object | 0 | 61 | 0.877 |
| v4/openai | building_097 | 0.1993 | 0.3323 | 0.3193 | extract_json_object | 0 | 58 | 1.253 |
| v4/openai | building_033 | 0.1995 | 0.3327 | 0.2402 | extract_json_object | 0 | 23 | 1.238 |

### material_mismatch

| config | building | iou | f1 | material | parse_method | budget_viol | op_count | pred/gt |
|---|---|---:|---:|---:|---|---:|---:|---:|
| v4/openai | building_086 | 0.1154 | 0.2069 | 0.1858 | extract_json_object | 0 | 48 | 0.183 |
| v4/claude | building_045 | 0.1168 | 0.2091 | 0.4499 | extract_json_object | 2 | 109 | 0.763 |
| v4/openai | building_045 | 0.1177 | 0.2106 | 0.0705 | extract_json_object | 2 | 76 | 0.823 |
| v4/openai | building_067 | 0.1190 | 0.2127 | 0.1378 |  | 2 | 78 | 0.672 |
| v4/openai | building_027 | 0.1304 | 0.2306 | 0.1731 | extract_json_object | 0 | 60 | 0.835 |
| v4/openai | building_039 | 0.1342 | 0.2367 | 0.2170 | extract_json_object | 4 | 73 | 1.192 |
| v1/claude | building_025 | 0.1350 | 0.2379 | 0.1156 | extract_json_object | 0 | 22 | 1.719 |
| v4/openai | building_021 | 0.1382 | 0.2429 | 0.2553 | extract_json_object | 1 | 50 | 0.716 |
