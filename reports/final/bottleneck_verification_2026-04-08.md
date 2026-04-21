# Bottleneck Verification (No New LLM Runs)

- created_at: `2026-04-08T01:23:04.348460+00:00`
- op_budget: `260`

## A. Stage-Isolation Upper-Bound (Oracle-like GT Cuboid Injection)

- `oracle_copy` (GT voxel copy): theoretical IoU/F1=1.0 (sanity upper bound).
- `oracle_budgeted_cuboids`: GT-derived cuboid decomposition under operation budget.

| dataset | exact_ops_mean | budget_iou_mean | budget_f1_mean | budget_recall_mean |
|---|---:|---:|---:|---:|
| buildings_100_v1 | 81.9 | 1.0000 | 1.0000 | 1.0000 |
| buildings_100_v4 | 683.1 | 0.8630 | 0.9216 | 0.8630 |

- all-200 budgeted upper-bound (IoU): `0.9315` vs Main OpenAI `0.2375` / Main Claude `0.2055`
- description->plan dimension retention: mean desc `0.5971` -> plan `0.6274` (delta `+0.0303`)

## B. Plan Fidelity Audit

- analyzed_rows: `400`
- strongest |spearman| predictors for IoU:
  - `operations_assigned_role_count`: spearman=-0.2060, pearson=-0.1821
  - `plan_operation_count`: spearman=-0.1989, pearson=-0.1775
  - `strict_blocking_count`: spearman=+0.1948, pearson=+0.1679
  - `has_strict_blocking`: spearman=+0.1948, pearson=+0.1679
  - `valid_strict`: spearman=-0.1948, pearson=-0.1679
  - `budget_violation_count`: spearman=+0.1607, pearson=+0.1579
  - `bbox_outside_operation_count`: spearman=+0.1146, pearson=+0.0775
  - `role_fixed_block_count`: spearman=+0.0803, pearson=+0.0625
- case-weighted IoU predictors (controls v1/v4 + model mix):
  - `operations_assigned_role_count`: spearman_w=+0.1429, pearson_w=+0.1409
  - `plan_operation_count`: spearman_w=+0.1429, pearson_w=+0.1427
  - `strict_blocking_count`: spearman_w=+0.1133, pearson_w=+0.1046
  - `has_strict_blocking`: spearman_w=+0.1133, pearson_w=+0.1046
  - `valid_strict`: spearman_w=-0.1133, pearson_w=-0.1046
  - `coerce_repaired_count`: spearman_w=+0.0931, pearson_w=+0.0335

## C. Representation Ceiling Diagnostic

- op_budget=260, all-200 budgeted IoU mean: `0.9315`
- exact_ops_count mean: `382.5` (perfect under this decomposition).

## D. Description Metric Validity

- analyzed_rows: `400`
- Spearman(description -> rebuild) pooled over Main cases:
  - `auto_score`: IoU=+0.3390, F1=+0.3390, material=+0.1968, coarse=+0.1420, correct_placement=+0.2410
  - `strict_material_f1`: IoU=+0.2324, F1=+0.2324, material=+0.1892, coarse=+0.1170, correct_placement=+0.2249
  - `coarse_material_f1`: IoU=+0.1041, F1=+0.1041, material=+0.1886, coarse=+0.1118, correct_placement=+0.2125
  - `dimension_score`: IoU=+0.4471, F1=+0.4471, material=-0.0150, coarse=+0.0475, correct_placement=+0.0197
- Spearman(description -> rebuild) weighted within-case:
  - `auto_score`: IoU=+0.1075, F1=+0.1075, material=+0.1735, coarse=+0.0377, correct_placement=+0.1846
  - `strict_material_f1`: IoU=+0.0337, F1=+0.0337, material=+0.1537, coarse=+0.0456, correct_placement=+0.1722
  - `coarse_material_f1`: IoU=+0.0229, F1=+0.0229, material=+0.1336, coarse=-0.0016, correct_placement=+0.1576
  - `dimension_score`: IoU=+0.1711, F1=+0.1711, material=+0.0222, coarse=+0.0174, correct_placement=-0.0043

