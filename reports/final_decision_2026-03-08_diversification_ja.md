# 最終意思決定メモ（2026-03-08）

## 対象
OpenAI経路の candidate diversification 設定比較（limit=10, v1/v4）

- 比較A: `candidate_diversification_high_risk_only=ON`（diversify_on）
- 比較B: `candidate_diversification_high_risk_only=OFF`（diversify_all: 常時多様化ON）

## 結果

### buildings_100_v1
- diversify_on: IoU 0.3221, F1 0.4775, correct_placement_rate 0.1932, material_match 0.3697
- diversify_all: IoU 0.3188, F1 0.4746, correct_placement_rate 0.1826, material_match 0.3513
- 差分（all - on）: IoU -0.0032, F1 -0.0029, correct_placement_rate -0.0106, material_match -0.0184

### buildings_100_v4
- diversify_on: IoU 0.2058, F1 0.3367, correct_placement_rate 0.1219, material_match 0.3207
- diversify_all: IoU 0.2017, F1 0.3305, correct_placement_rate 0.1169, material_match 0.3099
- 差分（all - on）: IoU -0.0040, F1 -0.0062, correct_placement_rate -0.0050, material_match -0.0107

## 結論
- 常時多様化ON（diversify_all）は、今回設定では v1/v4 の両方で悪化。
- 最終採用設定は `candidate_diversification_high_risk_only=ON`。

## 保持する最終比較ファイル
- `datasets/buildings_100_v1/metrics_levels_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_diversify_on_s40_l10.json`
- `datasets/buildings_100_v4/metrics_levels_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_diversify_on_s40_l10.json`

## 非採用結果（アーカイブ）
- `datasets/_archive/rejected_runs_2026-03-08/metrics_levels_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_diversify_all_s40_l10.v1.json`
- `datasets/_archive/rejected_runs_2026-03-08/metrics_levels_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_diversify_all_s40_l10.v4.json`
