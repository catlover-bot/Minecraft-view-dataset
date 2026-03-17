# Outputs Directory Guide

このディレクトリは実験の出力専用です。

## i2t2b 出力

- `i2t2b/buildings_100_v1/`
- `i2t2b/buildings_100_v4/`

各データセット配下:

- `building_000/ ... building_099/`
  - `description_*`
  - `rebuild_plan_*`
  - `rebuild_world_*`
  - `rebuild_world_agentexec_*`（proxy）
  - `rebuild_world_agentexec_real_*`（real placement）
  - `rebuild_world_agentexec_hand_*`（creative手置き placement）
- `metrics/description/`
- `metrics/rebuild/`
  - `execution_gap*.json`（renderer上限 vs agent実運用の差分）

2026-03-10 時点の整理方針:
- 中間検証用の `shapefix*`, `*_self_refine_no_gt`（非tuned）, `capture_*`, `logs/` は削除済み
- 比較に使う本線のみ保持
  - Main（共通ハイパラ）: `*_common_v8_struct_self_refine_no_gt_tuned`
  - Supplementary（モデル別最適化）: `*_self_refine_no_gt_tuned`
  - 基本系: `rebuild_plan_schema_material_v5_repair_*`, `rebuild_world_schema_material_v5_repair_*`

方針:
- `datasets/` はキャプチャ済みデータ（`images`, `gt`, `meta.json`）のみ保持
- 生成・評価で増える成果物は `outputs/` にまとめる
