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
  - `logs/`
- `metrics/description/`
- `metrics/rebuild/`
- `logs/`（データセット単位ログ）

方針:
- `datasets/` はキャプチャ済みデータ（`images`, `gt`, `meta.json`）のみ保持
- 生成・評価で増える成果物は `outputs/` にまとめる

