# Datasets Directory Guide

主な実験データセット:

- `buildings_100_v1/`
- `buildings_100_v4/`

各データセットの基本構成:

- `building_000/ ... building_099/`: 建築ごとの生データ
  - `images/`
  - `gt/`
  - `meta.json`

注:
- 実験出力（`description*`, `rebuild_plan*`, `rebuild_world*`, `logs`, `metrics`）は
  `outputs/i2t2b/<dataset_name>/` に移動済みです。
- 今後の整理には `scripts/relocate_i2t2b_outputs.sh` を使います。
