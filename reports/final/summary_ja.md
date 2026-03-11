# 最終結果まとめ（日本語・口語版）

更新日: 2026-03-06

## 0. 2026-03-10 追記（Main結果の固定版）

研究整理のため、Main結果を「共通ハイパラ固定」に切り分けて再集計しました。

- Main（共通ハイパラ固定）:
  - `reports/final/main_shared_hparams_results_2026-03-10.md`
  - `reports/final/main_shared_hparams_results_2026-03-10.json`
- Supplementary（モデル別最適化）は、従来どおり各 `outputs/i2t2b/.../metrics/rebuild/*.json` を参照

## 1. 何をやったか

今回やったのは、次の一連の流れです。

- `画像 -> 説明文 -> 再建築plan -> ボクセル再建築 -> GT比較`
- データは `buildings_100_v1` と `buildings_100_v4`（合計200建築）
- モデルは OpenAI / Claude（合計400条件）
- 比較した設定は4つ
  1. Baseline
  2. pe_v2（強化プロンプト）
  3. pe_v2 + parser_v6（スキーマ/パーサ強化）
  4. v5 + tuned self_refine（最終設定）

## 2. 結果どうだったか

### 2.1 まず、fallback由来の失敗はほぼ消えた

`parser_v6` を入れたあと、400条件合算でこうなりました。

- 空operations（`accepted_zero`）: `43 -> 0`
- 欠損（`missing_or_none`）: `81 -> 0`

要するに、フォーマット崩れで止まる問題はかなり解消できています。

### 2.2 再建築精度（400条件平均）

| 設定 | IoU | F1 | material | correct_placement_rate | correct_placement_coverage |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.2156 | 0.3440 | 0.2715 | 0.1324 | 0.0731 |
| pe_v2 | 0.2151 | 0.3425 | 0.2892 | 0.1282 | 0.0918 |
| pe_v2 + parser_v6 | 0.2303 | 0.3664 | 0.2488 | 0.1022 | 0.0948 |
| v5 + tuned self_refine | **0.2456** | **0.3866** | 0.2426 | 0.0944 | **0.1122** |

Baseline比（最終設定）:
- IoU: `+0.0301`
- F1: `+0.0425`
- material: `-0.0289`

%で見ると（Baseline -> 最終設定）:
- IoU: `21.56% -> 24.56%`（`+3.01pt`, 相対 `+13.94%`）
- F1: `34.40% -> 38.66%`（`+4.25pt`, 相対 `+12.36%`）
- material: `27.15% -> 24.26%`（`-2.89pt`, 相対 `-10.63%`）
- correct_placement_rate: `13.24% -> 9.44%`（`-3.80pt`）
- correct_placement_coverage: `7.31% -> 11.22%`（`+3.91pt`）
- accepted_zero率: `10.75% -> 0.00%`
- missing_or_none率: `20.25% -> 0.00%`

## 3. ここがポイント

- 安定性は上がった: fallback起因の失敗を大きく減らせた。
- 形状は良くなった: IoU/F1 は継続して改善。
- 材質はまだ課題: 形状改善と引き換えで material が落ちる傾向。
- 配置系は挙動が変わった: `rate` は下がる一方で `coverage` は上がっていて、
  「少なく正確に置く」から「広く置くが誤配置も増える」方向になっています。

## 4. 図

### 4.1 全体比較（400条件平均）
![final overview](../figures/final_overview_ja.svg)

### 4.2 fallback削減（導入前後）
![fallback reduction](../figures/parser_v6_fallback_rates_ja.svg)

図データ:
- `reports/figures/final_overview_data_2026-03-02.json`
- `reports/figures/parser_v6_data_2026-03-02.json`

## 5. 条件別結果（最終設定）

| 条件 | IoU | F1 | material_match | material_match_relaxed_id | correct_placement_rate | correct_placement_rate_relaxed_id | correct_placement_coverage | correct_placement_coverage_relaxed_id |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 / OpenAI | 0.3030（30.30%） | 0.4588（45.88%） | 0.2247（22.47%） | 0.2553（25.53%） | 0.1014（10.14%） | 0.1135（11.35%） | 0.1420（14.20%） | 0.1652（16.52%） |
| v1 / Claude | 0.2793（27.93%） | 0.4293（42.93%） | 0.2094（20.94%） | 0.2172（21.72%） | 0.0887（8.87%） | 0.0922（9.22%） | 0.1009（10.09%） | 0.1044（10.44%） |
| v4 / OpenAI | 0.2045（20.45%） | 0.3348（33.48%） | 0.2952（29.52%） | 0.3092（30.92%） | 0.1047（10.47%） | 0.1101（11.01%） | 0.1178（11.78%） | 0.1233（12.33%） |
| v4 / Claude | 0.1957（19.57%） | 0.3234（32.34%） | 0.2413（24.13%） | 0.2468（24.68%） | 0.0825（8.25%） | 0.0843（8.43%） | 0.0882（8.82%） | 0.0901（9.01%） |
| 400条件平均 | 0.2456（24.56%） | 0.3866（38.66%） | 0.2426（24.26%） | 0.2571（25.71%） | 0.0944（9.44%） | 0.1000（10.00%） | 0.1122（11.22%） | 0.1207（12.07%） |

`relaxed_id` は同義IDの揺れ（例: `stone brick` / `minecraft:stone_bricks`）を正規化して判定する指標です。

400条件平均での差分:
- `material_match`: `24.26% -> 25.71%`（`+1.45pt`）
- `correct_placement_rate`: `9.44% -> 10.00%`（`+0.57pt`）
- `correct_placement_coverage`: `11.22% -> 12.07%`（`+0.85pt`）

## 6. Description評価（%）

Descriptionの評価は「説明文としてどれだけ情報を落とさず書けているか」を見るものです。

使っている指標:
- `auto_score_mean`: 説明文全体の総合点
- `strict_material_f1`: 材質を厳密IDで評価
- `coarse_material_f1`: 材質を粗カテゴリで評価
- `dimension_score`: 幅・奥行き・高さの整合度

| 条件 | auto_score_mean | strict_material_f1 | coarse_material_f1 | dimension_score |
|---|---:|---:|---:|---:|
| v1 / OpenAI | 0.8102（81.02%） | 0.7269（72.69%） | 0.9138（91.38%） | 0.6547（65.47%） |
| v1 / Claude | 0.7202（72.02%） | 0.5714（57.14%） | 0.7295（72.95%） | 0.6654（66.54%） |
| v4 / OpenAI | 0.7520（75.20%） | 0.6146（61.46%） | 0.8658（86.58%） | 0.6047（60.47%） |
| v4 / Claude | 0.6893（68.93%） | 0.5707（57.07%） | 0.8089（80.89%） | 0.4634（46.34%） |

平均:
- OpenAI: `auto 78.11%`, `strict 67.08%`, `coarse 88.98%`, `dimension 62.97%`
- Claude: `auto 70.47%`, `strict 57.11%`, `coarse 76.92%`, `dimension 56.44%`
- 全体: `auto 74.29%`, `strict 62.09%`, `coarse 82.95%`, `dimension 59.71%`

見方としては、Descriptionが高くても最終Rebuildが高いとは限りません。
`plan/render` 側で情報が崩れると、最終IoU/F1は落ちます。

## 7. 直近チューニングの判断（2026-03-08）

OpenAI経路で候補多様化を比較しました。

- `ON`（高リスク時のみ多様化）: `diversify_on_s40_l10`
- `OFF`（常時多様化ON）: `diversify_all_s40_l10`

差分（all - on）:
- `buildings_100_v1`: IoU `-0.32pt`, F1 `-0.29pt`, correct_placement_rate `-1.06pt`, material `-1.84pt`
- `buildings_100_v4`: IoU `-0.40pt`, F1 `-0.62pt`, correct_placement_rate `-0.50pt`, material `-1.07pt`

最終判断:
- 常時多様化ONは採用しない
- `candidate_diversification_high_risk_only=ON` を採用
- 詳細: `reports/final/decision_diversification_ja.md`
