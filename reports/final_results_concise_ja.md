# 最終結果まとめ　ざっくり

更新日: 2026-03-06

## 1. 何したか

- 対象: `buildings_100_v1` + `buildings_100_v4`（合計200建築）
- モデル: OpenAI + Claude（合計400条件）
- 主要改善:
  1. `rebuild_plan` の schema/パーサ強化（`parser_v6`）
  2. `v5` の材質整合ロジック + `self_refine tuned`

## 2. 結局

### A. fallback起因の失敗はほぼ解消

`parser_v6` で 400条件合算:
- 空operations (`accepted_zero`): **43件 -> 0件**
- `missing_or_none`: **81件 -> 0件**

つまり、**壊れたplanで止まる/空になる問題はほぼ潰せたぜ**。

### B. 再建築の最終精度（形状）は改善

全400条件平均（IoU/F1/material + 正配置率）:

| 設定 | IoU | F1 | material | correct_placement_rate | correct_placement_coverage |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.2156 | 0.3440 | 0.2715 | 0.1324 | 0.0731 |
| pe_v2 | 0.2151 | 0.3425 | 0.2892 | 0.1282 | 0.0918 |
| pe_v2 + parser_v6 | 0.2303 | 0.3664 | 0.2488 | 0.1022 | 0.0948 |
| v5 + tuned self_refine | **0.2456** | **0.3866** | 0.2426 | 0.0944 | **0.1122** |
Baseline比（最終設定）:
- IoU: **+0.0301**
- F1: **+0.0425**
- material: **-0.0289**（材質はまだトレードオフ）材質がマジで合わん。

## 3. 実験結果から

- **安定化（fallback削減）は達成**。
- **形状再現（IoU/F1）は明確に改善**。
- **材質一致は改善余地あり**（特にモデル差が大きい）。

## 4. 図

### 4.1 全体比較（400条件平均）
![final overview](figures/final_overview_ja.svg)

### 4.2 fallback削減（導入前後）
![fallback reduction](figures/parser_v6_fallback_rates_ja.svg)

図データ:
- `reports/figures/final_overview_data_2026-03-02.json`
- `reports/figures/parser_v6_data_2026-03-02.json`

- `reports/final_results_concise_ja.md`

詳細が必要なときだけ:
- `reports/two_experiment_types_summary_ja.md`
- `reports/statistical_validity_ablation_external_validity_ja.md`

### 6 実験の雑なまとめ
### 6.1 実験

- タスク: `画像 -> 説明文 -> 再建築plan -> ボクセル再建築 -> GT比較`
- データ: `buildings_100_v1` + `buildings_100_v4`（合計200建築）
- モデル: OpenAI / Claude（合計400条件）
- 比較した設定:
  1. Baseline
  2. pe_v2（強化プロンプト）
  3. pe_v2 + parser_v6（スキーマ/パーサ強化）
  4. v5 + tuned self_refine（最終）

### 6.2 どう評価したか

- 再建築評価:
  - `IoU`: 形状の重なり
  - `F1`: 形状の総合一致
  - `material_match`: 材質まで含む一致
- 安定性評価:
  - `accepted_zero`（空operations）
  - `missing_or_none`（plan欠損）
- 説明文評価（補助）:
  - `auto_score_mean`, `strict_material_f1`, `coarse_material_f1`, `dimension_score`

### 6.3 最終結果（図）

![final overview](figures/final_overview_ja.svg)

要点:
- Baseline比で最終設定は `IoU +0.0301`, `F1 +0.0425`
- fallback起因失敗は `accepted_zero 43 -> 0`, `missing_or_none 81 -> 0`
- materialは最終的に `-0.0289`（形状改善とのトレードオフ）

％で書くと:
- IoU: `21.56% -> 24.56%`（`+3.01pt`, 相対 `+13.94%`）
- F1: `34.40% -> 38.66%`（`+4.25pt`, 相対 `+12.36%`）
- material: `27.15% -> 24.26%`（`-2.89pt`, 相対 `-10.63%`）
- correct_placement_rate: `13.24% -> 9.44%`（`-3.80pt`）
- correct_placement_coverage: `7.31% -> 11.22%`（`+3.91pt`）
- accepted_zero率: `10.75% -> 0.00%`
- missing_or_none率: `20.25% -> 0.00%`

補足:
- `correct_placement_rate` は「置いたブロックのうち正解だった比率」。
- `correct_placement_coverage` は「GT全体に対して正しく再現できた比率」。
- 最終段では建物全体を多く再現するため `coverage` が伸び、同時に `rate` は下がることがあります。

### 6.4 最終設定の条件別結果（表）

| 条件 | IoU | F1 | material_match | material_match_relaxed_id | correct_placement_rate | correct_placement_rate_relaxed_id | correct_placement_coverage | correct_placement_coverage_relaxed_id |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 / OpenAI | 0.3030（30.30%） | 0.4588（45.88%） | 0.2247（22.47%） | 0.2553（25.53%） | 0.1014（10.14%） | 0.1135（11.35%） | 0.1420（14.20%） | 0.1652（16.52%） |
| v1 / Claude | 0.2793（27.93%） | 0.4293（42.93%） | 0.2094（20.94%） | 0.2172（21.72%） | 0.0887（8.87%） | 0.0922（9.22%） | 0.1009（10.09%） | 0.1044（10.44%） |
| v4 / OpenAI | 0.2045（20.45%） | 0.3348（33.48%） | 0.2952（29.52%） | 0.3092（30.92%） | 0.1047（10.47%） | 0.1101（11.01%） | 0.1178（11.78%） | 0.1233（12.33%） |
| v4 / Claude | 0.1957（19.57%） | 0.3234（32.34%） | 0.2413（24.13%） | 0.2468（24.68%） | 0.0825（8.25%） | 0.0843（8.43%） | 0.0882（8.82%） | 0.0901（9.01%） |
| 400条件平均 | 0.2456（24.56%） | 0.3866（38.66%） | 0.2426（24.26%） | 0.2571（25.71%） | 0.0944（9.44%） | 0.1000（10.00%） | 0.1122（11.22%） | 0.1207（12.07%） |

`relaxed_id` は、同義のブロックID揺れ（例: `stone brick` / `minecraft:stone_bricks`）を正規化して一致判定する新指標。
この再集計では、400条件平均で:
- `material_match`: `24.26% -> 25.71%`（`+1.45pt`）
- `correct_placement_rate`: `9.44% -> 10.00%`（`+0.57pt`）
- `correct_placement_coverage`: `11.22% -> 12.07%`（`+0.85pt`）

### 6.5 解釈

- 本実験の最大成果は、**壊れにくい再建築パイプラインを作れた**（fallbackほぼ解消）。
- その上で、**形状再現（IoU/F1）を一貫して引き上げた**。
- 残課題は **材質一致の改善** で、今後はモデル別に材質制約を最適化するのが有効。（無理ゲー？？）そもそもブロック数足りてない！

### 6.6 Description評価（%）

`description` の評価は 0〜1 スコアなので、以下は `%` 併記。個人的に％が好こ

各指標の意味:
- `auto_score_mean`:
  - 説明文全体の総合点（形・材質・寸法の情報）
- `strict_material_f1`:
  - 材質を厳密に一致判定（例: `stone_brick` と `stone` は別扱い）ここが低い、rebuildにもつながっている
- `coarse_material_f1`:
  - 材質を粗カテゴリで判定（例: どちらも STONE 系なら一致）
- `dimension_score`:
  - 幅・奥行き・高さなど、寸法情報の一致度

考察？:
- `coarse` が高く `strict` が低い場合:
  - 材質系統は当たっているが、ID/語彙が粗い（表記粒度が足りない）
- `dimension_score` が低い場合:
  - 形の大きさ説明が曖昧で、再建築時のスケール崩れを誘発しやすい
- `auto` が高くても再建築が高得点とは限らない:
  - 後段の `plan/render` でのスキーマ整合が崩れると最終IoU/F1は下がる

| 条件 | auto_score_mean | strict_material_f1 | coarse_material_f1 | dimension_score |
|---|---:|---:|---:|---:|
| v1 / OpenAI | 0.8102（81.02%） | 0.7269（72.69%） | 0.9138（91.38%） | 0.6547（65.47%） |
| v1 / Claude | 0.7202（72.02%） | 0.5714（57.14%） | 0.7295（72.95%） | 0.6654（66.54%） |
| v4 / OpenAI | 0.7520（75.20%） | 0.6146（61.46%） | 0.8658（86.58%） | 0.6047（60.47%） |
| v4 / Claude | 0.6893（68.93%） | 0.5707（57.07%） | 0.8089（80.89%） | 0.4634（46.34%） |

モデル平均:
- OpenAI平均: `auto 78.11%`, `strict 67.08%`, `coarse 88.98%`, `dimension 62.97%`
- Claude平均: `auto 70.47%`, `strict 57.11%`, `coarse 76.92%`, `dimension 56.44%`
- 全体平均: `auto 74.29%`, `strict 62.09%`, `coarse 82.95%`, `dimension 59.71%`

今回結果:
- description単体では OpenAI の方が高得点（特に `strict/coarse`）。
- ただし最終再建築は description だけでは決まらず、、、、、悲しい、、、
- description改善に加えて **parser強化 + self_refine** が必須だった。

補足:
- `description` は比較的高品質でも、最終IoU/F1は `plan/render` の整合に強く依存するため、  
  本研究では **parser強化とself_refineが最終品質を左右**した。

### 6.7 直近チューニングの最終採用（2026-03-08）

OpenAI経路で `candidate_diversification_high_risk_only` を比較:

- `ON`（高リスク時のみ多様化）: `diversify_on_s40_l10`
- `OFF`（常時多様化ON）: `diversify_all_s40_l10`

結果（all - on）:

- `buildings_100_v1`: IoU `-0.32pt`, F1 `-0.29pt`, correct_placement_rate `-1.06pt`, material `-1.84pt`
- `buildings_100_v4`: IoU `-0.40pt`, F1 `-0.62pt`, correct_placement_rate `-0.50pt`, material `-1.07pt`

結論:

- 常時多様化ONは悪化。
- 最終採用は `candidate_diversification_high_risk_only=ON`。
- 詳細: `reports/final_decision_2026-03-08_diversification_ja.md`
