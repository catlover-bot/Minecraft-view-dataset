# 2種類の実験比較まとめ（baseline vs 強化プロンプト）

更新日: 2026-03-01

## 1. 「2種類」の定義

このプロジェクトで実施した2種類の実験は次です。

1. **baseline**
- 通常プロンプトで `description -> rebuild_plan -> rebuild_world -> evaluate`
- 参照例: `metrics_levels_openai_gpt_5_mini.json`, `metrics_levels_anthropic_claude_haiku_4_5_20251001.json`

2. **強化プロンプト（pe_v2）**
- `rebuild_plan` 側を強化:
  - 出力の構造化（footprint/floor/openings/roof など）
  - few-shot
  - critic-revise
- 参照例: `metrics_levels_pe_v2_*.json`

補足:
- さらに運用面として `pe_v2 + schema repair`（パーサ修復）も実施済みです。
- これは「第3の実験設定」というより、`pe_v2` の出力崩れを吸収する後段改善です。

## 2. 比較対象（再建築評価）

評価指標:
- IoU, F1（形状）
- material_match（厳密材質一致）
- all_levels_pass_rate（L0〜L4を全て満たす割合）

## 3. 結果（100建築平均）

### 3.1 baseline vs pe_v2（純粋比較）

| 条件 | baseline IoU | pe_v2 IoU | baseline F1 | pe_v2 F1 | baseline material | pe_v2 material |
|---|---:|---:|---:|---:|---:|---:|
| v1/OpenAI | 0.2872 | 0.2622 | 0.4418 | 0.4016 | 0.2679 | 0.2376 |
| v1/Claude | 0.3025 | 0.2939 | 0.4610 | 0.4493 | 0.2042 | 0.2865 |
| v4/OpenAI | 0.1522 | 0.1515 | 0.2604 | 0.2579 | 0.3246 | 0.3104 |
| v4/Claude | 0.1205 | 0.1528 | 0.2130 | 0.2611 | 0.2893 | 0.3223 |

見やすい要約:
- **v1系**: pe_v2 は形状（IoU/F1）が下がりやすい
- **v4系**: pe_v2 は形状・材質とも改善（特に Claude）

### 3.2 pe_v2 + schema repair（運用改善後）

| 条件 | pe_v2 repaired IoU | pe_v2 repaired F1 | pe_v2 repaired material | all_levels_pass_rate |
|---|---:|---:|---:|---:|
| v1/OpenAI | 0.2467 | 0.3793 | 0.1817 | 0.04 |
| v1/Claude | 0.2280 | 0.3661 | 0.1512 | 0.00 |
| v4/OpenAI | 0.1624 | 0.2726 | 0.1783 | 0.00 |
| v4/Claude | 0.1695 | 0.2854 | 0.1758 | 0.00 |

注意:
- 修復後は「fallback多発」を解消する代わりに、材質一致が下がったケースが多いです。
- つまり、安定性は改善したが最終品質（特に材質）はまだ改善余地があります。

## 4. description評価との関係

description 側はモデル別に固定で、実験タイプ（baseline / pe_v2）で大きく変えていません。  
したがって、この比較で主に違うのは **rebuild_plan の設計** です。

## 5. 図（日本語）

### IoU比較
![IoU比較](figures/two_types_iou_ja.svg)

### F1比較
![F1比較](figures/two_types_f1_ja.svg)

### 材質一致比較
![材質一致比較](figures/two_types_material_ja.svg)

### 全レベル合格率比較
![全レベル合格率比較](figures/two_types_all_levels_pass_ja.svg)

### ベースライン比差分
![差分比較](figures/two_types_delta_vs_baseline_ja.svg)

図データ:
- `reports/figures/two_types_data_2026-03-01.json`

## 6. 結論（短く）

1. **2種類の比較では、データセット難易度（v1/v4）で傾向が逆転**する。
2. `pe_v2` は v4 では効くが、v1では逆効果が出る条件がある。
3. `schema repair` はパイプライン安定化に効いたが、材質品質の改善は別課題。
