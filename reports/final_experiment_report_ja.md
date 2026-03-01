# Minecraft I2T2B 実験 最終報告

更新日: 2026-03-01

---

## 1. 研究目的

本実験の目的は、以下の一連パイプラインの有効性を検証することです。

1. 建築画像（multi-view）から説明文を生成（image-to-text）
2. 説明文から再建築計画を生成（text-to-build-plan）
3. 計画を voxel にレンダリングし GT と比較（build-to-eval）

評価対象は主に 2つです。
- **Description品質**（説明文がGTをどれだけ表現できているか）
- **Rebuild品質**（最終再建築がGTにどれだけ一致しているか）

---

## 2. 実験条件

### 2.1 データセット
- `datasets/buildings_100_v1`（100建築）
- `datasets/buildings_100_v4`（100建築）

### 2.2 モデル
- OpenAI: `gpt-5-mini`
- Anthropic: `claude-haiku-4-5-20251001`

### 2.3 比較した実験タイプ

1. **baseline**
- 通常プロンプトで `description -> rebuild_plan -> rebuild_world -> evaluate`

2. **強化プロンプト（pe_v2）**
- rebuild_plan 側で構造化出力 + few-shot + critic-revise を導入

補足（運用改善）:
- **pe_v2 + schema repair**
  - 出力スキーマ崩れを吸収する自動修復パーサを適用
  - fallback多発問題を解消

---

## 3. 評価方法（要点）

### 3.1 Description評価
- `strict_material_f1`: 材質IDの厳密一致
- `coarse_material_f1`: 粗カテゴリ一致（STONE/WOOD/BRICK等）
- `dimension_score`: bbox由来寸法との近さ
- `completeness_score`: summary/materials/dimensions/elements/hints の充足
- `auto_score`: 上記の重み付き総合

### 3.2 Rebuild評価
- `IoU`, `F1`: 形状一致（非air occupancy）
- `material_match`: 重なり座標上の厳密材質一致
- `coarse_material_match`: 重なり座標上の粗材質一致
- `component_f1`: 2D連結成分レベルの構造一致
- レベル判定（L0〜L4）と `all_levels_pass`

詳細解説:
- `reports/rebuild_metrics_guide_ja.md`

---

## 4. Description評価結果（平均）

| 条件 | auto_score | strict_material_f1 | coarse_material_f1 | dimension_score | completeness |
|---|---:|---:|---:|---:|---:|
| v1/OpenAI | 0.8102 | 0.7269 | 0.9138 | 0.6547 | 1.0000 |
| v1/Claude | 0.7202 | 0.5714 | 0.7295 | 0.6654 | 1.0000 |
| v4/OpenAI | 0.7520 | 0.6146 | 0.8658 | 0.6047 | 1.0000 |
| v4/Claude | 0.6893 | 0.5707 | 0.8089 | 0.4634 | 1.0000 |

解釈:
- 4条件すべてで必須項目の充足は達成（completeness=1.0）。
- 材質表現は OpenAI 側が全体的に高め。
- 寸法推定は v1/Claude が比較的良いが、v4/Claude は低下。

---

## 5. Rebuild結果（2種類比較）

### 5.1 baseline vs pe_v2

| 条件 | baseline IoU | pe_v2 IoU | baseline F1 | pe_v2 F1 | baseline material | pe_v2 material |
|---|---:|---:|---:|---:|---:|---:|
| v1/OpenAI | 0.2872 | 0.2622 | 0.4418 | 0.4016 | 0.2679 | 0.2376 |
| v1/Claude | 0.3025 | 0.2939 | 0.4610 | 0.4493 | 0.2042 | 0.2865 |
| v4/OpenAI | 0.1522 | 0.1515 | 0.2604 | 0.2579 | 0.3246 | 0.3104 |
| v4/Claude | 0.1205 | 0.1528 | 0.2130 | 0.2611 | 0.2893 | 0.3223 |

要点:
- v1では pe_v2 が形状を落とす傾向。
- v4では pe_v2 が改善（特に Claude）。

### 5.2 pe_v2 + schema repair（最新運用）

| 条件 | IoU | F1 | material_match | all_levels_pass_rate |
|---|---:|---:|---:|---:|
| v1/OpenAI | 0.2467 | 0.3793 | 0.1817 | 0.04 |
| v1/Claude | 0.2280 | 0.3661 | 0.1512 | 0.00 |
| v4/OpenAI | 0.1624 | 0.2726 | 0.1783 | 0.00 |
| v4/Claude | 0.1695 | 0.2854 | 0.1758 | 0.00 |

要点:
- schema repair によりパイプラインは安定化したが、材質一致は低下傾向。
- all-level pass は v1/OpenAI の 4% のみ。

---

## 6. 安定性改善（fallback問題）

`pe_v2` に対する schema repair 適用後:
- v1/OpenAI: fallback `0/100`（修復 64, draft救済 7）
- v1/Claude: fallback `0/100`（修復 99）
- v4/OpenAI: fallback `0/100`（修復 95, draft救済 35）
- v4/Claude: fallback `0/100`（修復 100, draft救済 1）

結論:
- **壊れやすさ（フォーマット不一致）問題はほぼ解消**。
- 現在の主課題は「計画の意味内容（特に材質と形状）」。

---

## 7. なぜ精度が伸びきらないか（原因仮説）

1. Description品質とRebuild品質のギャップ
- description は高得点でも、plan への変換で意味が失われる。

2. 材質一致の劣化
- schema repair で op を救済する際、材質が一般化/欠落して `material_match` が下がる。

3. v1/v4で最適戦略が異なる
- 同一プロンプトで両方最適化しきれていない（domain shift）。

4. レベル閾値の厳しさと偏り
- `all_levels_pass` は L1/L3/L4 の同時達成が必要で難しい。

---

## 8. 精度向上のための優先アクション

以下は効果順（推奨）です。

### A. Rebuild計画の「材質制約」を明示強化（最優先）

狙い: `material_match` の底上げ  
方法:
- plan 出力に `material_budget`（材質比率の目標）を必須化
- 「外壁/屋根/窓/床」の材質を roleごとに固定して自己検証
- critic で `role-material consistency` を自動チェック

期待効果:
- 材質一致（L3）が最も改善しやすい

### B. Plan-Renderer 間の中間表現を厳格化

狙い: semantic loss を減らす  
方法:
- `plan_schema_version=v3` を導入し、opテンプレを限定
- `outline/windows_row` など高水準opは、LLM出力時点で展開させる
- 修復パーサ側は「救済」より「検証+再生成要求」を優先

期待効果:
- IoU/F1 の下振れを抑制

### C. 2段階生成（Structure-first -> Material-fill）

狙い: 形状と材質を分離最適化  
方法:
1. 形状専用plan（air/stone仮材質で形だけ）
2. 材質塗り替えplan（role別に置換）

期待効果:
- L1（形状）とL3（材質）のトレードオフ緩和

### D. Dataset別プロンプト最適化（v1/v4分離）

狙い: domain shift 対策  
方法:
- v1向け・v4向けに few-shot を分離
- 自動判別器で profile 切替

期待効果:
- pe_v2が v1 で悪化する問題の緩和

### E. 評価ループを学習的に回す

狙い: 改善サイクルの高速化  
方法:
- 失敗事例Top-K（低 IoU or 低 material）を抽出
- 失敗パターン別テンプレ（窓過剰、屋根欠損、材質崩れ）を追加
- 週次で再評価ダッシュボード更新

---

## 9. 次の実験計画（具体）

1. `pe_v3` プロンプトを作成
- role-material consistency + material budget + strict schema

2. 20建築のスモールAB
- `pe_v2` vs `pe_v3`（v1/v4 各10）
- 指標: IoU/F1/material_match, all_levels_pass

3. 良ければ100建築へ全量展開

---

## 10. 関連資料

- 総合サマリ: `reports/experiment_summary_2026-03-01.md`
- 再建築評価ガイド: `reports/rebuild_metrics_guide_ja.md`
- 2種類比較レポート: `reports/two_experiment_types_summary_ja.md`
- 図（日本語）:
  - `reports/figures/two_types_iou_ja.svg`
  - `reports/figures/two_types_f1_ja.svg`
  - `reports/figures/two_types_material_ja.svg`
  - `reports/figures/two_types_all_levels_pass_ja.svg`
  - `reports/figures/two_types_delta_vs_baseline_ja.svg`
