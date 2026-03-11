# Main結果（共通ハイパラ固定）

更新日: 2026-03-10

この結果は `OpenAI/Claudeで同一ハイパラ` だけを使った Main 評価です。

## 実行ルール

- Main: 共通ハイパラのみ（モデル別分岐なし）
- Supplementary: モデル別最適化は参考値として別管理
- 探索は dev のみ / test は最終1回
- 探索回数・探索範囲はモデル間で同一

## Main結果（再建築）

| Model | Dataset | IoU | F1 | Material | Coarse Material | Correct Placement | Correct Placement (relaxed) | Component F1 | Missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenAI | v1 (100) | 29.44% | 44.75% | 28.96% | 45.71% | 13.96% | 15.96% | 98.00% | 0 |
| OpenAI | v4 (100) | 18.06% | 30.16% | 30.60% | 44.68% | 12.39% | 13.18% | 5.07% | 0 |
| **OpenAI** | **all (200)** | **23.75%** | **37.46%** | **29.78%** | **45.19%** | **13.18%** | **14.57%** | **51.53%** | **0** |
| Claude | v1 (100) | 24.29% | 38.48% | 17.79% | 31.52% | 7.97% | 8.38% | 66.67% | 0 |
| Claude | v4 (100) | 16.81% | 28.42% | 23.53% | 36.15% | 8.62% | 8.79% | 4.15% | 0 |
| **Claude** | **all (200)** | **20.55%** | **33.45%** | **20.66%** | **33.83%** | **8.29%** | **8.59%** | **35.41%** | **0** |

## Main評価ファイル

- OpenAI v1: `outputs/i2t2b/buildings_100_v1/metrics/rebuild/schema_v5_repair_openai_self_refine_common_v8_struct_full.json`
- OpenAI v4: `outputs/i2t2b/buildings_100_v4/metrics/rebuild/schema_v5_repair_openai_self_refine_common_v8_struct_full.json`
- Claude v1: `outputs/i2t2b/buildings_100_v1/metrics/rebuild/schema_v5_repair_claude_self_refine_common_v8_struct_full.json`
- Claude v4: `outputs/i2t2b/buildings_100_v4/metrics/rebuild/schema_v5_repair_claude_self_refine_common_v8_struct_full.json`

## Supplementary（モデル別最適化）

モデル別最適化の結果は、Mainとは切り分けて既存ファイルを参照します。

- `outputs/i2t2b/buildings_100_v1/metrics/rebuild/schema_v5_repair_openai_self_refine_tuned.json`
- `outputs/i2t2b/buildings_100_v1/metrics/rebuild/schema_v5_repair_claude_self_refine_tuned_conditional_precboost_full.json`
- `outputs/i2t2b/buildings_100_v4/metrics/rebuild/schema_v5_repair_openai_self_refine_tuned.json`
- `outputs/i2t2b/buildings_100_v4/metrics/rebuild/schema_v5_repair_claude_self_refine_tuned_conditional_precboost_full.json`

## Main と Supplementary のざっくり比較（all 200）

| Model | Main IoU | Supp IoU | Main F1 | Supp F1 | Main Material | Supp Material | Main Correct Placement | Supp Correct Placement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenAI | 23.75% | 25.37% | 37.46% | 39.68% | 29.78% | 25.99% | 13.18% | 10.31% |
| Claude | 20.55% | 22.28% | 33.45% | 35.75% | 20.66% | 21.55% | 8.29% | 8.40% |

Mainは「モデル間の公平性」を優先した値、Supplementaryは「モデル別に詰めたときの上限寄り」の参考値です。
