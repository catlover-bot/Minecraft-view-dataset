# Original Benchmark Structured-Intermediate Summary

この結果は original benchmark (`buildings_100_v1/v4`) に structured-intermediate 条件を追加した補助分析です。
既存 Main/Supplementary の published direct 結果は上書きせず、比較行のみ追加しています。

## Coverage
- direct conditions: 8
- structured conditions: 4
- comparisons: 4

## Main: Direct vs Structured
- claude v1: ΔIoU +0.0701, ΔF1 +0.0852, Δmaterial +0.2006, Δcorrect +0.1606, Δedit -0.3735
- claude v4: ΔIoU -0.0310, ΔF1 -0.0462, Δmaterial +0.1015, Δcorrect +0.0789, Δedit -0.3042
- openai v1: ΔIoU +0.0381, ΔF1 +0.0433, Δmaterial +0.1287, Δcorrect +0.1132, Δedit -0.5893
- openai v4: ΔIoU -0.0288, ΔF1 -0.0414, Δmaterial +0.0617, Δcorrect +0.0077, Δedit -0.0053

## all_200 (main only)
- openai: direct IoU 23.75% -> structured 24.22% (Δ +0.46pt), direct edit 1.595 -> structured 1.298 (Δ -0.297)
- claude: direct IoU 20.55% -> structured 22.50% (Δ +1.95pt), direct edit 1.492 -> structured 1.153 (Δ -0.339)

## Guardrails
- repair-effort は IoU/F1 の置換ではなく追加診断です。
- Main と Supplementary は分離して解釈してください。
- `llm_authored_10` は本集計に含めていません。
