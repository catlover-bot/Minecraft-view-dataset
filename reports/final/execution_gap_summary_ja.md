# Execution Gap まとめ（Renderer上限 vs Agent実運用）

更新日: 2026-03-11-real-full

今回は `rebuild_world_*` をRenderer上限、`rebuild_world_agentexec_*` をAgent実運用として比較しました。
※ 今回の `agentexec` は、Malmo上で `chat /setblock` / `chat /fill` を実行した real placement です。

## 全体（4条件平均）

- IoU: Renderer `24.56%` -> Agent `24.42%`（gap `0.14%`）
- F1: Renderer `38.66%` -> Agent `38.39%`（gap `0.27%`）
- material_match: Renderer `24.26%` -> Agent `24.92%`（gap `-0.65%`）
- correct_placement_rate: Renderer `9.44%` -> Agent `9.93%`（gap `-0.49%`）

## 条件別

| 条件 | Renderer IoU | Agent IoU | IoU保持率 | Renderer F1 | Agent F1 | F1保持率 | Renderer材質 | Agent材質 | Renderer配置率 | Agent配置率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1/OpenAI | 30.30% | 30.52% | 100.73% | 45.88% | 46.08% | 100.45% | 22.47% | 23.90% | 10.14% | 10.99% |
| v1/Claude | 27.93% | 28.21% | 101.00% | 42.93% | 43.28% | 100.82% | 20.94% | 22.20% | 8.87% | 9.76% |
| v4/OpenAI | 20.45% | 19.68% | 96.25% | 33.48% | 32.33% | 96.56% | 29.52% | 29.43% | 10.47% | 10.65% |
| v4/Claude | 19.57% | 19.27% | 98.44% | 32.34% | 31.86% | 98.53% | 24.13% | 24.13% | 8.25% | 8.32% |

## 図

- `reports/figures/execution_gap_iou_f1_ja.svg`
- `reports/figures/execution_gap_material_placement_ja.svg`
- `reports/figures/execution_gap_retention_ja.svg`
- `reports/figures/execution_gap_absolute_ja.svg`

## 元データ

- `reports/figures/execution_gap_data_2026-03-11-real-full.json`
