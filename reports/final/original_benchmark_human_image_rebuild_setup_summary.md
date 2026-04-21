# Original Benchmark Human Image->Rebuild Pilot Setup Summary

この文書は人手実験の**実施基盤**のまとめです。人間成績の報告は含みません。

## Scope
- datasets: `buildings_100_v1`, `buildings_100_v4`
- selected cases: `8` (easy=2, medium=3, hard=3)
- conditions: image_only / image+description / image+description+structured_ir

## Output namespace
- `outputs/human_image_rebuild/`
- `reports/final/original_benchmark_human_image_rebuild_*`

## Notes
- 既存Main/Supplementaryベンチ結果は上書きしていません。
- 提出スコアはLLM評価と整合する同系指標で計算します。

