# Original Benchmark Human Minecraft Rebuild Setup Summary

この文書は人間実験実施基盤のまとめです（人間成績は未収集）。

## Scope
- datasets: `buildings_100_v1`, `buildings_100_v4`
- selected cases: `8` (easy=2, medium=3, hard=3)
- conditions: image_only / image+description / image+description+structured_ir

## Output namespace
- `outputs/human_minecraft_rebuild/`
- `reports/final/original_benchmark_human_minecraft_rebuild_*`

## Notes
- 既存ベンチ結果を上書きしない分離運用。
- Minecraftネイティブ提出物（structure.nbt）を評価用voxelへ変換して採点。
