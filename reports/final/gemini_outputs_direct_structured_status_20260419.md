# Gemini 実験ステータス (outputs基準)

- 本ファイルは `outputs/i2t2b` に保存された Gemini 実験の進捗集約です。
- 現時点: v1 は direct / structured / edit評価まで完了、v4 は実行中。

## v1 Direct
- IoU: 0.2610
- F1: 0.4085
- material_match: 0.2873
- correct_placement_rate: 0.1357
- edit_distance_over_gt: 1.4515

## v1 Structured (中間表現あり)
- IoU: 0.2755
- F1: 0.4253
- material_match: 0.4197
- correct_placement_rate: 0.1868
- edit_distance_over_gt: 1.4839

## 差分 (Structured - Direct)
- ΔIoU: +0.0146
- ΔF1: +0.0167
- Δmaterial_match: +0.1324
- Δcorrect_placement_rate: +0.0511
- Δedit_distance_over_gt: +0.0324
