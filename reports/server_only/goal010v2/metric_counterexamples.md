# 新指標の反証fixture

以下は凍結前に作成した合成例であり、実建築の観測結果ではない。実行値と入力hashはfixture_counterexamples.jsonに保存した。

| 合成例 | frame | occupancy IoU | Precision | Recall | surface F1@1 | footprint IoU | conditional材質正解率 |
|---|---|---:|---:|---:|---:|---:|---:|
| identical | absolute | 1 | 1 | 1 | 1 | 1 | 1 |
| identical | aligned | 1 | 1 | 1 | 1 | 1 | 1 |
| same_occupancy_wrong_material | absolute | 1 | 1 | 1 | 1 | 1 | 0 |
| same_occupancy_wrong_material | aligned | 1 | 1 | 1 | 1 | 1 | 0 |
| translated_point | absolute | 0 | 0 | 0 | 1 | 0 | NULL |
| translated_point | aligned | 1 | 1 | 1 | 1 | 1 | 1 |
| shell_GT_filled_prediction | absolute | 0.962963 | 0.962963 | 1 | 1 | 1 | 1 |
| shell_GT_filled_prediction | aligned | 0.962963 | 0.962963 | 1 | 1 | 1 | 1 |
| normal_empty_prediction | absolute | 0 | NULL | 0 | 0 | 0 | NULL |
| normal_empty_prediction | aligned | 0 | NULL | 0 | 0 | 0 | NULL |
| negative_floor_anchor | absolute | 0 | 0 | 0 | 1 | 0 | NULL |
| negative_floor_anchor | aligned | 1 | 1 | 1 | 1 | 1 | 1 |

同一占有でも全材質を誤れば条件付き材質正解率は0。1ブロックずれた点はabsolute IoU=0でもsurface F1@1=1であり、位置一致と近傍許容は異なる。3×3×3 shell GTを充填するとoccupancy IoU=26/27だがsurfaceとfootprintは1なので、表面・投影だけでは内部の過剰生成を検知できない。正常emptyのprecisionやC=0のconditional材質はNULLを維持する。

閾値2、sqrt2、幅1/2、連結成分、union列分母、負floor、固定shift不変、hiddenGT入力境界、source/NULL/failure0/networkの反証は新旧を合わせた322テストで検証した。実際の建築での指標依存の順序は、全数計算後のmetric_rank_reversals表に別途記録する。
