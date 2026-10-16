# Exact Oracleの材質・容量監査

POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC

E outcomes 120件、GT→固定cuboid仕様→参照fill→CPU実行の完全一致control 60建築。CPU controlをLLM成功へ加算しない。

Builder別の成功・失敗: {"anthropic": {"other": 0, "success": 46, "truncation": 14, "validation": 0}, "openai": {"other": 0, "success": 55, "truncation": 5, "validation": 0}}

独立再計算は元の占有最適平行移動を固定して行い、absolute/alignedのstrict値と照合した。材質IoUは座標と厳密材質IDの組の集合IoUであり、材質集合F1とは異なる。元メトリクス、配置、原本応答を変更していない。

occupancy IoU >=0.99 かつ exact-material IoU <0.5 の固定診断条件に該当: 8 outcomes。全例を保存し、この条件は母集団推定や因果的な閾値ではない。

building034 / anthropic: {"aligned": {"add_count": 0, "add_normalized_cost": 0.0, "delete_count": 0, "delete_normalized_cost": 0.0, "dimension_mae": 0.0, "gt_dimensions_xyz": [56, 20, 36], "gt_voxel_count": 4912, "material_aware_iou": 1.0, "material_match_count": 4912, "material_set_f1": 1.0, "occupancy_f1": 1.0, "occupancy_intersection": 4912, "occupancy_iou": 1.0, "occupancy_union": 4912, "pred_dimensions_xyz": [56, 20, 36], "pred_voxel_count": 4912, "replace_count": 0, "replace_normalized_cost": 0.0, "total_normalized_repair_cost": 0.0}, "lexical_sensitivity": {"absolute": {"lexical_material_iou_delta": 0.0, "material_aware_iou": 1.0, "material_match_count": 4912, "material_set_f1": 1.0, "strict_material_aware_iou": 1.0}, "aligned": {"lexical_material_iou_delta": 0.0, "material_aware_iou": 1.0, "material_match_count": 4912, "material_set_f1": 1.0, "strict_material_aware_iou": 1.0}}, "status": "success"}
building034 / openai: {"aligned": {"add_count": 0, "add_normalized_cost": 0.0, "delete_count": 0, "delete_normalized_cost": 0.0, "dimension_mae": 0.0, "gt_dimensions_xyz": [56, 20, 36], "gt_voxel_count": 4912, "material_aware_iou": 0.22341220423412203, "material_match_count": 1794, "material_set_f1": 0.9090909090909091, "occupancy_f1": 1.0, "occupancy_intersection": 4912, "occupancy_iou": 1.0, "occupancy_union": 4912, "pred_dimensions_xyz": [56, 20, 36], "pred_voxel_count": 4912, "replace_count": 3118, "replace_normalized_cost": 0.634771986970684, "total_normalized_repair_cost": 0.634771986970684}, "lexical_sensitivity": {"absolute": {"lexical_material_iou_delta": 0.0, "material_aware_iou": 0.22341220423412203, "material_match_count": 1794, "material_set_f1": 0.9090909090909091, "strict_material_aware_iou": 0.22341220423412203}, "aligned": {"lexical_material_iou_delta": 0.0, "material_aware_iou": 0.22341220423412203, "material_match_count": 1794, "material_set_f1": 0.9090909090909091, "strict_material_aware_iou": 0.22341220423412203}}, "status": "success"}

転記診断はE paletteと参照cuboidの境界・出力operation材質を照合する。異なる正しい分割も可能なため、境界不一致だけを欠落とは呼ばない。最終voxel confusionを優先する。

alias感度は歴史的DIRECT_TYPE_NORMから計算前に固定した狭い語彙規則のみ。物理的同値の検証ではなく、stonebrickとbrick/brick_blockを同一視しない。slab/fence/plank/glassの広いまとめを使用しない。strict結果は置換せず別欄へ保存した。

容量欄は実際に保存されたprovider token usage、元planのoutput上限、入力byte、occupied数、cuboid数、同材質Z-run数、参照operation数・byteを記載する。byteをtokenへ換算せず、参照列がtoken上限へ入るかは未認証。理由tokenが判明する場合は別欄に残す。未完了出力の正しさを外挿しない。

Eは完全離散仕様から操作への診断であり自然言語の母集団上限ではない。成功だけの値は選択条件付きである。研究API・モデル・token-count・キー照会は0件。
