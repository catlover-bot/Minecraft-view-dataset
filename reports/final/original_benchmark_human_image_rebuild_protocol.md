# Original Benchmark Human Image->Rebuild Pilot Protocol

この文書は **実験実施用プロトコル** です。ここには人間被験者の結果は含みません。

## 目的
画像からMinecraft建築を再構成し、GTと比較可能な形式で提出してもらうための小規模パイロットを実施する。

## 条件
- `image_only`
- `image_plus_description`
- `image_plus_description_plus_structured_ir`（任意）

## 参加者向け手順
1. `outputs/human_image_rebuild/case_packages/<case_id>/source_images/` を見る。
2. 条件に応じて `condition_assets/description/` と `condition_assets/structured_intermediate/` を使う。
3. 指定の許可ブロック・制約内で再構成する。
4. 提出先: `outputs/human_image_rebuild/submissions/<participant_id>/<condition>/<case_id>/`

## 提出形式
Primary:
- `bbox.json`
- `voxels.npy`

Secondary:
- `plan.json`（採点時に `voxels.npy` へ変換して評価）

## 評価指標
LLM評価と整合する形で次を算出:
- IoU, F1
- material_match, coarse_material_match
- correct_placement_rate
- repair-effort（additions/deletions/replacements/edit_distance）

## 注意
- このパイロット構築タスクでは人間成績を主張しない。
- プレースホルダ提出は配線確認専用で、研究結果に含めない。
