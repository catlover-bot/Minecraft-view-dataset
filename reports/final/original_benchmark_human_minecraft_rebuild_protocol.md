# Original Benchmark Human Minecraft Rebuild Protocol

この文書は **実験実施インフラ** の定義です。人間成績は含みません。

## 目的
提示画像を見て、参加者がMinecraft内で建築を再現し、Minecraftネイティブ提出物（Structure Block `.nbt`）を提出する。

## 条件
- `image_only`
- `image_plus_description`
- `image_plus_description_plus_structured_ir`（任意）

## 参加者向け手順（要約）
1. `outputs/human_minecraft_rebuild/case_packages/<case_id>/source_images/` を参照。
2. 条件に応じて `condition_assets/description` / `condition_assets/structured_intermediate` を使用。
3. クリエイティブモードで、`build_constraints.json` のローカル座標サイズに合わせて再構築。
4. Structure Block で構造をエクスポートし、`structure.nbt` を提出。
5. `submission_meta.json` を同梱して提出。

## 提出先
`outputs/human_minecraft_rebuild/submissions/<participant_id>/<condition>/<case_id>/`

必須:
- `structure.nbt`
- `submission_meta.json`

任意:
- `structure.zip`（`structure.nbt` を含むzip）

## 評価
提出物は `bbox.json + voxels.npy` に変換後、既存LLM系と整合する同系列指標で採点します。
- IoU, F1
- material_match, coarse_material_match
- correct_placement_rate
- repair-effort（additions/deletions/replacements/edit_distance）

## 注意
- 本タスクはインフラ整備のみであり、人間性能の主張は行いません。
- 検証用プレースホルダ提出は研究結果に含めません。
