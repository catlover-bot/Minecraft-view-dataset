# Submission Format Specification (Minecraft-native)

## Primary format
提出ディレクトリ:
`outputs/human_minecraft_rebuild/submissions/<participant_id>/<condition>/<case_id>/`

必須ファイル:
- `structure.nbt`
- `submission_meta.json`

`structure.nbt` はStructure Blockエクスポート（Java Edition）を想定。

## Secondary format
- `structure.zip`（zip内に`.nbt`が1つ以上。最初に見つかった`.nbt`を使用）

## submission_meta.json 最低項目
- `participant_id`
- `case_id`
- `condition`
- `minecraft_version`
- `notes`

## 変換後の内部形式
採点前に以下へ変換:
- `bbox.json`
- `voxels.npy`（軸順Y,X,Z）
