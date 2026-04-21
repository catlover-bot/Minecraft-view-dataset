# Submission Format Specification

## Primary format (推奨)
必須ファイル:
- `bbox.json`
- `voxels.npy`

`voxels.npy` は軸順 `Y,X,Z`。ブロック名文字列配列。

## Secondary format
- `plan.json`

`plan.json` を提出した場合、採点側で `fill/carve/set` をレンダリングし `voxels.npy` に変換して評価します。

## Path convention
`outputs/human_image_rebuild/submissions/<participant_id>/<condition>/<case_id>/`
