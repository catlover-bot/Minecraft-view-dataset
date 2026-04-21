# Experimenter Protocol (Human Minecraft Rebuild Pilot)

## 推奨デザイン
- 参加者: 6-10名（パイロット）
- ケース: 8ケース（v1=4, v4=4）
- デザイン: within-subject（条件順はカウンターバランス）

## 推奨時間
- easy: 20分
- medium: 30分
- hard: 40分

## 実施手順
1. ケース配布: `outputs/human_minecraft_rebuild/case_packages/`
2. 提出回収: `outputs/human_minecraft_rebuild/submissions/`
3. 変換: `tools/convert_human_minecraft_submissions.py`
4. 採点: `tools/score_human_image_rebuild_submissions.py`（変換出力を入力）

## ガードレール
- 既存ベンチ（Main/Supplementary/Execution-gap）とは別管理。
- placeholder結果を人間結果として扱わない。
