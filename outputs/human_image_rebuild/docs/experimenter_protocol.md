# Experimenter Protocol (Human Image->Rebuild Pilot)

## 推奨実施デザイン
- 小規模パイロット: 6-10名
- ケース数: 8ケース（v1=4, v4=4）
- デザイン: 参加者内比較（within-subject）
- セッション分割例:
  - Session A: image_only
  - Session B: image_plus_description
  - Session C (optional): image_plus_description_plus_structured_ir

## 推奨時間
- easy: 20分
- medium: 30分
- hard: 40分

## 実施手順
1. ケース配布: `case_packages/`
2. 提出回収: `submissions/`
3. 採点実行: `tools/score_human_image_rebuild_submissions.py`
4. 比較表更新: `reports/final/original_benchmark_human_image_rebuild_comparison_template.csv`

## 重要ガードレール
- 人間結果と既存ベンチ結果を混ぜない。
- プレースホルダ提出は必ず別ラベルで管理する。
