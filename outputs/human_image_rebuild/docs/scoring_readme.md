# Scoring README

## Manifest
`/Users/hirotaka-m/Minecraft-view-dataset/reports/final/original_benchmark_human_image_rebuild_cases.json`

## Main command
```bash
python3 tools/score_human_image_rebuild_submissions.py \
  --cases_manifest /Users/hirotaka-m/Minecraft-view-dataset/reports/final/original_benchmark_human_image_rebuild_cases.json \
  --submission_root outputs/human_image_rebuild/submissions \
  --out_root outputs/human_image_rebuild/scored_submissions
```

## Outputs
- `human_scores.json`
- `human_scores.csv`
- `human_scores_summary.md`
- `human_vs_llm_case_table.csv`

※ このREADMEは評価インフラ説明であり、人間結果の主張ではありません。
