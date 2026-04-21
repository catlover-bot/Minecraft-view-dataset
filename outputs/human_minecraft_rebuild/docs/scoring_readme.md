# Scoring README (Minecraft-native human submissions)

## 1) Convert Minecraft-native submission artifacts
```bash
python3 tools/convert_human_minecraft_submissions.py   --cases_manifest /Users/hirotaka-m/Minecraft-view-dataset/reports/final/original_benchmark_human_minecraft_rebuild_cases.json   --submission_root outputs/human_minecraft_rebuild/submissions   --out_root outputs/human_minecraft_rebuild/converted_submissions
```

## 2) Score converted submissions
```bash
python3 tools/score_human_image_rebuild_submissions.py   --cases_manifest /Users/hirotaka-m/Minecraft-view-dataset/reports/final/original_benchmark_human_minecraft_rebuild_cases.json   --submission_root outputs/human_minecraft_rebuild/converted_submissions   --out_root outputs/human_minecraft_rebuild/scored_submissions
```

出力はインフラ検証目的。人間成績の主張には直接使いません。
