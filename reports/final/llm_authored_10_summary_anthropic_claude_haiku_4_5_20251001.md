# LLM-authored 10-case Diagnostic Summary

- This is a diagnostic validation set. It is separate from main/supplementary/execution-gap benchmark reports.
- provider_tag: `anthropic_claude_haiku_4_5_20251001`

## Description
- auto_score: 75.61%
- strict_material_f1: 60.82%
- coarse_material_f1: 77.53%
- dimension_score: 71.93%

## Rebuild
- Direct (description -> plan -> render):
  IoU=25.56%, F1=39.88%, material=11.06%, correct=6.36%, repair_edit=1.4617
- Structured (description -> structured IR -> deterministic plan -> render):
  IoU=27.86%, F1=42.45%, material=44.19%, correct=25.62%, repair_edit=1.3014

## Direct vs Structured delta (structured - direct)
- IoU: +2.30 pt
- F1: +2.58 pt
- material_match: +33.13 pt
- correct_placement: +19.26 pt
- repair_edit_distance: -0.1603

## Notes
- Human kit is protocol-only; no human outcomes are claimed.
- Interpretation should remain diagnostic; do not use these numbers as headline benchmark replacement.
