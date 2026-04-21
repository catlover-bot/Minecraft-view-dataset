# LLM-authored 10-case Diagnostic Summary

- This is a diagnostic validation set. It is separate from main/supplementary/execution-gap benchmark reports.
- provider_tag: `openai_gpt_5_mini`

## Description
- auto_score: 80.62%
- strict_material_f1: 72.52%
- coarse_material_f1: 88.95%
- dimension_score: 66.50%

## Rebuild
- Direct (description -> plan -> render):
  IoU=29.16%, F1=44.56%, material=23.79%, correct=13.24%, repair_edit=1.3346
- Structured (description -> structured IR -> deterministic plan -> render):
  IoU=32.69%, F1=48.82%, material=51.40%, correct=33.08%, repair_edit=1.1261

## Direct vs Structured delta (structured - direct)
- IoU: +3.53 pt
- F1: +4.26 pt
- material_match: +27.61 pt
- correct_placement: +19.84 pt
- repair_edit_distance: -0.2084

## Notes
- Human kit is protocol-only; no human outcomes are claimed.
- Interpretation should remain diagnostic; do not use these numbers as headline benchmark replacement.
