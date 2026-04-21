# LLM-authored 10-case Diagnostic Summary

- This is a diagnostic validation set. It is separate from main/supplementary/execution-gap benchmark reports.
- provider_tag: `anthropic_claude_haiku_4_5_20251001`

## Description
- auto_score: 66.97%
- strict_material_f1: 37.62%
- coarse_material_f1: 59.29%
- dimension_score: 83.47%

## Rebuild
- Direct (description -> plan -> render):
  IoU=32.07%, F1=47.90%, material=17.84%, correct=7.29%, repair_edit=1.9804
- Structured (description -> structured IR -> deterministic plan -> render):
  IoU=29.23%, F1=44.37%, material=34.65%, correct=11.84%, repair_edit=2.1263

## Direct vs Structured delta (structured - direct)
- IoU: -2.83 pt
- F1: -3.54 pt
- material_match: +16.81 pt
- correct_placement: +4.55 pt
- repair_edit_distance: +0.1460

## Notes
- Human kit is protocol-only; no human outcomes are claimed.
- Interpretation should remain diagnostic; do not use these numbers as headline benchmark replacement.
