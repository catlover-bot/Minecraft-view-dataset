# LLM-authored 10-case Diagnostic Summary

- This is a diagnostic validation set. It is separate from main/supplementary/execution-gap benchmark reports.
- provider_tag: `openai_gpt_5_mini`

## Description
- auto_score: 72.71%
- strict_material_f1: 58.63%
- coarse_material_f1: 83.12%
- dimension_score: 57.35%

## Rebuild
- Direct (description -> plan -> render):
  IoU=22.47%, F1=36.22%, material=18.00%, correct=4.77%, repair_edit=3.0427
- Structured (description -> structured IR -> deterministic plan -> render):
  IoU=20.49%, F1=33.47%, material=30.49%, correct=7.71%, repair_edit=3.0045

## Direct vs Structured delta (structured - direct)
- IoU: -1.98 pt
- F1: -2.75 pt
- material_match: +12.49 pt
- correct_placement: +2.95 pt
- repair_edit_distance: -0.0382

## Notes
- Human kit is protocol-only; no human outcomes are claimed.
- Interpretation should remain diagnostic; do not use these numbers as headline benchmark replacement.
