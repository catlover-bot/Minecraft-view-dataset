# LLM-authored 10-case Diagnostic Summary

- This is a diagnostic validation set. It is separate from main/supplementary/execution-gap benchmark reports.
- provider_tag: `mock_mock_model`

## Description
- auto_score: 5.83%
- strict_material_f1: 0.00%
- coarse_material_f1: 0.00%
- dimension_score: 7.32%

## Rebuild
- Direct (description -> plan -> render):
  IoU=17.36%, F1=28.87%, material=22.54%, correct=17.30%, repair_edit=1.0182
- Structured (description -> structured IR -> deterministic plan -> render):
  IoU=19.26%, F1=31.46%, material=70.35%, correct=50.13%, repair_edit=0.9372

## Direct vs Structured delta (structured - direct)
- IoU: +1.90 pt
- F1: +2.59 pt
- material_match: +47.81 pt
- correct_placement: +32.83 pt
- repair_edit_distance: -0.0811

## Notes
- Human kit is protocol-only; no human outcomes are claimed.
- Interpretation should remain diagnostic; do not use these numbers as headline benchmark replacement.
