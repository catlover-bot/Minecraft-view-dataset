# llm_case_005 (medium)

- provider_tag: `anthropic_claude_haiku_4_5_20251001`
- source_spec: `llm_case_005/source_spec.json`
- description: `llm_case_005/description_direct/description.json`
- structured_ir: `llm_case_005/structured_intermediate/intermediate.json`

## Description
- auto_score: 58.56%
- strict_material_f1: 25.00%
- coarse_material_f1: 40.00%
- dimension_score: 84.24%

## Rebuild Comparison
- direct IoU/F1/material/correct: 32.66% / 49.24% / 6.19% / 2.92%
- structured IoU/F1/material/correct: 35.88% / 52.81% / 16.89% / 7.68%

## Repair Effort
- direct normalized_edit_distance: 1.5427
- structured normalized_edit_distance: 1.6491
- direct edits(add/del/rep): 614 overlap ref, 580/686/576
- structured edits(add/del/rep): 752 overlap ref, 442/902/625

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
