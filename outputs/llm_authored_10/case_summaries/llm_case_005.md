# llm_case_005 (medium)

- provider_tag: `openai_gpt_5_mini`
- source_spec: `llm_case_005/source_spec.json`
- description: `llm_case_005/description_direct/description.json`
- structured_ir: `llm_case_005/structured_intermediate/intermediate.json`

## Description
- auto_score: 80.40%
- strict_material_f1: 54.55%
- coarse_material_f1: 85.71%
- dimension_score: 90.44%

## Rebuild Comparison
- direct IoU/F1/material/correct: 33.77% / 50.49% / 4.53% / 1.95%
- structured IoU/F1/material/correct: 23.06% / 37.48% / 15.65% / 5.12%

## Repair Effort
- direct normalized_edit_distance: 1.7781
- structured normalized_edit_distance: 1.8342
- direct edits(add/del/rep): 728 overlap ref, 466/962/695
- structured edits(add/del/rep): 524 overlap ref, 670/1078/442

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
