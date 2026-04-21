# llm_case_004 (medium)

- provider_tag: `openai_gpt_5_mini`
- source_spec: `llm_case_004/source_spec.json`
- description: `llm_case_004/description_direct/description.json`
- structured_ir: `llm_case_004/structured_intermediate/intermediate.json`

## Description
- auto_score: 67.68%
- strict_material_f1: 60.00%
- coarse_material_f1: 75.00%
- dimension_score: 43.72%

## Rebuild Comparison
- direct IoU/F1/material/correct: 21.52% / 35.42% / 2.28% / 0.62%
- structured IoU/F1/material/correct: 25.58% / 40.74% / 64.11% / 17.56%

## Repair Effort
- direct normalized_edit_distance: 2.3632
- structured normalized_edit_distance: 2.5979
- direct edits(add/del/rep): 394 overlap ref, 377/1060/385
- structured edits(add/del/rep): 613 overlap ref, 158/1625/220

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
