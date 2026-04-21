# llm_case_002 (simple)

- provider_tag: `openai_gpt_5_mini`
- source_spec: `llm_case_002/source_spec.json`
- description: `llm_case_002/description_direct/description.json`
- structured_ir: `llm_case_002/structured_intermediate/intermediate.json`

## Description
- auto_score: 68.93%
- strict_material_f1: 50.00%
- coarse_material_f1: 85.71%
- dimension_score: 50.00%

## Rebuild Comparison
- direct IoU/F1/material/correct: 23.70% / 38.32% / 27.05% / 7.80%
- structured IoU/F1/material/correct: 17.58% / 29.90% / 58.39% / 12.10%

## Repair Effort
- direct normalized_edit_distance: 2.2564
- structured normalized_edit_distance: 2.7378
- direct edits(add/del/rep): 292 overlap ref, 219/721/213
- structured edits(add/del/rep): 274 overlap ref, 237/1048/114

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
