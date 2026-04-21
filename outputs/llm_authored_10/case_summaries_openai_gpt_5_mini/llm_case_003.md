# llm_case_003 (simple)

- provider_tag: `openai_gpt_5_mini`
- source_spec: `llm_case_003/source_spec.json`
- description: `llm_case_003/description_direct/description.json`
- structured_ir: `llm_case_003/structured_intermediate/intermediate.json`

## Description
- auto_score: 57.90%
- strict_material_f1: 40.00%
- coarse_material_f1: 66.67%
- dimension_score: 36.94%

## Rebuild Comparison
- direct IoU/F1/material/correct: 27.52% / 43.17% / 26.04% / 8.85%
- structured IoU/F1/material/correct: 16.72% / 28.66% / 32.64% / 5.63%

## Repair Effort
- direct normalized_edit_distance: 2.0455
- structured normalized_edit_distance: 4.7832
- direct edits(add/del/rep): 169 overlap ref, 131/342/112
- structured edits(add/del/rep): 242 overlap ref, 44/1161/163

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
