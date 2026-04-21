# llm_case_003 (simple)

- provider_tag: `anthropic_claude_haiku_4_5_20251001`
- source_spec: `llm_case_003/source_spec.json`
- description: `llm_case_003/description_direct/description.json`
- structured_ir: `llm_case_003/structured_intermediate/intermediate.json`

## Description
- auto_score: 65.96%
- strict_material_f1: 40.00%
- coarse_material_f1: 40.00%
- dimension_score: 95.83%

## Rebuild Comparison
- direct IoU/F1/material/correct: 30.99% / 47.32% / 15.46% / 7.61%
- structured IoU/F1/material/correct: 24.27% / 39.06% / 2.82% / 0.78%

## Repair Effort
- direct normalized_edit_distance: 1.3991
- structured normalized_edit_distance: 2.7277
- direct edits(add/del/rep): 97 overlap ref, 116/100/82
- structured edits(add/del/rep): 142 overlap ref, 71/372/138

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
