# llm_case_001 (simple)

- provider_tag: `openai_gpt_5_mini`
- source_spec: `llm_case_001/source_spec.json`
- description: `llm_case_001/description_direct/description.json`
- structured_ir: `llm_case_001/structured_intermediate/intermediate.json`

## Description
- auto_score: 66.58%
- strict_material_f1: 44.44%
- coarse_material_f1: 80.00%
- dimension_score: 52.98%

## Rebuild Comparison
- direct IoU/F1/material/correct: 15.99% / 27.57% / 18.01% / 2.88%
- structured IoU/F1/material/correct: 18.63% / 31.41% / 32.75% / 6.54%

## Repair Effort
- direct normalized_edit_distance: 6.0740
- structured normalized_edit_distance: 3.7106
- direct edits(add/del/rep): 311 overlap ref, 0/1634/255
- structured edits(add/del/rep): 229 overlap ref, 82/918/154

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
