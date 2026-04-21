# llm_case_007 (medium)

- provider_tag: `openai_gpt_5_mini`
- source_spec: `llm_case_007/source_spec.json`
- description: `llm_case_007/description_direct/description.json`
- structured_ir: `llm_case_007/structured_intermediate/intermediate.json`

## Description
- auto_score: 69.01%
- strict_material_f1: 66.67%
- coarse_material_f1: 66.67%
- dimension_score: 49.37%

## Rebuild Comparison
- direct IoU/F1/material/correct: 13.39% / 23.61% / 43.00% / 6.95%
- structured IoU/F1/material/correct: 17.93% / 30.40% / 3.85% / 0.76%

## Repair Effort
- direct normalized_edit_distance: 3.0795
- structured normalized_edit_distance: 3.6235
- direct edits(add/del/rep): 1128 overlap ref, 1451/5848/643
- structured edits(add/del/rep): 1687 overlap ref, 892/6831/1622

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
