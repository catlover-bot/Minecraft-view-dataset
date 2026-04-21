# llm_case_006 (medium)

- provider_tag: `openai_gpt_5_mini`
- source_spec: `llm_case_006/source_spec.json`
- description: `llm_case_006/description_direct/description.json`
- structured_ir: `llm_case_006/structured_intermediate/intermediate.json`

## Description
- auto_score: 72.96%
- strict_material_f1: 54.55%
- coarse_material_f1: 100.00%
- dimension_score: 46.37%

## Rebuild Comparison
- direct IoU/F1/material/correct: 15.29% / 26.52% / 5.86% / 0.96%
- structured IoU/F1/material/correct: 10.86% / 19.60% / 0.00% / 0.00%

## Repair Effort
- direct normalized_edit_distance: 4.5068
- structured normalized_edit_distance: 3.5264
- direct edits(add/del/rep): 461 overlap ref, 202/2352/434
- structured edits(add/del/rep): 254 overlap ref, 409/1675/254

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
