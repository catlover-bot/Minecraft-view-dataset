# llm_case_008 (complex)

- provider_tag: `openai_gpt_5_mini`
- source_spec: `llm_case_008/source_spec.json`
- description: `llm_case_008/description_direct/description.json`
- structured_ir: `llm_case_008/structured_intermediate/intermediate.json`

## Description
- auto_score: 89.70%
- strict_material_f1: 83.33%
- coarse_material_f1: 100.00%
- dimension_score: 78.80%

## Rebuild Comparison
- direct IoU/F1/material/correct: 29.91% / 46.04% / 19.91% / 8.90%
- structured IoU/F1/material/correct: 30.17% / 46.36% / 21.32% / 7.96%

## Repair Effort
- direct normalized_edit_distance: 1.4932
- structured normalized_edit_distance: 1.8949
- direct edits(add/del/rep): 1527 overlap ref, 1689/1890/1223
- structured edits(add/del/rep): 1965 overlap ref, 1251/3297/1546

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
