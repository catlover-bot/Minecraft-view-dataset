# llm_case_010 (complex)

- provider_tag: `openai_gpt_5_mini`
- source_spec: `llm_case_010/source_spec.json`
- description: `llm_case_010/description_direct/description.json`
- structured_ir: `llm_case_010/structured_intermediate/intermediate.json`

## Description
- auto_score: 82.27%
- strict_material_f1: 72.73%
- coarse_material_f1: 85.71%
- dimension_score: 76.11%

## Rebuild Comparison
- direct IoU/F1/material/correct: 26.49% / 41.89% / 14.81% / 5.53%
- structured IoU/F1/material/correct: 32.73% / 49.32% / 39.51% / 16.50%

## Repair Effort
- direct normalized_edit_distance: 1.7283
- structured normalized_edit_distance: 1.6024
- direct edits(add/del/rep): 905 overlap ref, 994/1517/771
- structured edits(add/del/rep): 1144 overlap ref, 755/1596/692

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
