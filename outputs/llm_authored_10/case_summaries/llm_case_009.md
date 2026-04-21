# llm_case_009 (complex)

- provider_tag: `openai_gpt_5_mini`
- source_spec: `llm_case_009/source_spec.json`
- description: `llm_case_009/description_direct/description.json`
- structured_ir: `llm_case_009/structured_intermediate/intermediate.json`

## Description
- auto_score: 71.62%
- strict_material_f1: 60.00%
- coarse_material_f1: 85.71%
- dimension_score: 48.77%

## Rebuild Comparison
- direct IoU/F1/material/correct: 17.09% / 29.19% / 18.51% / 3.22%
- structured IoU/F1/material/correct: 11.62% / 20.82% / 36.62% / 4.95%

## Repair Effort
- direct normalized_edit_distance: 5.1023
- structured normalized_edit_distance: 3.7344
- direct edits(add/del/rep): 1329 overlap ref, 147/6301/1083
- structured edits(add/del/rep): 669 overlap ref, 807/4281/424

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
