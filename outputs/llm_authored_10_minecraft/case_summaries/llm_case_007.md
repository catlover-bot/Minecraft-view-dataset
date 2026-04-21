# llm_case_007 (medium)

- provider_tag: `anthropic_claude_haiku_4_5_20251001`
- source_spec: `llm_case_007/source_spec.json`
- description: `llm_case_007/description_direct/description.json`
- structured_ir: `llm_case_007/structured_intermediate/intermediate.json`

## Description
- auto_score: 59.42%
- strict_material_f1: 40.00%
- coarse_material_f1: 57.14%
- dimension_score: 52.54%

## Rebuild Comparison
- direct IoU/F1/material/correct: 13.94% / 24.47% / 0.21% / 0.08%
- structured IoU/F1/material/correct: 15.09% / 26.22% / 31.89% / 16.59%

## Repair Effort
- direct normalized_edit_distance: 1.2976
- structured normalized_edit_distance: 1.1087
- direct edits(add/del/rep): 476 overlap ref, 2155/784/475
- structured edits(add/del/rep): 461 overlap ref, 2180/435/302

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
