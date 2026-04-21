# llm_case_006 (medium)

- provider_tag: `anthropic_claude_haiku_4_5_20251001`
- source_spec: `llm_case_006/source_spec.json`
- description: `llm_case_006/description_direct/description.json`
- structured_ir: `llm_case_006/structured_intermediate/intermediate.json`

## Description
- auto_score: 79.72%
- strict_material_f1: 72.73%
- coarse_material_f1: 88.89%
- dimension_score: 62.70%

## Rebuild Comparison
- direct IoU/F1/material/correct: 27.87% / 43.59% / 13.83% / 7.57%
- structured IoU/F1/material/correct: 29.08% / 45.06% / 44.56% / 26.13%

## Repair Effort
- direct normalized_edit_distance: 1.1977
- structured normalized_edit_distance: 1.0950
- direct edits(add/del/rep): 282 overlap ref, 529/265/139
- structured edits(add/del/rep): 285 overlap ref, 494/201/158

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
