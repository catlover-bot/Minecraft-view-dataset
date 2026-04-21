# llm_case_002 (simple)

- provider_tag: `anthropic_claude_haiku_4_5_20251001`
- source_spec: `llm_case_002/source_spec.json`
- description: `llm_case_002/description_direct/description.json`
- structured_ir: `llm_case_002/structured_intermediate/intermediate.json`

## Description
- auto_score: 77.17%
- strict_material_f1: 44.44%
- coarse_material_f1: 85.71%
- dimension_score: 89.63%

## Rebuild Comparison
- direct IoU/F1/material/correct: 41.66% / 58.81% / 26.26% / 12.84%
- structured IoU/F1/material/correct: 42.87% / 60.01% / 58.26% / 26.22%

## Repair Effort
- direct normalized_edit_distance: 1.5773
- structured normalized_edit_distance: 1.5753
- direct edits(add/del/rep): 377 overlap ref, 134/394/278
- structured edits(add/del/rep): 460 overlap ref, 51/562/192

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
